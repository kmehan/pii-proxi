"""Shared proxy plumbing for the Anthropic and OpenAI route families.

Both families run the same pipeline: extract text → detect → mask → inject
→ forward → unmask. The per-route modules only differ in which extractor
they call and which upstream URL they target.

The proxy is designed to be *transparent* at the auth layer. Every client
header is forwarded verbatim except for a small set the HTTP stack must own
(``host``, ``content-length``, hop-by-hop). That's what makes OAuth
subscription tokens work: we never look at the credential, just relay it.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from typing import Any, Callable, Iterable

import httpx
from fastapi import Request, Response
from fastapi.responses import StreamingResponse

from ..masking.placeholder import PlaceholderMap, Span, apply_spans
from ..masking.injector import inject
from ..masking.unmask_stream import UnmaskStream


_log = logging.getLogger("pii_proxi.mask")


# Hop-by-hop headers per RFC 7230 §6.1, plus a couple of well-known
# connection-management headers. Forwarding these across a proxy hop is
# either meaningless or actively harmful.
_HOP_BY_HOP = frozenset(
    h.lower()
    for h in (
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "trailers",
        "transfer-encoding",
        "upgrade",
    )
)

# Headers the HTTP client layer must compute itself; forwarding the client's
# values would mismatch the re-serialized body.
_CLIENT_OWNED = frozenset({"host", "content-length", "accept-encoding"})

# Response headers we must not forward as-is: encoding the upstream applied
# may no longer match our (potentially re-encoded) body, and framing is our
# responsibility downstream.
_RESPONSE_STRIP = frozenset({"content-length", "content-encoding", "transfer-encoding"})


def forward_request_headers(src: Iterable[tuple[str, str]]) -> list[tuple[str, str]]:
    """Strip hop-by-hop + client-owned headers; keep auth headers verbatim."""
    out: list[tuple[str, str]] = []
    for k, v in src:
        lk = k.lower()
        if lk in _HOP_BY_HOP or lk in _CLIENT_OWNED:
            continue
        out.append((k, v))
    return out


def forward_response_headers(src: Iterable[tuple[str, str]]) -> list[tuple[bytes, bytes]]:
    out: list[tuple[bytes, bytes]] = []
    for k, v in src:
        if k.lower() in _RESPONSE_STRIP:
            continue
        out.append((k.encode("latin-1"), v.encode("latin-1")))
    return out


ExtractFn = Callable[[dict], list[tuple[str, str]]]


async def proxy_roundtrip(
    request: Request,
    upstream_url: str,
    extract: ExtractFn,
) -> Response:
    """Run the full extract/mask/forward/unmask pipeline for one request.

    Returns a FastAPI ``Response`` or ``StreamingResponse``. Upstream errors
    (non-2xx) pass through unchanged — the body carries useful diagnostics the
    user should see, and masking error text risks corrupting structured error
    shapes the client may parse.
    """
    raw = await request.body()

    try:
        body: Any = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        # Not JSON: nothing to mask. Forward raw and relay response.
        return await _passthrough(request, upstream_url, raw)

    if not isinstance(body, dict):
        return await _passthrough(request, upstream_url, raw)

    detector = request.app.state.detector
    pmap: PlaceholderMap = request.app.state.placeholder_map

    cfg = request.app.state.config
    disabled = frozenset(cfg.disabled_labels)
    log_entities = bool(getattr(cfg, "log_entities", False))

    pairs = extract(body)
    masked_pairs: list[tuple[str, str]] = []
    # Aggregate counts only — never the plaintext, unless log_entities is on.
    label_counts: Counter[str] = Counter()
    # (label, plaintext) tuples collected only when log_entities is enabled.
    entity_samples: list[tuple[str, str]] = []
    if pairs:
        texts = [t for _, t in pairs]
        all_spans: list[list[Span]] = detector.detect(texts)
        if len(all_spans) != len(pairs):
            raise RuntimeError(
                f"detector returned {len(all_spans)} span-lists for {len(pairs)} texts"
            )
        for (ptr, text), spans in zip(pairs, all_spans):
            if disabled:
                spans = [s for s in spans if s.label not in disabled]
            label_counts.update(s.label for s in spans)
            if log_entities and spans:
                for s in spans:
                    entity_samples.append((s.label, text[s.start : s.end]))
            masked_text = apply_spans(text, spans, pmap) if spans else text
            masked_pairs.append((ptr, masked_text))

    total = sum(label_counts.values())
    if total:
        breakdown = ", ".join(f"{lbl}={n}" for lbl, n in sorted(label_counts.items()))
        _log.info("masked %d span(s) across %d text(s): %s", total, len(pairs), breakdown)
        if log_entities:
            for label, sample in entity_samples:
                _log.info("  %s: %r", label, sample)
    elif pairs:
        _log.info("no spans detected across %d text(s)", len(pairs))

    new_body = inject(body, masked_pairs) if masked_pairs else body
    forward_bytes = json.dumps(new_body, ensure_ascii=False).encode("utf-8")

    fwd_headers = forward_request_headers(request.headers.items())
    # Ensure the upstream sees the masked content length. httpx will set it
    # from the body length, but we strip the client's value in
    # forward_request_headers anyway.

    wants_stream = _wants_stream(body, request.headers)

    client: httpx.AsyncClient = request.app.state.http_client
    req = client.build_request(
        "POST",
        upstream_url,
        headers=fwd_headers,
        content=forward_bytes,
    )

    if wants_stream:
        return await _stream_response(client, req, pmap)
    return await _buffered_response(client, req, pmap)


async def _passthrough(request: Request, upstream_url: str, raw: bytes) -> Response:
    """Forward the raw body and relay the response without masking.

    Used when the request body isn't JSON-decodable; we still want to act as a
    transparent proxy rather than 400 back to the client.
    """
    client: httpx.AsyncClient = request.app.state.http_client
    fwd_headers = forward_request_headers(request.headers.items())
    req = client.build_request("POST", upstream_url, headers=fwd_headers, content=raw)
    upstream = await client.send(req, stream=False)
    try:
        return Response(
            content=upstream.content,
            status_code=upstream.status_code,
            headers={
                k: v
                for k, v in upstream.headers.items()
                if k.lower() not in _RESPONSE_STRIP
            },
            media_type=upstream.headers.get("content-type"),
        )
    finally:
        await upstream.aclose()


async def _buffered_response(
    client: httpx.AsyncClient,
    req: httpx.Request,
    pmap: PlaceholderMap,
) -> Response:
    upstream = await client.send(req, stream=False)
    try:
        body = upstream.content
        if 200 <= upstream.status_code < 300 and body:
            transform = UnmaskStream(pmap)
            body = transform.feed(body) + transform.flush()
        return Response(
            content=body,
            status_code=upstream.status_code,
            headers={
                k: v
                for k, v in upstream.headers.items()
                if k.lower() not in _RESPONSE_STRIP
            },
            media_type=upstream.headers.get("content-type"),
        )
    finally:
        await upstream.aclose()


async def _stream_response(
    client: httpx.AsyncClient,
    req: httpx.Request,
    pmap: PlaceholderMap,
) -> StreamingResponse:
    # We open the stream inside the generator so the connection stays alive
    # for the whole response lifetime without us having to thread cleanup
    # through a background task.
    upstream = await client.send(req, stream=True)

    if not (200 <= upstream.status_code < 300):
        # Error: collect body and relay unchanged. No masking / unmasking.
        body = await upstream.aread()
        await upstream.aclose()
        return Response(
            content=body,
            status_code=upstream.status_code,
            headers={
                k: v
                for k, v in upstream.headers.items()
                if k.lower() not in _RESPONSE_STRIP
            },
            media_type=upstream.headers.get("content-type"),
        )

    async def gen() -> Any:
        transform = UnmaskStream(pmap)
        try:
            async for chunk in upstream.aiter_bytes():
                out = transform.feed(chunk)
                if out:
                    yield out
            tail = transform.flush()
            if tail:
                yield tail
        finally:
            await upstream.aclose()

    headers = {
        k: v
        for k, v in upstream.headers.items()
        if k.lower() not in _RESPONSE_STRIP
    }
    return StreamingResponse(
        gen(),
        status_code=upstream.status_code,
        headers=headers,
        media_type=upstream.headers.get("content-type"),
    )


def _wants_stream(body: dict, headers: Any) -> bool:
    if body.get("stream") is True:
        return True
    accept = headers.get("accept", "")
    return "text/event-stream" in accept.lower()
