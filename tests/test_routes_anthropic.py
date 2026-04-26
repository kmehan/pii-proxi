"""Integration tests for the Anthropic route family.

Upstream is mocked with ``respx`` so we verify two things at once:
    1. What the proxy sends **to** the upstream (masked body, auth headers
       forwarded verbatim).
    2. What the proxy sends **back to the client** after unmasking the
       upstream response.
"""

from __future__ import annotations

import json
import re

import httpx
import respx
from fastapi.testclient import TestClient

from pii_proxi.config import Config
from pii_proxi.masking.placeholder import PlaceholderMap
from pii_proxi.server import create_app
from pii_proxi.session import new_session_key

from .conftest import FakeDetector, FakeSpan


UPSTREAM = "https://api.anthropic.com"
SECRET = "sk-live-AAAABBBB"
PROMPT = f"here is my key {SECRET} — write a curl"


def _make_app(detector: FakeDetector, pmap: PlaceholderMap | None = None):
    cfg = Config(anthropic_upstream=UPSTREAM, openai_upstream="https://api.openai.com")
    # Own HTTP client so respx intercepts it; create_app would otherwise
    # build one in lifespan but that's fine too — respx patches httpx
    # globally via its mock transport.
    return create_app(config=cfg, detector=detector, placeholder_map=pmap)


def test_non_streaming_masks_request_and_unmasks_response():
    secret_start = PROMPT.index(SECRET)
    detector = FakeDetector({
        PROMPT: [FakeSpan(secret_start, secret_start + len(SECRET), "SECRET")],
    })
    pmap = PlaceholderMap(new_session_key())
    app = _make_app(detector, pmap=pmap)

    placeholder_re = re.compile(r"⟦SECRET_[0-9a-f]{8}⟧")

    with respx.mock(assert_all_called=True) as mock:
        route = mock.post(f"{UPSTREAM}/v1/messages").mock(
            return_value=httpx.Response(
                200,
                # Upstream "echoes" the placeholder in its reply; the proxy
                # must unmask it before the client sees the response.
                json={"content": [{"type": "text", "text": "placeholder was: ⟦SECRET_PLACEHOLDER⟧"}]},
            )
        )

        with TestClient(app) as client:
            # Mint the placeholder up-front so we know what token to embed in
            # the mocked upstream reply.
            ph = pmap.mask(SECRET, "SECRET")
            # Rewire the upstream mock to return the *actual* minted
            # placeholder so unmask_stream maps it back to SECRET.
            route.mock(
                return_value=httpx.Response(
                    200,
                    json={"content": [{"type": "text", "text": f"use {ph} please"}]},
                )
            )

            resp = client.post(
                "/anthropic/v1/messages",
                headers={
                    "x-api-key": "sk-ant-fake",
                    "anthropic-version": "2023-06-01",
                    "authorization": "Bearer oauth-token-xyz",
                    "content-type": "application/json",
                },
                json={
                    "model": "claude-3-opus-20240229",
                    "max_tokens": 128,
                    "messages": [{"role": "user", "content": PROMPT}],
                },
            )

    assert resp.status_code == 200

    # 1. What went upstream: body has placeholder, not plaintext secret.
    sent = json.loads(route.calls.last.request.content)
    assert SECRET not in json.dumps(sent)
    assert placeholder_re.search(sent["messages"][0]["content"]) is not None

    # Auth headers relayed verbatim.
    assert route.calls.last.request.headers["x-api-key"] == "sk-ant-fake"
    assert route.calls.last.request.headers["authorization"] == "Bearer oauth-token-xyz"
    assert route.calls.last.request.headers["anthropic-version"] == "2023-06-01"

    # 2. What came back to the client: unmasked.
    body = resp.json()
    text_out = body["content"][0]["text"]
    assert SECRET in text_out
    assert "⟦" not in text_out


def test_streaming_sse_unmasks_placeholders_back_to_plaintext():
    secret_start = PROMPT.index(SECRET)
    detector = FakeDetector({
        PROMPT: [FakeSpan(secret_start, secret_start + len(SECRET), "SECRET")],
    })
    pmap = PlaceholderMap(new_session_key())
    app = _make_app(detector, pmap=pmap)

    # Pre-mint so we know exactly what bytes the "upstream" sends.
    ph = pmap.mask(SECRET, "SECRET")

    sse_body = (
        b"event: content_block_delta\n"
        b'data: {"delta":{"text":"run: curl -H \\"x-api-key: ' + ph.encode() + b'\\""}}\n\n'
        b"event: message_stop\ndata: {}\n\n"
    )

    with respx.mock(assert_all_called=True) as mock:
        route = mock.post(f"{UPSTREAM}/v1/messages").mock(
            return_value=httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=sse_body,
            )
        )

        with TestClient(app) as client:
            resp = client.post(
                "/anthropic/v1/messages",
                headers={
                    "x-api-key": "sk-ant-fake",
                    "anthropic-version": "2023-06-01",
                    "accept": "text/event-stream",
                    "content-type": "application/json",
                },
                json={
                    "model": "claude-3-opus-20240229",
                    "max_tokens": 128,
                    "stream": True,
                    "messages": [{"role": "user", "content": PROMPT}],
                },
            )

    assert resp.status_code == 200
    # Upstream got masked body:
    sent = json.loads(route.calls.last.request.content)
    assert SECRET not in json.dumps(sent)
    # Client got unmasked text:
    received = resp.content
    assert SECRET.encode() in received
    assert ph.encode() not in received


def test_streaming_unmasks_lowercase_label_placeholder():
    """Regression: detector emits lowercase labels (``private_person``,
    ``private_email``...). An older placeholder regex required uppercase and
    silently leaked these back to the client. End-to-end check using the same
    label shape the real model emits, with synthetic data only.
    """
    sample = "Ada Lovelace"
    prompt = f"My name is {sample}. Who am I?"
    name_start = prompt.index(sample)
    detector = FakeDetector({
        prompt: [FakeSpan(name_start, name_start + len(sample), "private_person")],
    })
    pmap = PlaceholderMap(new_session_key())
    app = _make_app(detector, pmap=pmap)

    ph = pmap.mask(sample, "private_person")
    assert "⟦private_person_" in ph  # sanity: lowercase label preserved in placeholder

    sse_body = (
        b"event: content_block_delta\n"
        b'data: {"delta":{"text":"You said your name is ' + ph.encode() + b'."}}\n\n'
        b"event: message_stop\ndata: {}\n\n"
    )

    with respx.mock(assert_all_called=True) as mock:
        mock.post(f"{UPSTREAM}/v1/messages").mock(
            return_value=httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=sse_body,
            )
        )
        with TestClient(app) as client:
            resp = client.post(
                "/anthropic/v1/messages",
                headers={
                    "x-api-key": "sk-ant-fake",
                    "anthropic-version": "2023-06-01",
                    "accept": "text/event-stream",
                    "content-type": "application/json",
                },
                json={
                    "model": "claude-3-opus-20240229",
                    "max_tokens": 128,
                    "stream": True,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )

    assert resp.status_code == 200
    received = resp.content
    assert sample.encode() in received, "client should see the unmasked plaintext"
    assert b"private_person_" not in received, "no placeholder bytes should leak"
    assert "⟦".encode() not in received


def test_streaming_unmasks_placeholder_split_across_events():
    """Regression: real upstream tokenizers fragment a placeholder across
    multiple ``content_block_delta`` events. Each fragment lands in its own
    JSON ``text`` field, separated by SSE/JSON framing bytes that aren't part
    of the logical text stream. A byte-level scan saw ``⟦…⟧`` enclosing the
    framing and gave up, leaking placeholders to the client. The unmasker must
    reconstruct the per-block text stream and substitute even when the
    placeholder spans events.
    """
    sample = "Ada Lovelace"
    prompt = f"My name is {sample}. Who am I?"
    name_start = prompt.index(sample)
    detector = FakeDetector({
        prompt: [FakeSpan(name_start, name_start + len(sample), "private_person")],
    })
    pmap = PlaceholderMap(new_session_key())
    app = _make_app(detector, pmap=pmap)

    ph = pmap.mask(sample, "private_person")
    # Split the placeholder across three text_delta events. The bytes between
    # the open ⟦ and close ⟧ are SSE/JSON framing — not the placeholder body.
    third = len(ph) // 3
    chunk1 = ph[:third]
    chunk2 = ph[third : 2 * third]
    chunk3 = ph[2 * third :]

    sse_body = (
        b"event: content_block_start\n"
        b'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n\n'
        b"event: content_block_delta\n"
        b'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"You said your name is '
        + chunk1.encode()
        + b'"}}\n\n'
        b"event: content_block_delta\n"
        b'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"'
        + chunk2.encode()
        + b'"}}\n\n'
        b"event: content_block_delta\n"
        b'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"'
        + chunk3.encode()
        + b'."}}\n\n'
        b"event: content_block_stop\n"
        b'data: {"type":"content_block_stop","index":0}\n\n'
        b"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"
    )

    with respx.mock(assert_all_called=True) as mock:
        mock.post(f"{UPSTREAM}/v1/messages").mock(
            return_value=httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=sse_body,
            )
        )
        with TestClient(app) as client:
            resp = client.post(
                "/anthropic/v1/messages",
                headers={
                    "x-api-key": "sk-ant-fake",
                    "anthropic-version": "2023-06-01",
                    "accept": "text/event-stream",
                    "content-type": "application/json",
                },
                json={
                    "model": "claude-3-opus-20240229",
                    "max_tokens": 128,
                    "stream": True,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )

    assert resp.status_code == 200
    received = resp.content
    # Decode every text_delta the client would render and concatenate.
    rendered_parts: list[str] = []
    for raw_event in received.split(b"\n\n"):
        for line in raw_event.splitlines():
            if not line.startswith(b"data: "):
                continue
            payload = line[len(b"data: "):]
            try:
                data = json.loads(payload)
            except json.JSONDecodeError:
                continue
            delta = data.get("delta") if isinstance(data, dict) else None
            if isinstance(delta, dict) and isinstance(delta.get("text"), str):
                rendered_parts.append(delta["text"])
    rendered = "".join(rendered_parts)
    assert sample in rendered, f"client should see plaintext, got: {rendered!r}"
    assert "private_person_" not in rendered
    assert "⟦" not in rendered


def test_upstream_with_subpath_prefix_concatenates_correctly():
    """DeepSeek's Anthropic-compatible endpoint sits at
    ``https://api.deepseek.com/anthropic``, i.e. the configured upstream
    already has a path component. The route must append ``/v1/messages``
    onto that prefix rather than replacing it.
    """
    deepseek_upstream = "https://api.deepseek.com/anthropic"
    detector = FakeDetector()
    cfg = Config(
        anthropic_upstream=deepseek_upstream,
        openai_upstream="https://api.openai.com",
    )
    app = create_app(config=cfg, detector=detector)

    with respx.mock(assert_all_called=True) as mock:
        route = mock.post(f"{deepseek_upstream}/v1/messages").mock(
            return_value=httpx.Response(200, json={"content": []})
        )

        with TestClient(app) as client:
            resp = client.post(
                "/anthropic/v1/messages",
                headers={"x-api-key": "sk-ds-fake", "anthropic-version": "2023-06-01"},
                json={
                    "model": "deepseek-v4-pro",
                    "max_tokens": 16,
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )

    assert resp.status_code == 200
    assert str(route.calls.last.request.url) == f"{deepseek_upstream}/v1/messages"


def test_upstream_error_passes_through_unchanged():
    detector = FakeDetector()
    app = _make_app(detector)

    with respx.mock() as mock:
        mock.post(f"{UPSTREAM}/v1/messages").mock(
            return_value=httpx.Response(
                401,
                json={"error": {"type": "authentication_error", "message": "bad key"}},
            )
        )

        with TestClient(app) as client:
            resp = client.post(
                "/anthropic/v1/messages",
                headers={"x-api-key": "sk-ant-bad", "anthropic-version": "2023-06-01"},
                json={
                    "model": "claude-3-opus-20240229",
                    "max_tokens": 16,
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )

    assert resp.status_code == 401
    body = resp.json()
    assert body["error"]["type"] == "authentication_error"


def test_hop_by_hop_headers_are_stripped_on_forward():
    detector = FakeDetector()
    app = _make_app(detector)

    with respx.mock() as mock:
        route = mock.post(f"{UPSTREAM}/v1/messages").mock(
            return_value=httpx.Response(200, json={"ok": True})
        )

        with TestClient(app) as client:
            client.post(
                "/anthropic/v1/messages",
                headers={
                    "x-api-key": "sk-ant-fake",
                    "connection": "close",
                    "transfer-encoding": "chunked",
                },
                json={"messages": [{"role": "user", "content": "hi"}]},
            )

    # httpx re-adds ``connection`` itself (keep-alive), so assert on a
    # hop-by-hop header that httpx doesn't set and that we strip from the
    # client's input — ``transfer-encoding`` fits both requirements.
    sent_headers = {k.lower() for k in route.calls.last.request.headers.keys()}
    assert "transfer-encoding" not in sent_headers
    # Auth headers survive the strip.
    assert "x-api-key" in sent_headers


def test_healthz():
    detector = FakeDetector()
    app = _make_app(detector)
    with TestClient(app) as client:
        resp = client.get("/healthz")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["detector_loaded"] is True


def test_clear_session_replaces_placeholder_map():
    detector = FakeDetector()
    pmap = PlaceholderMap(new_session_key())
    app = _make_app(detector, pmap=pmap)
    pmap.mask("hello", "SECRET")
    assert len(pmap) == 1

    with TestClient(app) as client:
        resp = client.post("/admin/clear-session")

    assert resp.status_code == 200
    # App-state map was replaced with a fresh one (the original ``pmap``
    # handle above is stale, but the app-scoped map is what routes use).
    assert len(app.state.placeholder_map) == 0
