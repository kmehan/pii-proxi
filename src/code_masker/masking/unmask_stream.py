"""SSE-aware byte-stream transform: placeholder → original.

The upstream response streams as bytes (SSE ``data:`` lines for Anthropic and
OpenAI). A placeholder like ``⟦EMAIL_a1b2c3d4⟧`` can straddle a chunk boundary,
so we tail-buffer a short suffix until we're sure it can't start an open
placeholder that we haven't seen the close of yet.

We scan for placeholders in **two forms** because upstream servers vary in
whether they ASCII-escape non-ASCII string content inside JSON payloads:

    raw      ⟦EMAIL_a1b2c3d4⟧                 (3-byte UTF-8 delimiters)
    escaped  \\u27e6EMAIL_a1b2c3d4\\u27e7     (6-byte ASCII escape form)

``json.dumps`` defaults to ``ensure_ascii=True``, so the escaped form is the
common case when the upstream re-encodes a JSON body that contained our
delimiters. If we only scanned for the raw form we'd silently leak
placeholders back to the client — a correctness bug that defeats the proxy's
whole purpose — so both forms are required.

When we substitute in ASCII-escape context we re-escape the original with
``ensure_ascii=True`` to keep the surrounding JSON string valid.

Upper-bound math for the tail-safe buffer:

    raw     len(⟦) [3B] + LABEL [<=32 ASCII] + '_' + 8 hex + len(⟧) [3B]  = 47 B
    escaped len(\\u27e6) [6B] + LABEL + '_' + 8 hex + len(\\u27e7) [6B]    = 53 B

We use a 256-byte cap on unterminated opens to avoid stalling pathological
streams.
"""

from __future__ import annotations

import json

from .placeholder import DELIM_CLOSE, DELIM_OPEN, PlaceholderMap


_OPEN_RAW = DELIM_OPEN.encode("utf-8")
_CLOSE_RAW = DELIM_CLOSE.encode("utf-8")
_OPEN_ESC = b"\\u27e6"
_CLOSE_ESC = b"\\u27e7"

# (open_bytes, close_bytes, is_escape) for each form we scan for.
_FORMS: tuple[tuple[bytes, bytes, bool], ...] = (
    (_OPEN_RAW, _CLOSE_RAW, False),
    (_OPEN_ESC, _CLOSE_ESC, True),
)

_MAX_OPEN_PREFIX = max(len(_OPEN_RAW), len(_OPEN_ESC))
_MAX_OPEN_WAIT = 256


class UnmaskStream:
    """Stateful bytes-in / bytes-out transform.

    Usage::

        u = UnmaskStream(pmap)
        for chunk in upstream:
            client.write(u.feed(chunk))
        client.write(u.flush())
    """

    __slots__ = ("_pmap", "_buf")

    def __init__(self, pmap: PlaceholderMap) -> None:
        self._pmap = pmap
        self._buf = b""

    def feed(self, chunk: bytes) -> bytes:
        if not chunk:
            return b""
        self._buf += chunk
        return self._drain(final=False)

    def flush(self) -> bytes:
        out = self._drain(final=True)
        self._buf = b""
        return out

    # ------------------------------------------------------------------

    def _drain(self, final: bool) -> bytes:
        out = bytearray()
        buf = self._buf
        i = 0

        while True:
            hit = _earliest_open(buf, i)
            if hit is None:
                if final:
                    out += buf[i:]
                    i = len(buf)
                else:
                    tail = _safe_tail(buf, i)
                    out += buf[i:tail]
                    i = tail
                break

            open_idx, open_bytes, close_bytes, is_escape = hit
            out += buf[i:open_idx]

            close_idx = buf.find(close_bytes, open_idx + len(open_bytes))
            if close_idx < 0:
                remaining = len(buf) - open_idx
                if final or remaining > _MAX_OPEN_WAIT:
                    out += open_bytes
                    i = open_idx + len(open_bytes)
                    continue
                i = open_idx
                break

            inner_bytes = buf[open_idx + len(open_bytes) : close_idx]
            # Both forms carry ASCII-only inner content (label + '_' + 8 hex),
            # so UTF-8 decode is effectively an ASCII check.
            try:
                inner = inner_bytes.decode("utf-8")
            except UnicodeDecodeError:
                out += open_bytes
                i = open_idx + len(open_bytes)
                continue

            # Match against the canonical raw-form pattern.
            raw_text = DELIM_OPEN + inner + DELIM_CLOSE
            if self._pmap.pattern.fullmatch(raw_text):
                original = self._pmap.unmask(raw_text)
                if original is not None:
                    if is_escape:
                        # Re-serialize with ensure_ascii so the surrounding
                        # JSON string stays syntactically valid regardless of
                        # what the original contained.
                        out += json.dumps(original, ensure_ascii=True)[1:-1].encode("ascii")
                    else:
                        out += original.encode("utf-8")
                else:
                    # Shape-correct but unminted placeholder — leave intact.
                    out += buf[open_idx : close_idx + len(close_bytes)]
                i = close_idx + len(close_bytes)
            else:
                out += open_bytes
                i = open_idx + len(open_bytes)

        self._buf = buf[i:]
        return bytes(out)


def _earliest_open(buf: bytes, start: int) -> tuple[int, bytes, bytes, bool] | None:
    """Return the earliest open-marker hit at or after ``start``, or None."""
    best: tuple[int, bytes, bytes, bool] | None = None
    for open_bytes, close_bytes, is_escape in _FORMS:
        idx = buf.find(open_bytes, start)
        if idx < 0:
            continue
        if best is None or idx < best[0]:
            best = (idx, open_bytes, close_bytes, is_escape)
    return best


def _safe_tail(buf: bytes, start: int) -> int:
    """Index up to which it's safe to emit.

    Holds back up to ``_MAX_OPEN_PREFIX - 1`` trailing bytes in case they form
    the start of either an escape-form or a raw-form open delimiter whose
    remainder arrives in the next feed.
    """
    keep = _MAX_OPEN_PREFIX - 1
    if keep <= 0:
        return len(buf)
    end = len(buf)
    cutoff = max(start, end - keep)
    for i in range(cutoff, end):
        suffix = buf[i:end]
        for open_bytes, _, _ in _FORMS:
            if open_bytes.startswith(suffix):
                return i
    return end
