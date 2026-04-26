"""SSE-aware unmasking for streaming proxy responses.

The byte-level :class:`~pii_proxi.masking.unmask_stream.UnmaskStream` recovers
placeholders split across TCP chunks within a single SSE event, but it cannot
recover when the upstream tokenizer fragments a placeholder across multiple
``content_block_delta`` events. Each fragment lands in its own JSON ``text``
field, separated by SSE/JSON framing bytes (``"}}\\n\\ndata: {...,"text":"``)
that aren't part of the logical text stream — a byte-level scan sees ``⟦…⟧``
enclosing JSON syntax, fails the placeholder pattern, and emits the open
delimiter verbatim. End result: the client sees the placeholder.

This module reconstructs the logical text stream per content-block index,
substitutes placeholders that complete across event boundaries, and re-emits
``content_block_delta`` (Anthropic) / ``choices[].delta.content`` (OpenAI)
events carrying substituted text. Structural events (``content_block_start``,
``content_block_stop``, ``message_*``) pass through unchanged — but any held
text is flushed before a structural event for the same block index, so the SSE
protocol stays well-ordered.

We deliberately re-serialize the held event using the most recently observed
text-delta event as a template. That keeps the upstream-chosen ``index``,
event-name, and surrounding JSON shape so downstream SDKs (Anthropic Python
SDK, OpenAI SDK, Claude Code's renderer) parse our output the same way they
parse the upstream's.
"""

from __future__ import annotations

import copy
import json
import re

from .placeholder import DELIM_CLOSE, DELIM_OPEN, PlaceholderMap


__all__ = ["SSEUnmask"]


_PLACEHOLDER_RE = re.compile(
    rf"{re.escape(DELIM_OPEN)}([A-Za-z][A-Za-z0-9_]*)_([0-9a-f]{{8}}){re.escape(DELIM_CLOSE)}"
)

# Cap on how many trailing characters we hold back for a possibly-open
# placeholder. A real placeholder is at most ~50 chars; 256 covers pathological
# label lengths and prevents unbounded buffering on malformed streams.
_MAX_HOLD_CHARS = 256


def _safe_split(text: str) -> tuple[str, str]:
    """Split ``text`` into (safe-to-emit, hold-back).

    Hold-back starts at the rightmost ``⟦`` with no matching ``⟧`` after it.
    If the hold would exceed ``_MAX_HOLD_CHARS`` we give up and emit
    everything — the alternative is stalling the stream forever on a stray
    bracket.
    """
    if not text:
        return "", ""
    open_idx = text.rfind(DELIM_OPEN)
    if open_idx < 0:
        return text, ""
    if text.find(DELIM_CLOSE, open_idx) >= 0:
        return text, ""
    if len(text) - open_idx > _MAX_HOLD_CHARS:
        return text, ""
    return text[:open_idx], text[open_idx:]


def _substitute(text: str, pmap: PlaceholderMap) -> str:
    def repl(m: re.Match[str]) -> str:
        original = pmap.unmask(m.group(0))
        return original if original is not None else m.group(0)

    return _PLACEHOLDER_RE.sub(repl, text)


def _substitute_json_string(text: str, pmap: PlaceholderMap) -> str:
    """Substitute placeholders inside a JSON-string fragment.

    ``partial_json`` deltas concatenate downstream into a JSON document. If
    the original plaintext contains a ``"`` or ``\\``, naive substitution
    would corrupt that document — so we JSON-encode each replacement and
    strip the outer quotes to land an escaped fragment in the partial_json
    string. The placeholder pattern itself contains no JSON-special chars,
    so the surrounding text is safe to leave alone.
    """

    def repl(m: re.Match[str]) -> str:
        original = pmap.unmask(m.group(0))
        if original is None:
            return m.group(0)
        return json.dumps(original, ensure_ascii=False)[1:-1]

    return _PLACEHOLDER_RE.sub(repl, text)


class _Channel:
    """Per-block-index text accumulator + last-seen event template.

    ``orig`` is the concatenated original text the upstream emitted for this
    block. ``emitted_chars`` tracks how many chars of ``orig`` we have already
    flushed downstream (after substitution). ``template`` is a deep-copy-safe
    parsed JSON dict from the most recent text-delta event for this block; we
    use it as the shape for synthesized output events so downstream SDKs see
    the same fields they'd see from the real upstream.
    """

    __slots__ = ("orig", "emitted_chars", "template", "event_name", "kind")

    def __init__(self) -> None:
        self.orig: str = ""
        self.emitted_chars: int = 0
        self.template: dict | None = None
        self.event_name: bytes | None = None
        # One of: "anthropic_text", "anthropic_input_json",
        # "anthropic_thinking", "openai". Picks both the substitution
        # function (raw vs JSON-string-escaped) and the emit field.
        self.kind: str = "anthropic_text"


class SSEUnmask:
    """Stateful SSE-aware unmasker.

    Usage mirrors :class:`UnmaskStream`::

        u = SSEUnmask(pmap)
        async for chunk in upstream:
            client.write(u.feed(chunk))
        client.write(u.flush())
    """

    __slots__ = ("_pmap", "_buf", "_channels")

    def __init__(self, pmap: PlaceholderMap) -> None:
        self._pmap = pmap
        self._buf = b""
        self._channels: dict[tuple[str, int], _Channel] = {}

    # ------------------------------------------------------------------

    def feed(self, chunk: bytes) -> bytes:
        if not chunk:
            return b""
        self._buf += chunk
        out = bytearray()
        # An SSE event terminates with a blank line — i.e. the byte sequence
        # \n\n (or \r\n\r\n which contains \n\n as a substring; fine). We
        # process events in order, holding any partial trailing event in
        # ``self._buf``.
        while True:
            sep = self._buf.find(b"\n\n")
            if sep < 0:
                break
            event_bytes = self._buf[: sep + 2]
            self._buf = self._buf[sep + 2 :]
            out += self._process_event(event_bytes)
        return bytes(out)

    def flush(self) -> bytes:
        out = bytearray()
        if self._buf:
            # Trailing bytes without a terminator: best-effort process; if
            # unparseable, pass through verbatim (better than swallowing).
            tail = self._process_event(self._buf, force=True)
            out += tail
            self._buf = b""
        # Flush any text held back by every open channel. ``final=True`` means
        # we no longer wait for placeholders to complete — emit whatever we
        # have, substituting any complete ones.
        for key, ch in self._channels.items():
            extra = self._drain_channel(key, ch, final=True)
            if extra:
                out += extra
        self._channels.clear()
        return bytes(out)

    # ------------------------------------------------------------------

    def _process_event(self, event_bytes: bytes, *, force: bool = False) -> bytes:
        event_name, data_payload = _parse_sse_event(event_bytes)
        if data_payload is None:
            # No data line (comment, ``: keep-alive``, etc.) — pass through.
            return event_bytes

        try:
            data = json.loads(data_payload)
        except json.JSONDecodeError:
            # ``data: [DONE]`` (OpenAI termination) and similar non-JSON
            # markers fall here. Flush every channel before passing through so
            # the client sees substituted text in arrival order.
            return self._flush_all_channels() + event_bytes

        if not isinstance(data, dict):
            return self._flush_all_channels() + event_bytes

        info = _classify(event_name, data)
        if info is None:
            # Structural / non-text event — flush channels first.
            return self._flush_all_channels() + event_bytes

        ch_key = info["channel_key"]
        text = info["text"]
        ch = self._channels.get(ch_key)
        if ch is None:
            ch = _Channel()
            self._channels[ch_key] = ch
        ch.template = info["template"]
        ch.event_name = event_name
        ch.kind = info["kind"]
        ch.orig += text

        # Decide what's safe to emit now; final=force only on flush.
        return self._drain_channel(ch_key, ch, final=force)

    def _drain_channel(
        self, key: tuple[str, int], ch: _Channel, *, final: bool
    ) -> bytes:
        if ch.template is None:
            return b""
        if final:
            safe = ch.orig
        else:
            safe, _ = _safe_split(ch.orig)
        if len(safe) <= ch.emitted_chars:
            return b""
        new_segment = safe[ch.emitted_chars :]
        ch.emitted_chars = len(safe)
        if ch.kind == "anthropic_input_json":
            substituted = _substitute_json_string(new_segment, self._pmap)
        else:
            substituted = _substitute(new_segment, self._pmap)
        if not substituted and not new_segment:
            return b""
        return _emit_text_event(ch.event_name, ch.template, ch.kind, substituted)

    def _flush_all_channels(self) -> bytes:
        if not self._channels:
            return b""
        out = bytearray()
        for key, ch in self._channels.items():
            extra = self._drain_channel(key, ch, final=True)
            if extra:
                out += extra
        return bytes(out)


# ----------------------------------------------------------------------
# SSE event parsing + classification
# ----------------------------------------------------------------------


def _parse_sse_event(event_bytes: bytes) -> tuple[bytes | None, bytes | None]:
    """Return (event_name, data_payload). Either may be ``None``.

    SSE allows multiple ``data:`` lines per event; per the spec they're joined
    with ``\\n``. We honor that.
    """
    event_name: bytes | None = None
    data_lines: list[bytes] = []
    for raw_line in event_bytes.splitlines():
        if not raw_line or raw_line.startswith(b":"):
            continue
        if raw_line.startswith(b"event:"):
            event_name = raw_line[len(b"event:") :].strip()
        elif raw_line.startswith(b"data:"):
            payload = raw_line[len(b"data:") :]
            if payload.startswith(b" "):
                payload = payload[1:]
            data_lines.append(payload)
    if not data_lines:
        return event_name, None
    return event_name, b"\n".join(data_lines)


def _classify(event_name: bytes | None, data: dict) -> dict | None:
    """Return ``{"channel_key", "text", "template"}`` for a text-bearing event.

    Returns ``None`` for any other event shape, signalling "pass through after
    flushing". Recognized streaming-text shapes:

    * Anthropic ``content_block_delta`` with ``text_delta`` → ``delta.text``
    * Anthropic ``content_block_delta`` with ``input_json_delta`` →
      ``delta.partial_json`` (tool_use input streaming; placeholders here
      drive the bash/edit/read commands the model is composing, so leaking
      them breaks the tools)
    * Anthropic ``content_block_delta`` with ``thinking_delta`` →
      ``delta.thinking`` (extended thinking)
    * OpenAI ``choices[].delta.content``

    Each shape gets its own channel key so per-block-index text and
    partial_json don't bleed into each other when the upstream interleaves
    multiple content blocks.
    """
    delta = data.get("delta")
    if isinstance(delta, dict):
        dtype = delta.get("type")
        idx = data.get("index")
        if not isinstance(idx, int):
            idx = 0

        # Anthropic text_delta — the common path. ``type`` may be omitted by
        # lightweight upstreams / test fixtures; the ``text`` key is the
        # load-bearing signal there.
        if isinstance(delta.get("text"), str) and (dtype is None or dtype == "text_delta"):
            return {
                "channel_key": ("anthropic_text", idx),
                "text": delta["text"],
                "template": data,
                "kind": "anthropic_text",
            }

        # Anthropic input_json_delta — partial_json fragments concatenate into
        # a JSON document. Placeholders inside this stream are commonly real
        # paths or names the model is using to call tools; we must substitute
        # them with JSON-string escaping so the eventual document still parses.
        if dtype == "input_json_delta" and isinstance(delta.get("partial_json"), str):
            return {
                "channel_key": ("anthropic_input_json", idx),
                "text": delta["partial_json"],
                "template": data,
                "kind": "anthropic_input_json",
            }

        # Anthropic thinking_delta — plaintext like text_delta.
        if dtype == "thinking_delta" and isinstance(delta.get("thinking"), str):
            return {
                "channel_key": ("anthropic_thinking", idx),
                "text": delta["thinking"],
                "template": data,
                "kind": "anthropic_thinking",
            }

    # OpenAI streaming: choices is a list; each entry has .delta.content.
    # Multi-choice events are rare but we handle them by collapsing to the
    # first choice that carries content (the common single-choice shape).
    choices = data.get("choices")
    if isinstance(choices, list):
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            cdelta = choice.get("delta")
            if isinstance(cdelta, dict) and isinstance(cdelta.get("content"), str):
                idx = choice.get("index")
                if not isinstance(idx, int):
                    idx = 0
                return {
                    "channel_key": ("openai", idx),
                    "text": cdelta["content"],
                    "template": data,
                    "kind": "openai",
                }

    return None


def _emit_text_event(
    event_name: bytes | None,
    template: dict,
    kind: str,
    new_text: str,
) -> bytes:
    """Re-serialize a text-bearing event with ``new_text`` in the right field.

    For ``anthropic_input_json`` callers must pre-escape ``new_text`` as a
    JSON-string fragment (see ``_substitute_json_string``) — we don't double-
    encode here because the outer ``json.dumps`` will treat the value as an
    ordinary string.
    """
    out = copy.deepcopy(template)
    if kind == "anthropic_text":
        delta = out.get("delta")
        if isinstance(delta, dict):
            delta["text"] = new_text
    elif kind == "anthropic_input_json":
        delta = out.get("delta")
        if isinstance(delta, dict):
            delta["partial_json"] = new_text
    elif kind == "anthropic_thinking":
        delta = out.get("delta")
        if isinstance(delta, dict):
            delta["thinking"] = new_text
    else:  # openai
        choices = out.get("choices")
        if isinstance(choices, list):
            for choice in choices:
                if not isinstance(choice, dict):
                    continue
                cdelta = choice.get("delta")
                if isinstance(cdelta, dict) and "content" in cdelta:
                    cdelta["content"] = new_text
                    break
    # ensure_ascii=False keeps any plaintext non-ASCII bytes (e.g. user names
    # with accents) compact; downstream JSON parsers handle either form, but
    # avoiding the escape keeps the wire smaller and matches the original
    # request bytes the user typed.
    serialized = json.dumps(out, ensure_ascii=False).encode("utf-8")
    parts: list[bytes] = []
    if event_name:
        parts.append(b"event: " + event_name + b"\n")
    parts.append(b"data: " + serialized + b"\n\n")
    return b"".join(parts)
