"""Unit tests for the SSE-aware unmasker.

The byte-level UnmaskStream is exercised in test_unmask_stream.py; here we
cover the cases SSEUnmask exists for: placeholders fragmented across multiple
``content_block_delta`` events, OpenAI-shaped streams, byte-level chunking
inside SSE events, and structural events that force a flush.

Synthetic identities only (Ada Lovelace, Grace Hopper, *@example.com, etc.)
per project convention.
"""

from __future__ import annotations

import json

from pii_proxi.masking.placeholder import PlaceholderMap
from pii_proxi.masking.sse_unmask import SSEUnmask


def _pmap_with(original: str, label: str) -> tuple[PlaceholderMap, str]:
    pmap = PlaceholderMap(b"\x11" * 32)
    return pmap, pmap.mask(original, label)


def _feed(pmap: PlaceholderMap, chunks: list[bytes]) -> bytes:
    u = SSEUnmask(pmap)
    out = bytearray()
    for c in chunks:
        out += u.feed(c)
    out += u.flush()
    return bytes(out)


def _decode_anthropic_text(stream: bytes) -> str:
    """Concatenate every ``delta.text`` field a downstream SDK would render."""
    parts: list[str] = []
    for raw_event in stream.split(b"\n\n"):
        for line in raw_event.splitlines():
            if not line.startswith(b"data:"):
                continue
            payload = line[len(b"data:") :].lstrip()
            try:
                data = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if not isinstance(data, dict):
                continue
            delta = data.get("delta")
            if isinstance(delta, dict) and isinstance(delta.get("text"), str):
                parts.append(delta["text"])
    return "".join(parts)


def _decode_openai_content(stream: bytes) -> str:
    parts: list[str] = []
    for raw_event in stream.split(b"\n\n"):
        for line in raw_event.splitlines():
            if not line.startswith(b"data:"):
                continue
            payload = line[len(b"data:") :].lstrip()
            if payload == b"[DONE]":
                continue
            try:
                data = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if not isinstance(data, dict):
                continue
            for choice in data.get("choices", []):
                if not isinstance(choice, dict):
                    continue
                delta = choice.get("delta")
                if isinstance(delta, dict) and isinstance(delta.get("content"), str):
                    parts.append(delta["content"])
    return "".join(parts)


def _anthropic_delta(text: str, index: int = 0) -> bytes:
    payload = {
        "type": "content_block_delta",
        "index": index,
        "delta": {"type": "text_delta", "text": text},
    }
    return (
        b"event: content_block_delta\n"
        b"data: " + json.dumps(payload, ensure_ascii=False).encode("utf-8") + b"\n\n"
    )


def _anthropic_input_json_delta(partial_json: str, index: int = 0) -> bytes:
    payload = {
        "type": "content_block_delta",
        "index": index,
        "delta": {"type": "input_json_delta", "partial_json": partial_json},
    }
    return (
        b"event: content_block_delta\n"
        b"data: " + json.dumps(payload, ensure_ascii=False).encode("utf-8") + b"\n\n"
    )


def _decode_anthropic_partial_json(stream: bytes) -> str:
    parts: list[str] = []
    for raw_event in stream.split(b"\n\n"):
        for line in raw_event.splitlines():
            if not line.startswith(b"data:"):
                continue
            payload = line[len(b"data:") :].lstrip()
            try:
                data = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if not isinstance(data, dict):
                continue
            delta = data.get("delta")
            if (
                isinstance(delta, dict)
                and delta.get("type") == "input_json_delta"
                and isinstance(delta.get("partial_json"), str)
            ):
                parts.append(delta["partial_json"])
    return "".join(parts)


def _openai_delta(content: str, index: int = 0) -> bytes:
    payload = {"choices": [{"index": index, "delta": {"content": content}}]}
    return b"data: " + json.dumps(payload, ensure_ascii=False).encode("utf-8") + b"\n\n"


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------


def test_anthropic_placeholder_inside_one_event_is_substituted():
    pmap, ph = _pmap_with("alice@example.com", "private_email")
    stream = _anthropic_delta(f"contact {ph} please")
    out = _feed(pmap, [stream])
    assert _decode_anthropic_text(out) == "contact alice@example.com please"
    assert b"\xe2\xa6\xa6" not in out  # raw ⟦ bytes
    assert b"private_email_" not in out


def test_anthropic_placeholder_split_across_three_events():
    """The bug this module exists to fix: placeholder fragmented across three
    text-delta events. Byte-level scan can't recover; SSE-aware must."""
    pmap, ph = _pmap_with("Ada Lovelace", "private_person")
    third = len(ph) // 3
    stream = (
        _anthropic_delta(f"You are {ph[:third]}")
        + _anthropic_delta(ph[third : 2 * third])
        + _anthropic_delta(f"{ph[2 * third :]}.")
        + b"event: content_block_stop\ndata: {\"index\":0}\n\n"
    )
    out = _feed(pmap, [stream])
    rendered = _decode_anthropic_text(out)
    assert rendered == "You are Ada Lovelace."


def test_anthropic_held_text_flushed_before_block_stop():
    """If the upstream closes the block while we're still holding text, we
    must flush before passing through ``content_block_stop`` — otherwise the
    client renders a closed block with missing trailing characters."""
    pmap, ph = _pmap_with("Grace Hopper", "private_person")
    stream = (
        _anthropic_delta(f"hi {ph[:5]}")
        + b"event: content_block_stop\ndata: {\"index\":0}\n\n"
    )
    out = _feed(pmap, [stream])
    # The held text (which includes a partial ``⟦priv``) must be flushed; the
    # placeholder is incomplete so it leaks as-is — that's correct: we
    # promised never to drop bytes, and the upstream gave us a truncated
    # placeholder.
    rendered = _decode_anthropic_text(out)
    assert rendered.startswith("hi ")
    assert ph[:5] in rendered or "Grace Hopper" in rendered  # held tail emitted
    # And content_block_stop must appear after the delta, in order.
    stop_pos = out.find(b"content_block_stop")
    last_delta = out.rfind(b"content_block_delta")
    assert last_delta >= 0 and stop_pos > last_delta


def test_anthropic_byte_chunking_inside_event_is_handled():
    """A single SSE event arriving byte-by-byte must still parse correctly."""
    pmap, ph = _pmap_with("alan@example.com", "private_email")
    stream = (
        _anthropic_delta(f"mail {ph} now")
        + b"event: message_stop\ndata: {}\n\n"
    )
    one_shot = _feed(pmap, [stream])
    pmap2, _ = _pmap_with("alan@example.com", "private_email")
    byte_by_byte = _feed(pmap2, [stream[i : i + 1] for i in range(len(stream))])
    assert _decode_anthropic_text(one_shot) == _decode_anthropic_text(byte_by_byte)
    assert _decode_anthropic_text(one_shot) == "mail alan@example.com now"


def test_anthropic_unknown_placeholder_passes_through():
    """Shape-correct but unminted placeholder (e.g. from a previous session)
    must be left intact rather than substituted with garbage."""
    pmap = PlaceholderMap(b"\x22" * 32)
    fake = "⟦private_email_deadbeef⟧"
    stream = _anthropic_delta(f"see {fake} ok")
    out = _feed(pmap, [stream])
    assert _decode_anthropic_text(out) == f"see {fake} ok"


def test_anthropic_multiple_placeholders_one_event():
    pmap = PlaceholderMap(b"\x33" * 32)
    a = pmap.mask("Ada", "private_person")
    b = pmap.mask("turing@example.com", "private_email")
    stream = _anthropic_delta(f"{a} mailed {b}")
    out = _feed(pmap, [stream])
    assert _decode_anthropic_text(out) == "Ada mailed turing@example.com"


def test_anthropic_structural_passthrough_preserves_event_names():
    pmap = PlaceholderMap(b"\x44" * 32)
    raw = (
        b"event: message_start\ndata: {\"type\":\"message_start\"}\n\n"
        b"event: ping\ndata: {\"type\":\"ping\"}\n\n"
    )
    out = _feed(pmap, [raw])
    assert b"event: message_start" in out
    assert b"event: ping" in out


# ---------------------------------------------------------------------------
# Anthropic input_json_delta (tool_use input streaming)
# ---------------------------------------------------------------------------


def test_input_json_delta_substitutes_placeholder_in_one_event():
    pmap, ph = _pmap_with("kunalmehan", "private_person")
    fragment = f'{{"command": "ls /Users/{ph}/.claude/"}}'
    stream = _anthropic_input_json_delta(fragment)
    out = _feed(pmap, [stream])
    rendered = _decode_anthropic_partial_json(out)
    # The substituted fragment must concatenate into valid JSON the client
    # can parse and execute.
    assert rendered == '{"command": "ls /Users/kunalmehan/.claude/"}'
    parsed = json.loads(rendered)
    assert parsed == {"command": "ls /Users/kunalmehan/.claude/"}


def test_input_json_delta_placeholder_split_across_events():
    pmap, ph = _pmap_with("kunalmehan", "private_person")
    head = '{"command": "ls /Users/'
    tail = '/.claude/"}'
    third = len(ph) // 3
    stream = (
        _anthropic_input_json_delta(head + ph[:third])
        + _anthropic_input_json_delta(ph[third : 2 * third])
        + _anthropic_input_json_delta(ph[2 * third :] + tail)
        + b"event: content_block_stop\ndata: {\"index\":0}\n\n"
    )
    out = _feed(pmap, [stream])
    rendered = _decode_anthropic_partial_json(out)
    assert rendered == '{"command": "ls /Users/kunalmehan/.claude/"}'
    assert json.loads(rendered) == {"command": "ls /Users/kunalmehan/.claude/"}


def test_input_json_delta_escapes_json_special_chars_in_substitution():
    """If the original plaintext contains ``"`` or ``\\``, raw substitution
    would break the eventual JSON document. The substitute must JSON-escape
    so the partial_json fragment stays valid."""
    pmap, ph = _pmap_with('he said "hi"\\there', "secret")
    fragment = f'{{"text": "msg: {ph}"}}'
    stream = _anthropic_input_json_delta(fragment)
    out = _feed(pmap, [stream])
    rendered = _decode_anthropic_partial_json(out)
    parsed = json.loads(rendered)
    assert parsed == {"text": 'msg: he said "hi"\\there'}


def test_input_json_delta_separate_channel_from_text_delta():
    """Same content_block_index, both text and input_json deltas — must not
    bleed into each other (different field names, different substitution
    rules)."""
    pmap = PlaceholderMap(b"\x77" * 32)
    ph_name = pmap.mask("Ada", "private_person")
    ph_path = pmap.mask("ada-home", "private_person")
    stream = (
        _anthropic_delta(f"hello {ph_name}", index=0)
        + _anthropic_input_json_delta(
            f'{{"path": "/{ph_path}/x"}}', index=0
        )
        + b"event: content_block_stop\ndata: {\"index\":0}\n\n"
    )
    out = _feed(pmap, [stream])
    assert _decode_anthropic_text(out) == "hello Ada"
    assert (
        _decode_anthropic_partial_json(out) == '{"path": "/ada-home/x"}'
    )


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------


def test_openai_placeholder_split_across_two_chunks():
    pmap, ph = _pmap_with("hopper@example.com", "private_email")
    half = len(ph) // 2
    stream = (
        _openai_delta(f"contact: {ph[:half]}")
        + _openai_delta(f"{ph[half:]} please")
        + b"data: [DONE]\n\n"
    )
    out = _feed(pmap, [stream])
    rendered = _decode_openai_content(out)
    assert rendered == "contact: hopper@example.com please"


def test_openai_done_marker_flushes_held_text():
    pmap, ph = _pmap_with("Ada", "private_person")
    # Single delta with a complete placeholder, then [DONE].
    stream = _openai_delta(f"hi {ph}") + b"data: [DONE]\n\n"
    out = _feed(pmap, [stream])
    assert _decode_openai_content(out) == "hi Ada"
    assert b"data: [DONE]" in out


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_stream_is_noop():
    pmap = PlaceholderMap(b"\x55" * 32)
    assert _feed(pmap, []) == b""


def test_comment_lines_pass_through():
    pmap = PlaceholderMap(b"\x66" * 32)
    raw = b": keep-alive\n\n" + _anthropic_delta("plain text")
    out = _feed(pmap, [raw])
    assert b": keep-alive" in out
    assert _decode_anthropic_text(out) == "plain text"
