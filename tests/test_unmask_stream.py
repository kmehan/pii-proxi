from __future__ import annotations


from pii_proxi.masking.placeholder import PlaceholderMap
from pii_proxi.masking.unmask_stream import UnmaskStream


def _pmap_with(original: str, label: str) -> tuple[PlaceholderMap, str]:
    pmap = PlaceholderMap(b"\x11" * 32)
    ph = pmap.mask(original, label)
    return pmap, ph


def _feed_chunks(pmap: PlaceholderMap, chunks: list[bytes]) -> bytes:
    u = UnmaskStream(pmap)
    out = bytearray()
    for c in chunks:
        out += u.feed(c)
    out += u.flush()
    return bytes(out)


def test_whole_chunk_replacement():
    pmap, ph = _pmap_with("alice@example.com", "EMAIL")
    payload = f"hello {ph} world".encode("utf-8")
    assert _feed_chunks(pmap, [payload]) == b"hello alice@example.com world"


def test_byte_by_byte_feed_is_identical_to_whole_chunk():
    pmap, ph = _pmap_with("alice@example.com", "EMAIL")
    payload = f"prefix {ph} middle {ph} suffix".encode("utf-8")

    whole = _feed_chunks(pmap, [payload])
    one_by_one = _feed_chunks(pmap, [payload[i : i + 1] for i in range(len(payload))])
    assert whole == one_by_one


def test_split_at_every_offset_inside_placeholder():
    """Load-bearing: a placeholder split at any byte boundary must still decode."""
    pmap, ph = _pmap_with("sk-live-XYZABC123", "SECRET")
    payload = f"before {ph} after".encode("utf-8")
    expected = b"before sk-live-XYZABC123 after"

    for split in range(len(payload) + 1):
        pmap2, ph2 = _pmap_with("sk-live-XYZABC123", "SECRET")
        # Same key, same input -> same placeholder bytes.
        assert ph2 == ph
        out = _feed_chunks(pmap2, [payload[:split], payload[split:]])
        assert out == expected, f"mismatch at split={split}: {out!r}"


def test_three_way_split_inside_delimiters():
    # Exercises the multi-byte UTF-8 prefix hold-back: splits at every byte
    # offset inside both the opening and closing brackets.
    pmap, ph = _pmap_with("bob@x.io", "EMAIL")
    payload = f"x{ph}y".encode("utf-8")
    expected = b"xbob@x.ioy"
    n = len(payload)
    for i in range(n + 1):
        for j in range(i, n + 1):
            chunks = [payload[:i], payload[i:j], payload[j:]]
            assert _feed_chunks(pmap, chunks) == expected


def test_unknown_placeholder_passed_through():
    pmap = PlaceholderMap(b"\x22" * 32)
    fake = "⟦EMAIL_deadbeef⟧".encode("utf-8")
    u = UnmaskStream(pmap)
    out = u.feed(b"before " + fake + b" after") + u.flush()
    assert out == b"before " + fake + b" after"


def test_non_placeholder_brackets_are_emitted_literally():
    pmap = PlaceholderMap(b"\x33" * 32)
    blob = "⟦ just prose ⟧".encode("utf-8")
    assert _feed_chunks(pmap, [blob]) == blob


def test_unterminated_open_bracket_flushed_at_end():
    pmap = PlaceholderMap(b"\x44" * 32)
    # Feed an open bracket with no close — flush must emit it.
    u = UnmaskStream(pmap)
    out = u.feed("hello ⟦EMAIL_cafebabe".encode("utf-8")) + u.flush()
    assert out == "hello ⟦EMAIL_cafebabe".encode("utf-8")


def test_bracket_close_never_arrives_within_cap():
    # Pathological producer: keeps writing past the wait cap without a close.
    pmap = PlaceholderMap(b"\x55" * 32)
    u = UnmaskStream(pmap)
    blob = "⟦" + "A" * 400
    out = u.feed(blob.encode("utf-8")) + u.flush()
    # We must emit everything — no infinite buffering.
    assert out == blob.encode("utf-8")


def test_sse_style_framing_split_across_events():
    pmap, ph = _pmap_with("alice@example.com", "EMAIL")
    event1 = f"data: {{\"delta\": \"pre {ph[:5]}".encode("utf-8")
    event2 = f"{ph[5:]} post\"}}\n\n".encode("utf-8")
    out = _feed_chunks(pmap, [event1, event2])
    assert b"alice@example.com" in out
    assert "⟦".encode("utf-8") not in out  # no stray ⟦ bytes
    assert b"\\u27e6" not in out  # and no stray ASCII-escape form either


def test_json_ascii_escaped_placeholder_is_unmasked():
    """Upstream JSON encoders default to ensure_ascii=True, which turns ⟦ into
    the 6-byte ASCII sequence \\u27e6. The unmasker must handle both forms."""
    import json as _json

    pmap, ph = _pmap_with("alice@example.com", "EMAIL")
    payload = _json.dumps({"text": f"key {ph}"}).encode("utf-8")
    assert b"\\u27e6" in payload  # sanity: upstream really ASCII-escaped it

    out = _feed_chunks(pmap, [payload])
    decoded = _json.loads(out.decode("utf-8"))
    assert decoded == {"text": "key alice@example.com"}


def test_json_ascii_escaped_split_at_every_offset():
    import json as _json

    pmap, ph = _pmap_with("sk-live-ABC", "SECRET")
    payload = _json.dumps({"t": f"pre {ph} post"}).encode("utf-8")
    expected = _json.loads(_json.dumps({"t": "pre sk-live-ABC post"}).encode("utf-8"))

    for split in range(len(payload) + 1):
        pmap2, ph2 = _pmap_with("sk-live-ABC", "SECRET")
        assert ph2 == ph
        out = _feed_chunks(pmap2, [payload[:split], payload[split:]])
        assert _json.loads(out.decode("utf-8")) == expected, f"split={split}"


def test_mixed_raw_and_escaped_placeholders_in_one_stream():
    pmap, ph = _pmap_with("secret-value", "SECRET")
    raw = f"raw: {ph}".encode("utf-8")
    # Escaped form as the upstream would emit after json.dumps
    esc = f'esc: "{ph}"'
    import json as _json

    esc_bytes = _json.dumps(esc).encode("utf-8")
    payload = raw + b"\n" + esc_bytes
    out = _feed_chunks(pmap, [payload])
    assert b"raw: secret-value" in out
    # The esc section should decode to the unmasked value inside a JSON string
    assert b"esc: \\\"secret-value\\\"" in out


def test_empty_feeds_and_flush_noop():
    pmap = PlaceholderMap(b"\x66" * 32)
    u = UnmaskStream(pmap)
    assert u.feed(b"") == b""
    assert u.flush() == b""


def test_lowercase_label_round_trip_raw_form():
    """Regression: the privacy-filter detector emits lowercase labels
    (``private_person``, ``private_email``, ``secret``). An older regex
    accepted only ``[A-Z][A-Z0-9_]*`` and silently leaked every real
    placeholder back through the client. Use synthetic data — never real PII.
    """
    pmap, ph = _pmap_with("Ada Lovelace", "private_person")
    payload = f"hello {ph}, welcome".encode("utf-8")
    assert _feed_chunks(pmap, [payload]) == b"hello Ada Lovelace, welcome"


def test_lowercase_label_round_trip_json_escaped_form():
    """Same regression, JSON-escaped variant. SSE streams from upstream JSON
    encoders default to ``ensure_ascii=True``, which turns ⟦ → \\u27e6."""
    import json as _json

    pmap, ph = _pmap_with("ada@example.com", "private_email")
    payload = _json.dumps({"text": f"contact {ph}"}).encode("utf-8")
    assert b"\\u27e6" in payload  # sanity: upstream really ASCII-escaped it
    out = _feed_chunks(pmap, [payload])
    assert _json.loads(out.decode("utf-8")) == {"text": "contact ada@example.com"}


def test_lowercase_label_split_at_every_offset():
    """The streaming hold-back logic must work for lowercase labels too —
    a placeholder split at any byte boundary still has to decode."""
    pmap, ph = _pmap_with("Grace Hopper", "private_person")
    payload = f"before {ph} after".encode("utf-8")
    expected = b"before Grace Hopper after"
    for split in range(len(payload) + 1):
        pmap2, ph2 = _pmap_with("Grace Hopper", "private_person")
        assert ph2 == ph
        out = _feed_chunks(pmap2, [payload[:split], payload[split:]])
        assert out == expected, f"mismatch at split={split}: {out!r}"


def test_backtoback_placeholders():
    pmap = PlaceholderMap(b"\x77" * 32)
    a = pmap.mask("one", "A")
    b = pmap.mask("two", "B")
    payload = (a + b).encode("utf-8")
    assert _feed_chunks(pmap, [payload]) == b"onetwo"
    # And under pathological byte-by-byte feed.
    assert _feed_chunks(pmap, [payload[i : i + 1] for i in range(len(payload))]) == b"onetwo"
