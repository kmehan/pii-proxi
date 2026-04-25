from __future__ import annotations

import threading

import pytest

from pii_proxi.masking.placeholder import (
    DELIM_CLOSE,
    DELIM_OPEN,
    PlaceholderMap,
    Span,
    apply_spans,
)
from pii_proxi.session import new_session_key


def _fresh() -> PlaceholderMap:
    return PlaceholderMap(new_session_key())


def test_mint_shape_matches_pattern():
    pmap = _fresh()
    ph = pmap.mask("sk-live-xyz", "SECRET")
    assert ph.startswith(DELIM_OPEN)
    assert ph.endswith(DELIM_CLOSE)
    assert pmap.pattern.fullmatch(ph) is not None


def test_determinism_within_session():
    pmap = _fresh()
    a = pmap.mask("sk-abc", "SECRET")
    b = pmap.mask("sk-abc", "SECRET")
    assert a == b
    assert pmap.unmask(a) == "sk-abc"


def test_different_sessions_yield_different_placeholders():
    p1 = PlaceholderMap(b"\x00" * 32)
    p2 = PlaceholderMap(b"\x01" * 32)
    assert p1.mask("alice@example.com", "EMAIL") != p2.mask(
        "alice@example.com", "EMAIL"
    )


def test_unmask_unknown_returns_none():
    pmap = _fresh()
    assert pmap.unmask(f"{DELIM_OPEN}EMAIL_deadbeef{DELIM_CLOSE}") is None


def test_repr_hides_plaintext():
    pmap = _fresh()
    pmap.mask("secret-value", "SECRET")
    r = repr(pmap)
    assert "secret-value" not in r
    assert "entries=1" in r


def test_apply_spans_reverse_order_and_splice():
    pmap = _fresh()
    text = "contact alice@example.com or bob@example.com today"
    spans = [
        Span(8, 25, "EMAIL"),
        Span(29, 44, "EMAIL"),
    ]
    out = apply_spans(text, spans, pmap)
    assert "alice@example.com" not in out
    assert "bob@example.com" not in out
    # both placeholders present, and pattern round-trips.
    matches = pmap.pattern.findall(out)
    assert len(matches) == 2


def test_apply_spans_longest_wins_on_overlap():
    pmap = _fresh()
    text = "xxxhelloworldxxx"
    # Two overlapping candidates: "hello" (3..8) and "helloworld" (3..13).
    spans = [Span(3, 8, "A"), Span(3, 13, "B")]
    out = apply_spans(text, spans, pmap)
    # "helloworld" (label B) should have won; no "A" placeholder produced.
    assert pmap.unmask(_only_match(pmap, out)) == "helloworld"


def test_apply_spans_empty_spans_returns_text_unchanged():
    pmap = _fresh()
    assert apply_spans("hello", [], pmap) == "hello"


def test_apply_spans_rejects_out_of_bounds():
    pmap = _fresh()
    with pytest.raises(ValueError):
        apply_spans("abc", [Span(0, 99, "X")], pmap)


def test_apply_spans_drops_zero_length_spans():
    pmap = _fresh()
    out = apply_spans("abc", [Span(1, 1, "X")], pmap)
    assert out == "abc"


def test_pattern_matches_lowercase_label():
    """The privacy-filter model emits lowercase labels (``private_email``,
    ``private_person``, ``secret``...). The pattern must match them, otherwise
    ``UnmaskStream`` silently leaks every real placeholder back to the client.
    """
    pmap = _fresh()
    assert pmap.pattern.fullmatch(f"{DELIM_OPEN}private_email_deadbeef{DELIM_CLOSE}")
    assert pmap.pattern.fullmatch(f"{DELIM_OPEN}private_person_cafebabe{DELIM_CLOSE}")
    assert pmap.pattern.fullmatch(f"{DELIM_OPEN}secret_01234567{DELIM_CLOSE}")
    # Uppercase still matches too (synthetic test labels rely on it).
    assert pmap.pattern.fullmatch(f"{DELIM_OPEN}EMAIL_deadbeef{DELIM_CLOSE}")


def test_pattern_rejects_non_label_garbage_between_delimiters():
    pmap = _fresh()
    # Random prose between the delimiters must not match — that would let the
    # unmasker chew on legitimate text that happens to sit inside ⟦ ⟧.
    assert pmap.pattern.fullmatch(f"{DELIM_OPEN} just prose {DELIM_CLOSE}") is None
    assert pmap.pattern.fullmatch(f"{DELIM_OPEN}1email_deadbeef{DELIM_CLOSE}") is None  # leading digit
    assert pmap.pattern.fullmatch(f"{DELIM_OPEN}email_deadbee{DELIM_CLOSE}") is None  # short hex
    assert pmap.pattern.fullmatch(f"{DELIM_OPEN}email_DEADBEEF{DELIM_CLOSE}") is None  # uppercase hex


def test_thread_safety_stress():
    pmap = _fresh()
    n_threads = 8
    per_thread = 200
    secrets = [f"secret-{i}" for i in range(per_thread)]

    results: list[list[str]] = [[] for _ in range(n_threads)]

    def worker(idx: int) -> None:
        for s in secrets:
            results[idx].append(pmap.mask(s, "SECRET"))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Every thread saw the same placeholder for each secret.
    for col in range(per_thread):
        col_values = {results[row][col] for row in range(n_threads)}
        assert len(col_values) == 1
    assert len(pmap) == per_thread


def test_rejects_empty_session_key():
    with pytest.raises(ValueError):
        PlaceholderMap(b"")


def test_rejects_non_bytes_session_key():
    with pytest.raises(TypeError):
        PlaceholderMap("not-bytes")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------


def _only_match(pmap: PlaceholderMap, text: str) -> str:
    matches = [m.group(0) for m in pmap.pattern.finditer(text)]
    assert len(matches) == 1, f"expected 1 placeholder, saw {matches}"
    return matches[0]
