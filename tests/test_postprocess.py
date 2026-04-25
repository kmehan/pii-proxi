"""Tests for the shared BIOES post-processor.

These tests exercise the two pieces of real logic in ``postprocess.py``:
the tiktoken offset-recovery path and the constrained Viterbi. Both
backends delegate here, so backend parity is guaranteed as long as these
pass.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pii_proxi.detection.base import Span
from pii_proxi.detection.postprocess import (
    PostProcessor,
    load_calibration,
)


# Small toy label set: O + BIOES for EMAIL and SECRET.
# 0: O
# 1: B-EMAIL, 2: I-EMAIL, 3: E-EMAIL, 4: S-EMAIL
# 5: B-SECRET, 6: I-SECRET, 7: E-SECRET, 8: S-SECRET
TOY_ID2LABEL = {
    0: "O",
    1: "B-EMAIL", 2: "I-EMAIL", 3: "E-EMAIL", 4: "S-EMAIL",
    5: "B-SECRET", 6: "I-SECRET", 7: "E-SECRET", 8: "S-SECRET",
}

ZERO_CALIBRATION = {
    "transition_bias_background_stay": 0.0,
    "transition_bias_background_to_start": 0.0,
    "transition_bias_end_to_background": 0.0,
    "transition_bias_end_to_start": 0.0,
    "transition_bias_inside_to_continue": 0.0,
    "transition_bias_inside_to_end": 0.0,
}


def _one_hot_logits(tag_sequence: list[int], num_labels: int, strength: float = 10.0) -> np.ndarray:
    """Build ``[T, num_labels]`` logits whose argmax is ``tag_sequence``."""

    T = len(tag_sequence)
    logits = np.zeros((T, num_labels), dtype=np.float64)
    for t, tag in enumerate(tag_sequence):
        logits[t, tag] = strength
    return logits


# ---------------------------------------------------------------------------
# Calibration loader
# ---------------------------------------------------------------------------
def test_load_calibration_real_file_schema(tmp_path: Path) -> None:
    """The loader should handle the schema we observed on HF verbatim."""

    sample = {
        "operating_points": {
            "default": {
                "biases": {
                    "transition_bias_background_stay": 0.1,
                    "transition_bias_inside_to_end": -0.2,
                }
            }
        }
    }
    p = tmp_path / "viterbi_calibration.json"
    p.write_text(json.dumps(sample), encoding="utf-8")
    biases = load_calibration(p)
    # Specified keys come through as floats.
    assert biases["transition_bias_background_stay"] == pytest.approx(0.1)
    assert biases["transition_bias_inside_to_end"] == pytest.approx(-0.2)
    # Unspecified keys default to 0.0.
    assert biases["transition_bias_end_to_start"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Tokenization & char-offset recovery
# ---------------------------------------------------------------------------
def test_tokenize_ascii_offsets_cover_input() -> None:
    pp = PostProcessor(TOY_ID2LABEL, ZERO_CALIBRATION)
    text = "hello world"
    input_ids, attn, char_spans = pp.tokenize([text])
    assert input_ids.dtype == np.int64
    assert attn.dtype == np.int64
    assert input_ids.shape[0] == 1
    assert char_spans[0][0][0] == 0
    # The concatenated token spans should cover the full string.
    assert char_spans[0][-1][1] == len(text)
    # Every span should slice back to a substring that, concatenated in
    # order, matches the original text.
    recovered = "".join(text[s:e] for s, e in char_spans[0])
    assert recovered == text


def test_tokenize_multibyte_char_offsets_are_character_not_byte() -> None:
    """``naïve`` has ``ï`` which is 2 bytes in UTF-8. Span offsets must
    still be character offsets into the *string*, not bytes."""

    pp = PostProcessor(TOY_ID2LABEL, ZERO_CALIBRATION)
    text = "email: naïve@ex.com"
    # Character position of "naïve" in the string is index 7.
    assert text.index("naïve") == 7
    # Byte position differs (because "na" is 2 bytes but "ï" is 2 bytes → "nai" would be 3 bytes, etc.).
    assert text.encode("utf-8").index("naïve".encode("utf-8")) == 7  # coincidence here; prefix is ASCII
    # More pointed test: an emoji earlier in the string makes byte!=char.
    emoji_text = "📧 naïve@ex.com"
    # ``📧`` is 1 character but 4 bytes.
    assert emoji_text.index("naïve") == 2  # character index (space counts)
    assert emoji_text.encode("utf-8").index(b"na") == 5  # byte index

    _input_ids, _attn, char_spans = pp.tokenize([emoji_text])
    # Verify every token span is a valid character slice: re-slicing the
    # string with .start/.end gives back a prefix-coherent reconstruction.
    recovered = "".join(emoji_text[s:e] for s, e in char_spans[0])
    assert recovered == emoji_text

    # And that the spans for the "naïve" substring are genuinely
    # character-indexed: at least one span must start at a char index
    # that falls inside the "naïve" character range [2..7].
    assert any(2 <= s <= 7 for s, _ in char_spans[0])


def test_tokenize_empty_string_produces_zero_length_row() -> None:
    pp = PostProcessor(TOY_ID2LABEL, ZERO_CALIBRATION)
    input_ids, attn, char_spans = pp.tokenize([""])
    # Empty input → no spans; attention mask is all zeros.
    assert char_spans == [[]]
    assert int(attn.sum()) == 0


# ---------------------------------------------------------------------------
# Viterbi decode → Span extraction
# ---------------------------------------------------------------------------
def test_decode_clean_bie_run_yields_one_span() -> None:
    pp = PostProcessor(TOY_ID2LABEL, ZERO_CALIBRATION)
    # Pretend we have 5 tokens covering characters [0,3), [3,5), [5,10), [10,12), [12,17).
    char_spans = [[(0, 3), (3, 5), (5, 10), (10, 12), (12, 17)]]
    # Tag sequence: O, B-EMAIL, I-EMAIL, E-EMAIL, O  →  one EMAIL span spanning chars 3..12.
    logits = _one_hot_logits([0, 1, 2, 3, 0], num_labels=9)[None, :, :]
    text_lens = [5]
    spans = pp.decode(logits, char_spans, text_lens)[0]
    assert spans == [Span(start=3, end=12, label="EMAIL")]


def test_decode_single_s_tag_yields_one_span() -> None:
    pp = PostProcessor(TOY_ID2LABEL, ZERO_CALIBRATION)
    char_spans = [[(0, 4), (5, 15)]]
    logits = _one_hot_logits([0, 8], num_labels=9)[None, :, :]  # O, S-SECRET
    spans = pp.decode(logits, char_spans, [2])[0]
    assert spans == [Span(start=5, end=15, label="SECRET")]


def test_decode_viterbi_rejects_illegal_transition() -> None:
    """Argmax would be ``B-EMAIL → B-SECRET`` (illegal inside an open
    entity). Constrained Viterbi must pick a legal next-best path."""

    pp = PostProcessor(TOY_ID2LABEL, ZERO_CALIBRATION)
    char_spans = [[(0, 3), (3, 6), (6, 9), (9, 12)]]
    # Logits: favour B-EMAIL, then B-SECRET, then E-SECRET, then O.
    # Under argmax that's [1, 5, 7, 0] — the B-EMAIL → B-SECRET hop is
    # illegal (can't open a new entity while another is open). Viterbi
    # should route to a legal alternative. The cleanest legal path
    # through the given logits involves emitting an EMAIL span that
    # closes with E-EMAIL (id 3) on token 2, since the model's scores
    # give 5 (B-SECRET) a big margin over 3 (E-EMAIL) only at t=1.
    T, n = 4, 9
    logits = np.zeros((T, n), dtype=np.float64)
    logits[0, 1] = 10.0  # B-EMAIL
    logits[1, 5] = 10.0  # B-SECRET  <- preferred by argmax but illegal after B-EMAIL
    logits[1, 2] = 9.0   # I-EMAIL    <- next-best legal
    logits[2, 7] = 10.0  # E-SECRET  <- preferred by argmax (continues the SECRET)
    logits[2, 3] = 9.0   # E-EMAIL   <- next-best legal (closes the EMAIL)
    logits[3, 0] = 10.0  # O
    logits = logits[None, :, :]

    spans = pp.decode(logits, char_spans, [T])[0]
    # Viterbi should have produced a single EMAIL span — not a SECRET,
    # and not two broken spans.
    assert len(spans) == 1
    assert spans[0].label == "EMAIL"
    # The span should cover the B and E positions.
    assert spans[0].start == 0
    assert spans[0].end == 9


def test_decode_transition_matrix_has_no_illegal_legal_edges() -> None:
    """White-box check: build the transition matrix and assert a handful
    of canonically-illegal edges are ``-inf``."""

    pp = PostProcessor(TOY_ID2LABEL, ZERO_CALIBRATION)
    T = pp._transitions  # type: ignore[attr-defined]

    # O → I-EMAIL is illegal (can't start mid-entity).
    assert T[0, 2] < -1e20
    # B-EMAIL → B-SECRET is illegal (can't open new entity inside one).
    assert T[1, 5] < -1e20
    # B-EMAIL → I-SECRET is illegal (wrong body).
    assert T[1, 6] < -1e20
    # B-EMAIL → E-EMAIL is legal.
    assert T[1, 3] > -1e20
    # B-EMAIL → I-EMAIL is legal.
    assert T[1, 2] > -1e20
    # E-EMAIL → O is legal.
    assert T[3, 0] > -1e20
    # E-EMAIL → B-SECRET is legal.
    assert T[3, 5] > -1e20


def test_decode_no_entities_returns_empty() -> None:
    pp = PostProcessor(TOY_ID2LABEL, ZERO_CALIBRATION)
    char_spans = [[(0, 1), (1, 2), (2, 3)]]
    logits = _one_hot_logits([0, 0, 0], num_labels=9)[None, :, :]
    assert pp.decode(logits, char_spans, [3])[0] == []


def test_decode_end_to_end_tokenize_then_argmax_emulated_run() -> None:
    """Sanity: feed a real tokenized string through tokenize, fabricate
    logits that mark a specific token run as E-MAIL, and confirm the
    span's character offsets line up with the substring in the text."""

    pp = PostProcessor(TOY_ID2LABEL, ZERO_CALIBRATION)
    text = "contact naïve@ex.com for info"
    _ids, attn, char_spans = pp.tokenize([text])
    n = int(attn[0].sum())
    # Find the token that starts with the email's 'n' character (char 8 = 'n').
    email_start_char = text.index("naïve@ex.com")
    email_end_char = email_start_char + len("naïve@ex.com")
    # Identify the token range that overlaps the email substring. Any
    # token whose char range intersects [email_start_char, email_end_char)
    # is considered part of the email.
    overlap = [
        j
        for j, (cs, ce) in enumerate(char_spans[0])
        if cs < email_end_char and ce > email_start_char
    ]
    assert overlap, "expected at least one token overlapping the email"
    first_email_tok = overlap[0]
    last_email_tok = overlap[-1]
    # Build a fake tag sequence: O everywhere, B-EMAIL at first, I-EMAIL
    # in the middle, E-EMAIL at last. Handle the edge case where the
    # substring is a single token (promote to S-EMAIL).
    tags = [0] * n
    if first_email_tok == last_email_tok:
        tags[first_email_tok] = 4  # S-EMAIL
    else:
        tags[first_email_tok] = 1  # B-EMAIL
        for j in range(first_email_tok + 1, last_email_tok):
            tags[j] = 2  # I-EMAIL
        tags[last_email_tok] = 3  # E-EMAIL

    logits = _one_hot_logits(tags, num_labels=9)[None, :, :]
    spans = pp.decode(logits, char_spans, [n])[0]
    assert len(spans) == 1
    assert spans[0].label == "EMAIL"
    assert spans[0].start == char_spans[0][first_email_tok][0]
    assert spans[0].end == char_spans[0][last_email_tok][1]
    # Most importantly: the recovered substring must *contain* the email.
    # The leading-space token and similar BPE artefacts may cause the
    # span to extend a little past the literal email boundary on either
    # side; as long as the email itself lies inside the span we're good.
    recovered = text[spans[0].start : spans[0].end]
    assert "naïve@ex.com" in recovered
