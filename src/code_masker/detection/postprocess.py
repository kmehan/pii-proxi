"""Shared post-processing for privacy-filter detectors.

Both the MLX and ONNX backends run the same tokenizer (tiktoken
``o200k_base``) and the same BIOES-constrained Viterbi decoder. Keeping
this logic in one module guarantees span parity between backends — the
only thing that differs is the forward pass.

Pipeline
--------
1. ``PostProcessor.tokenize`` encodes each text into token ids with
   ``o200k_base`` and records, for each token, the ``(char_start,
   char_end)`` span of the source substring the token covers. tiktoken
   does not expose offsets directly, so we recover them by decoding each
   token back to bytes and walking a byte-cursor through the input's
   UTF-8 encoding.
2. The backend runs a forward pass and produces
   ``logits[B, S, num_labels]`` (float).
3. ``PostProcessor.decode`` runs a Viterbi over the BIOES label space
   with only legal transitions permitted, then walks the decoded tag
   sequence to emit one ``Span`` per ``B (I)* E`` run or ``S`` singleton.

Calibration file format
-----------------------
The model ships a ``viterbi_calibration.json`` alongside its weights.
The observed schema (as of privacy-filter v1) is::

    {
      "operating_points": {
        "default": {
          "biases": {
            "transition_bias_background_stay":     0.0,
            "transition_bias_background_to_start": 0.0,
            "transition_bias_end_to_background":   0.0,
            "transition_bias_end_to_start":        0.0,
            "transition_bias_inside_to_continue":  0.0,
            "transition_bias_inside_to_end":       0.0
          }
        }
      }
    }

The six biases are **log-space additive adjustments** applied to all
transitions in the corresponding category — they do *not* replace the
legality mask; illegal BIOES transitions are always ``-inf`` regardless
of calibration. In the stock file every bias is 0.0, so the effective
behaviour is "pure constrained Viterbi". The fields are mapped to
transition categories like so:

* ``background_stay`` — ``O → O``
* ``background_to_start`` — ``O → B-X`` and ``O → S-X``
* ``end_to_background`` — ``E-X → O`` and ``S-X → O``
* ``end_to_start`` — ``(E-X|S-X) → (B-Y|S-Y)`` (any type Y, same or other)
* ``inside_to_continue`` — ``B-X → I-X`` and ``I-X → I-X``
* ``inside_to_end`` — ``(B-X|I-X) → E-X``

Additional operating points (other than ``default``) are ignored; callers
can pass a different key via ``operating_point`` if the model ships
several.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import tiktoken

from code_masker.detection.base import Span


# tiktoken's pad token id. o200k_base doesn't ship with a canonical pad;
# we use 0 as a safe sentinel since the model's attention_mask zeros out
# padded positions anyway. (Both model configs list pad_token_id=199999,
# which is also valid but would require a special-token exception in
# encode(); 0 + attention_mask is simpler and yields identical logits in
# valid positions.)
_PAD_TOKEN_ID = 0

_NEG_INF = -1e30  # large negative, kept finite so nan-safe arithmetic works

_BIAS_KEYS = (
    "transition_bias_background_stay",
    "transition_bias_background_to_start",
    "transition_bias_end_to_background",
    "transition_bias_end_to_start",
    "transition_bias_inside_to_continue",
    "transition_bias_inside_to_end",
)


def load_calibration(path: str | Path, operating_point: str = "default") -> dict:
    """Load a ``viterbi_calibration.json`` and return the flat biases dict.

    Missing keys default to 0.0 so a truncated/partial calibration file
    still produces a working decoder.
    """

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    ops = raw.get("operating_points") or {}
    point = ops.get(operating_point) or {}
    biases = dict(point.get("biases") or {})
    return {key: float(biases.get(key, 0.0)) for key in _BIAS_KEYS}


def _parse_label(label: str) -> tuple[str, str]:
    """Split a BIOES label into (prefix, body). Prefix ∈ {O,B,I,E,S}.

    ``"O"`` returns ``("O", "")``. Any label that doesn't match the
    ``"<prefix>-<body>"`` shape is treated as ``O``-like (prefix ``"O"``),
    so a malformed id2label doesn't crash the decoder.
    """

    if label == "O":
        return "O", ""
    if len(label) >= 2 and label[1] == "-" and label[0] in "BIES":
        return label[0], label[2:]
    return "O", ""


class PostProcessor:
    """Tokenize, Viterbi-decode, and span-extract for privacy-filter models.

    ``id2label`` must contain contiguous integer keys ``0..N-1``. Typical
    privacy-filter models use 33 labels (O + 8 entity types × BIOES).
    """

    def __init__(
        self,
        id2label: Mapping[int, str],
        calibration: Mapping[str, float],
        tokenizer_name: str = "o200k_base",
    ) -> None:
        self.num_labels = len(id2label)
        # Normalise keys to int — they're often str when coming from JSON.
        self.id2label: list[str] = [""] * self.num_labels
        for k, v in id2label.items():
            self.id2label[int(k)] = v
        self._parsed: list[tuple[str, str]] = [_parse_label(l) for l in self.id2label]

        # Pre-compute legal transitions + additive bias.
        self._transitions = self._build_transitions(calibration)

        self._encoder = tiktoken.get_encoding(tokenizer_name)

    # ------------------------------------------------------------------
    # Transition matrix
    # ------------------------------------------------------------------
    def _build_transitions(self, calibration: Mapping[str, float]) -> np.ndarray:
        """Return ``T[num_labels, num_labels]`` where ``T[a, b]`` is the
        additive log-score for transitioning from tag ``a`` to tag ``b``.
        Illegal transitions are ``_NEG_INF``.
        """

        n = self.num_labels
        T = np.full((n, n), _NEG_INF, dtype=np.float64)

        b_stay = float(calibration.get("transition_bias_background_stay", 0.0))
        b_to_start = float(calibration.get("transition_bias_background_to_start", 0.0))
        b_end_to_bg = float(calibration.get("transition_bias_end_to_background", 0.0))
        b_end_to_start = float(calibration.get("transition_bias_end_to_start", 0.0))
        b_in_continue = float(calibration.get("transition_bias_inside_to_continue", 0.0))
        b_in_end = float(calibration.get("transition_bias_inside_to_end", 0.0))

        for a in range(n):
            pa, ba = self._parsed[a]
            for b in range(n):
                pb, bb = self._parsed[b]
                score = self._score_transition(
                    pa, ba, pb, bb,
                    b_stay, b_to_start, b_end_to_bg,
                    b_end_to_start, b_in_continue, b_in_end,
                )
                if score is not None:
                    T[a, b] = score
        return T

    @staticmethod
    def _score_transition(
        pa: str, ba: str, pb: str, bb: str,
        b_stay: float, b_to_start: float, b_end_to_bg: float,
        b_end_to_start: float, b_in_continue: float, b_in_end: float,
    ) -> float | None:
        """Return the additive score for ``a → b`` or ``None`` if illegal.

        Legality rules (standard BIOES):
          * From ``O``: go to ``O``, ``B-X``, or ``S-X`` (any X).
          * From ``B-X`` or ``I-X``: go to ``I-X`` or ``E-X`` (same X only).
          * From ``E-X`` or ``S-X``: go to ``O``, ``B-Y``, or ``S-Y`` (any Y).
        """

        # From O (background).
        if pa == "O":
            if pb == "O":
                return b_stay
            if pb in ("B", "S"):
                return b_to_start
            return None

        # From B-X or I-X: must continue same entity.
        if pa in ("B", "I"):
            if pb == "I" and bb == ba:
                return b_in_continue
            if pb == "E" and bb == ba:
                return b_in_end
            return None

        # From E-X or S-X: entity closed, can go anywhere non-continuing.
        if pa in ("E", "S"):
            if pb == "O":
                return b_end_to_bg
            if pb in ("B", "S"):
                return b_end_to_start
            return None

        return None

    def _initial_mask(self) -> np.ndarray:
        """Additive log-score for the *first* token's tag.
        Only ``O``, ``B-X``, and ``S-X`` are legal starting tags."""

        n = self.num_labels
        m = np.full((n,), _NEG_INF, dtype=np.float64)
        for i, (p, _body) in enumerate(self._parsed):
            if p in ("O", "B", "S"):
                m[i] = 0.0
        return m

    def _final_mask(self) -> np.ndarray:
        """Additive log-score for the *last* token's tag. A sequence
        cannot end mid-entity, so ``B-X`` and ``I-X`` are illegal finals."""

        n = self.num_labels
        m = np.full((n,), _NEG_INF, dtype=np.float64)
        for i, (p, _body) in enumerate(self._parsed):
            if p in ("O", "E", "S"):
                m[i] = 0.0
        return m

    # ------------------------------------------------------------------
    # Tokenization
    # ------------------------------------------------------------------
    def tokenize(
        self, texts: list[str]
    ) -> tuple[np.ndarray, np.ndarray, list[list[tuple[int, int]]]]:
        """Encode ``texts`` and recover per-token character spans.

        Returns:
            input_ids: ``[B, S] int64``. Padded to the longest sequence
                with ``_PAD_TOKEN_ID``.
            attention_mask: ``[B, S] int64``. 1 for real tokens, 0 for padding.
            char_spans: ``char_spans[i][j] = (char_start, char_end)`` of
                token ``j`` in ``texts[i]``. Exclusive end, character
                (not byte) offsets.
        """

        batch_ids: list[list[int]] = []
        batch_char_spans: list[list[tuple[int, int]]] = []

        for text in texts:
            ids = self._encoder.encode(text)
            spans = self._char_spans_for_tokens(text, ids)
            batch_ids.append(ids)
            batch_char_spans.append(spans)

        max_len = max((len(ids) for ids in batch_ids), default=0)
        # Always produce at least width 1 so downstream shape checks pass.
        if max_len == 0:
            max_len = 1

        B = len(batch_ids)
        input_ids = np.full((B, max_len), _PAD_TOKEN_ID, dtype=np.int64)
        attn = np.zeros((B, max_len), dtype=np.int64)
        for i, ids in enumerate(batch_ids):
            n = len(ids)
            if n:
                input_ids[i, :n] = np.asarray(ids, dtype=np.int64)
                attn[i, :n] = 1

        return input_ids, attn, batch_char_spans

    def _char_spans_for_tokens(
        self, text: str, token_ids: list[int]
    ) -> list[tuple[int, int]]:
        """Map each token to its character span in ``text``.

        Implementation: we know ``text.encode('utf-8')`` reproduces the
        bytes exactly, and tiktoken's ``decode_single_token_bytes`` gives
        us each token's UTF-8 bytes. Concatenating all token bytes in
        order reproduces the input bytes (for valid BPE encodings). We
        walk a byte cursor through the input, then convert each token's
        byte range to a character range.

        Mid-character splits
        --------------------
        tiktoken's ``o200k_base`` is byte-level BPE — a single Unicode
        character whose UTF-8 encoding is multi-byte can end up split
        across two tokens (emoji are the classic case). To keep the
        per-token char spans a clean partition of the source string, we
        assign each character to exactly *one* token: the token that
        contains its **last** UTF-8 byte. So a 4-byte emoji split 3/1
        across two tokens is wholly credited to the second token; the
        first token gets an empty ``(k, k)`` span. The alternative —
        duplicating the character on both sides — would make the token
        spans overlap and break span-extraction arithmetic.

        This is loss-free for span extraction: what matters is that the
        union of char spans for a ``B (I)* E`` run covers exactly the
        character range of the detected entity, and empty tokens at the
        start of a run contribute nothing to either ``start`` or ``end``.
        """

        if not token_ids:
            return []

        encoded = text.encode("utf-8")

        # Precompute, for each character, (first_byte, last_byte_exclusive).
        char_byte_end: list[int] = []  # index: char_i, value: byte index just past char
        byte_i = 0
        for ch in text:
            byte_i += len(ch.encode("utf-8"))
            char_byte_end.append(byte_i)

        # Token byte boundaries.
        token_byte_end: list[int] = []
        cursor = 0
        for tid in token_ids:
            cursor += len(self._encoder.decode_single_token_bytes(tid))
            if cursor > len(encoded):
                cursor = len(encoded)  # defensive clamp
            token_byte_end.append(cursor)

        # Walk characters and tokens in lock-step. Each character is
        # "owned" by the first token whose end-byte is >= the character's
        # end-byte — i.e., the token that contains the character's last
        # byte.
        spans: list[tuple[int, int]] = []
        tok_idx = 0
        n_tokens = len(token_ids)
        # Track the start-char of each token. Start char for token k is
        # the first character not yet assigned when token k begins; we
        # initialise all to len(text) and narrow as we go.
        tok_start_char = [len(text)] * n_tokens
        tok_end_char = [0] * n_tokens
        tok_has_char = [False] * n_tokens
        prev_char_assigned_to: int | None = None
        for char_i, end_byte in enumerate(char_byte_end):
            while tok_idx < n_tokens and token_byte_end[tok_idx] < end_byte:
                tok_idx += 1
            if tok_idx >= n_tokens:
                # Shouldn't happen for valid encodings; bail on remaining chars.
                break
            owner = tok_idx
            if not tok_has_char[owner]:
                tok_start_char[owner] = char_i
                tok_has_char[owner] = True
            tok_end_char[owner] = char_i + 1
            prev_char_assigned_to = owner

        # For tokens that received no character (because they only held
        # internal bytes of a char that was credited to the next token),
        # emit a zero-width span at the start position of the *next*
        # character that will appear. We anchor it to the boundary so
        # callers iterating spans still see contiguous char positions.
        running_char = 0
        for k in range(n_tokens):
            if tok_has_char[k]:
                spans.append((tok_start_char[k], tok_end_char[k]))
                running_char = tok_end_char[k]
            else:
                spans.append((running_char, running_char))

        return spans

    # ------------------------------------------------------------------
    # Viterbi decode + span extraction
    # ------------------------------------------------------------------
    def decode(
        self,
        logits: np.ndarray,
        char_spans: list[list[tuple[int, int]]],
        text_lens: list[int],
    ) -> list[list[Span]]:
        """Decode ``logits`` to per-text span lists.

        Args:
            logits: ``[B, S, num_labels] float``. The non-padded region is
                ``[:, :len(char_spans[i]), :]``; padded positions are
                ignored.
            char_spans: per-text list of ``(char_start, char_end)`` pairs,
                as returned by :meth:`tokenize`.
            text_lens: per-text number of *real* (non-padded) tokens.

        Returns:
            One list of ``Span`` per input text. Span labels are the
            BIOES body (e.g. ``"private_email"``), not the tagged form.
        """

        if logits.dtype != np.float64:
            logits = logits.astype(np.float64, copy=False)

        out: list[list[Span]] = []
        for i, n in enumerate(text_lens):
            if n <= 0:
                out.append([])
                continue
            seq_logits = logits[i, :n, :]
            tags = self._viterbi(seq_logits)
            spans = self._extract_spans(tags, char_spans[i])
            out.append(spans)
        return out

    def _viterbi(self, emit: np.ndarray) -> list[int]:
        """Constrained Viterbi over BIOES. ``emit`` is ``[T, num_labels]``."""

        T, n = emit.shape
        assert n == self.num_labels, f"logits width {n} != num_labels {self.num_labels}"

        # dp[t, k] = best score ending with tag k at step t.
        dp = np.full((T, n), _NEG_INF, dtype=np.float64)
        bp = np.zeros((T, n), dtype=np.int32)

        dp[0] = emit[0] + self._initial_mask()

        trans = self._transitions
        for t in range(1, T):
            # scores[k, j] = dp[t-1, k] + trans[k, j]
            scores = dp[t - 1, :, None] + trans  # [n, n]
            best_prev = np.argmax(scores, axis=0)  # [n]
            best_val = scores[best_prev, np.arange(n)]
            dp[t] = best_val + emit[t]
            bp[t] = best_prev

        # Apply final-tag constraint (can't end mid-entity).
        final_scores = dp[T - 1] + self._final_mask()
        last = int(np.argmax(final_scores))
        if final_scores[last] <= _NEG_INF / 2:
            # No legal path — fall back to argmax of last row. Shouldn't
            # happen in practice; defensive.
            last = int(np.argmax(dp[T - 1]))

        tags = [0] * T
        tags[T - 1] = last
        for t in range(T - 1, 0, -1):
            tags[t - 1] = int(bp[t, tags[t]])
        return tags

    def _extract_spans(
        self, tags: list[int], char_spans: list[tuple[int, int]]
    ) -> list[Span]:
        """Walk the decoded tag sequence and emit one Span per
        ``B (I)* E`` run or ``S`` singleton. The Viterbi output is
        guaranteed to be well-formed thanks to the transition mask, so
        this is a straight read-through.
        """

        out: list[Span] = []
        i = 0
        # Trim to the shorter of the two, just in case.
        n = min(len(tags), len(char_spans))
        while i < n:
            prefix, body = self._parsed[tags[i]]
            if prefix == "S":
                cs, ce = char_spans[i]
                out.append(Span(start=cs, end=ce, label=body))
                i += 1
            elif prefix == "B":
                start_char = char_spans[i][0]
                end_char = char_spans[i][1]
                label = body
                j = i + 1
                while j < n:
                    pj, bj = self._parsed[tags[j]]
                    if pj == "I" and bj == label:
                        end_char = char_spans[j][1]
                        j += 1
                        continue
                    if pj == "E" and bj == label:
                        end_char = char_spans[j][1]
                        j += 1
                        break
                    # Transition mask should prevent any other case.
                    break
                out.append(Span(start=start_char, end=end_char, label=label))
                i = j
            else:
                # O, stray I/E (shouldn't occur under the mask), or empty — skip.
                i += 1
        return out


def extract_id2label(config: Mapping[str, object]) -> dict[int, str]:
    """Parse ``id2label`` out of a Hugging Face ``config.json`` dict.

    The JSON spec keys the mapping with strings; we normalise to ints.
    """

    raw = config.get("id2label")
    if not isinstance(raw, Mapping):
        raise ValueError("config.json is missing an 'id2label' mapping")
    return {int(k): str(v) for k, v in raw.items()}


__all__ = [
    "PostProcessor",
    "load_calibration",
    "extract_id2label",
]
