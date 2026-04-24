"""Placeholder minting and reverse-lookup map.

Placeholder shape: ``⟦{LABEL}_{hex8}⟧`` where ``hex8`` is the first 8 hex chars
of a blake2b keyed hash of the original string. Delimiters are U+27E6 and
U+27E7 — rare in natural prompts and single Unicode codepoints, so they encode
to a stable 3-byte UTF-8 sequence and won't fragment the way ``[[`` can across
SSE chunks.

Determinism: the same (original, label) pair always mints the same placeholder
within a session, which keeps multi-turn conversations coherent (Claude Code
re-sends the full history on every turn; prior-turn references must still
resolve).
"""

from __future__ import annotations

import hashlib
import re
import threading

from ..detection.base import Span


__all__ = ["Span", "PlaceholderMap", "apply_spans", "DELIM_OPEN", "DELIM_CLOSE"]


DELIM_OPEN = "⟦"
DELIM_CLOSE = "⟧"

# Labels come from the detector's fixed BIOES class set — ASCII-uppercase and
# underscores only. Restricting the regex to that shape prevents accidentally
# chewing up legitimate prose that happens to sit between ⟦ and ⟧.
_PLACEHOLDER_RE = re.compile(
    rf"{re.escape(DELIM_OPEN)}([A-Z][A-Z0-9_]*)_([0-9a-f]{{8}}){re.escape(DELIM_CLOSE)}"
)


class PlaceholderMap:
    """Session-scoped bidirectional map between originals and placeholders.

    Thread-safe: a single ``threading.Lock`` guards both dicts. The map is
    shared across concurrent proxied requests, and mint/lookup are both short
    critical sections so a single lock is simpler than separating reads and
    writes.
    """

    __slots__ = ("_session_key", "_by_placeholder", "_by_original", "_lock")

    def __init__(self, session_key: bytes) -> None:
        if not isinstance(session_key, (bytes, bytearray)):
            raise TypeError("session_key must be bytes")
        if len(session_key) == 0:
            raise ValueError("session_key must be non-empty")
        # blake2b's key parameter accepts up to 64 bytes; 32 is typical.
        self._session_key = bytes(session_key)
        self._by_placeholder: dict[str, str] = {}
        self._by_original: dict[tuple[str, str], str] = {}
        self._lock = threading.Lock()

    def _mint(self, original: str, label: str) -> str:
        digest = hashlib.blake2b(
            original.encode("utf-8"), key=self._session_key, digest_size=4
        ).hexdigest()
        return f"{DELIM_OPEN}{label}_{digest}{DELIM_CLOSE}"

    def mask(self, original: str, label: str) -> str:
        """Return the placeholder for ``original`` under ``label``, minting if new.

        Same (original, label) pair always returns the same placeholder within
        the life of this map.
        """
        key = (label, original)
        with self._lock:
            cached = self._by_original.get(key)
            if cached is not None:
                return cached
            placeholder = self._mint(original, label)
            # Collision across different originals under the same label would
            # mean an 8-hex blake2b keyed-hash collision — astronomically
            # unlikely but we'd rather fail loud than silently corrupt data.
            existing = self._by_placeholder.get(placeholder)
            if existing is not None and existing != original:
                raise RuntimeError(
                    f"placeholder collision for label {label!r}"
                )
            self._by_placeholder[placeholder] = original
            self._by_original[key] = placeholder
            return placeholder

    def unmask(self, placeholder: str) -> str | None:
        """Return the original for a known placeholder, else ``None``."""
        with self._lock:
            return self._by_placeholder.get(placeholder)

    @property
    def pattern(self) -> re.Pattern[str]:
        """Compiled regex that matches any well-formed placeholder."""
        return _PLACEHOLDER_RE

    def __len__(self) -> int:
        with self._lock:
            return len(self._by_placeholder)

    def __bool__(self) -> bool:
        # Without this, an empty map is falsy (because ``__len__`` returns 0),
        # so ``x or PlaceholderMap(...)`` would silently replace a real but
        # empty caller-supplied map with a fresh one. The map is a live
        # collaborator, not a container whose emptiness should gate behavior.
        return True

    def __repr__(self) -> str:
        # Never expose plaintext in repr; this object sits in tracebacks.
        return f"PlaceholderMap(entries={len(self)})"


def apply_spans(text: str, spans: list[Span], pmap: PlaceholderMap) -> str:
    """Replace each span in ``text`` with its placeholder from ``pmap``.

    Overlap policy: **longest-wins**. When two spans overlap, the one covering
    more characters is kept; ties break by earlier start offset. All spans are
    then processed in reverse-offset order so earlier indices remain valid as
    we splice from the tail backwards.

    Zero-length spans (``start == end``) are dropped — there's nothing to mask.
    Spans out of ``[0, len(text)]`` raise ``ValueError`` rather than silently
    corrupting data.
    """
    if not spans:
        return text

    for s in spans:
        if s.start < 0 or s.end > len(text) or s.start > s.end:
            raise ValueError(f"span {s!r} out of bounds for text len {len(text)}")

    kept = _resolve_overlaps(spans)
    kept.sort(key=lambda s: s.start, reverse=True)

    buf = text
    for s in kept:
        if s.start == s.end:
            continue
        original = buf[s.start : s.end]
        placeholder = pmap.mask(original, s.label)
        buf = buf[: s.start] + placeholder + buf[s.end :]
    return buf


def _resolve_overlaps(spans: list[Span]) -> list[Span]:
    # Sort by (start asc, length desc) so when we sweep left-to-right, the
    # longest candidate at each starting position is seen first. Any later
    # span that overlaps it gets discarded.
    ordered = sorted(spans, key=lambda s: (s.start, -(s.end - s.start)))
    kept: list[Span] = []
    for cand in ordered:
        if cand.start == cand.end:
            continue
        clobbered = False
        for i, existing in enumerate(kept):
            if cand.start < existing.end and existing.start < cand.end:
                cand_len = cand.end - cand.start
                existing_len = existing.end - existing.start
                if cand_len > existing_len:
                    kept[i] = cand
                clobbered = True
                break
        if not clobbered:
            kept.append(cand)
    return kept
