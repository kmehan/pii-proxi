"""Core types for the detection subsystem.

The ``Span`` dataclass defined here is the single canonical type shared
between detector backends and the masking pipeline. All offsets are
expressed as **character** (i.e. ``str``-index) offsets into the original
source string — never byte or token offsets. This matters because
downstream placeholder injection uses Python string slicing, which is
character-based. ``masking.placeholder`` re-exports this class.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(slots=True, frozen=True)
class Span:
    """A detected entity span in a source string.

    Attributes:
        start: Inclusive character offset into the source string.
        end: Exclusive character offset into the source string.
        label: Entity class label, e.g. ``"private_email"`` or
            ``"secret"``. The label is the BIOES *body* — the
            ``B-``/``I-``/``E-``/``S-`` prefix has already been consumed
            by the Viterbi decoder.
    """

    start: int
    end: int
    label: str


@runtime_checkable
class Detector(Protocol):
    """Backend-agnostic detector interface.

    Implementations batch a list of source strings through a model,
    run shared BIOES post-processing, and return one list of spans per
    input string, in input order.
    """

    def detect(self, texts: list[str]) -> list[list[Span]]:
        """Classify ``texts`` and return detected spans per input."""
        ...

    def warmup(self) -> None:
        """Perform any one-shot initialization (load weights, run a
        throwaway forward pass to prime caches, etc.). Safe to call
        multiple times; subsequent calls should be no-ops."""
        ...
