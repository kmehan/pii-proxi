"""Shared test fixtures + ``src/`` importability shim.

The ``sys.path`` shim keeps tests runnable before ``pip install -e .`` has
been run (agents may run tests in isolated order during integration). Once
the package is installed into the venv, the shim is a harmless no-op.

``FakeDetector`` exists so route tests don't have to load the real MLX /
ONNX model — they seed exact text -> spans mappings per test.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


@dataclass(slots=True, frozen=True)
class FakeSpan:
    """Duck-compatible with both ``detection.base.Span`` and
    ``masking.placeholder.Span``; consumers read only the three fields."""

    start: int
    end: int
    label: str


class FakeDetector:
    """Detector returning pre-seeded spans by exact input-text match."""

    def __init__(self, fixture: dict[str, list[FakeSpan]] | None = None) -> None:
        self.fixture = fixture or {}
        self.calls: list[list[str]] = []
        self.warmup_calls = 0

    def detect(self, texts: list[str]) -> list[list[Any]]:
        self.calls.append(list(texts))
        return [list(self.fixture.get(t, [])) for t in texts]

    def warmup(self) -> None:
        self.warmup_calls += 1


@pytest.fixture
def fake_detector() -> FakeDetector:
    return FakeDetector()
