"""Parity test: MLX and ONNX detectors must produce identical spans.

Both backends share ``PostProcessor``, so the only way they can diverge
is if one of them feeds different logits into the decoder (e.g. from a
quantisation mismatch). We don't assert numeric logit equality — only
that the final ``(start, end, label)`` tuples agree.

Skipped automatically unless:
* both ``mlx``/``mlx_lm`` and ``onnxruntime`` are importable, AND
* both model paths are provided via env vars
  (``CODE_MASKER_MODEL_MLX`` / ``CODE_MASKER_MODEL_ONNX``), AND
* a calibration file path is provided via
  ``CODE_MASKER_CALIBRATION``.

Set these once you have the models cached locally. Example::

    export CODE_MASKER_MODEL_MLX=~/.cache/code-masker/models/mlx-8bit
    export CODE_MASKER_MODEL_ONNX=~/.cache/code-masker/models/onnx-fp16
    export CODE_MASKER_CALIBRATION=~/.cache/code-masker/calibration/viterbi_calibration.json
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest


FIXTURES = [
    "My API key is sk-live-ABCDEF1234567890 and the email is alice@example.com.",
    "Call me at +1 (415) 555-0142 on Monday.",
    "Ship logs to ops@corp.internal — token ghp_0123456789abcdef.",
    "Dr. Ada Lovelace lives at 221B Baker Street, London NW1 6XE.",
    "Visit https://example.com/reset?token=0xdeadbeefcafebabe for details.",
    "Nothing sensitive here, just a plain sentence about cats.",
    "DOB 1987-03-14, SSN 123-45-6789, account 4242 4242 4242 4242.",
    "Contact naïve@example.com or admin@ümlauts.test (unicode matters).",
]


def _deps_available() -> tuple[bool, str]:
    for mod in ("mlx", "mlx_lm", "onnxruntime"):
        if importlib.util.find_spec(mod) is None:
            return False, f"missing dependency: {mod}"
    for env in ("CODE_MASKER_MODEL_MLX", "CODE_MASKER_MODEL_ONNX", "CODE_MASKER_CALIBRATION"):
        val = os.environ.get(env)
        if not val:
            return False, f"missing env var: {env}"
        if not Path(val).expanduser().exists():
            return False, f"path from {env} does not exist: {val}"
    return True, ""


_ok, _reason = _deps_available()


@pytest.mark.skipif(not _ok, reason=_reason or "parity test prerequisites not met")
def test_mlx_and_onnx_agree_on_fixtures() -> None:
    from code_masker.detection.mlx_backend import MLXDetector
    from code_masker.detection.onnx_backend import ONNXDetector

    calibration = os.environ["CODE_MASKER_CALIBRATION"]
    mlx_det = MLXDetector(os.environ["CODE_MASKER_MODEL_MLX"], calibration)
    onnx_det = ONNXDetector(os.environ["CODE_MASKER_MODEL_ONNX"], calibration)

    mlx_spans = mlx_det.detect(FIXTURES)
    onnx_spans = onnx_det.detect(FIXTURES)

    assert len(mlx_spans) == len(onnx_spans) == len(FIXTURES)
    for i, (a, b) in enumerate(zip(mlx_spans, onnx_spans)):
        a_tuples = sorted((s.start, s.end, s.label) for s in a)
        b_tuples = sorted((s.start, s.end, s.label) for s in b)
        assert a_tuples == b_tuples, (
            f"parity mismatch on fixture {i!r}: MLX={a_tuples} ONNX={b_tuples}"
        )
