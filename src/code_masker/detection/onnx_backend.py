"""ONNX backend for the privacy-filter detector.

Expected model layout (``yasserrmd/privacy-filter-ONNX``):

    <model_path>/
        config.json           # contains id2label
        onnx/model_fp16.onnx  # FP16 weights + FP32 router graph

Inputs:  ``input_ids`` ``[B, S] int64``; ``attention_mask`` ``[B, S] int64``.
Output:  ``logits`` ``[B, S, 33] float32``.

We import ``onnxruntime`` lazily so this module stays importable on
systems without ORT. Providers are preferred in the order
``CoreMLExecutionProvider, CUDAExecutionProvider, CPUExecutionProvider``;
ORT silently falls through to CPU if the others aren't available.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from code_masker.detection.base import Span
from code_masker.detection.postprocess import (
    PostProcessor,
    extract_id2label,
    load_calibration,
)

if TYPE_CHECKING:  # pragma: no cover
    import onnxruntime as ort

_DEFAULT_PROVIDERS = (
    "CoreMLExecutionProvider",
    "CUDAExecutionProvider",
    "CPUExecutionProvider",
)


class ONNXDetector:
    """Privacy-filter detector running through onnxruntime."""

    def __init__(
        self,
        model_path: str | Path,
        calibration_path: str | Path,
        max_batch: int = 8,
        providers: tuple[str, ...] = _DEFAULT_PROVIDERS,
    ) -> None:
        try:
            import onnxruntime as ort  # type: ignore
        except ImportError as exc:  # pragma: no cover - env-specific
            raise ImportError(
                "ONNXDetector requires the 'onnxruntime' package. "
                "Install it via `pip install onnxruntime` (or "
                "`onnxruntime-gpu` / the CoreML build on macOS)."
            ) from exc

        self._ort = ort
        self._max_batch = max_batch
        self._model_path = Path(model_path).expanduser()

        config = json.loads(
            (self._model_path / "config.json").read_text(encoding="utf-8")
        )
        id2label = extract_id2label(config)
        calibration = load_calibration(calibration_path)
        self._post = PostProcessor(id2label=id2label, calibration=calibration)

        onnx_file = self._resolve_onnx_file(self._model_path)
        # Filter providers down to those ORT was built with to avoid noisy
        # warnings; ORT would accept unknowns but we're polite.
        available = set(ort.get_available_providers())
        chosen = [p for p in providers if p in available] or ["CPUExecutionProvider"]
        self._session = ort.InferenceSession(str(onnx_file), providers=chosen)

        # Cache input/output names so per-call lookup is trivial.
        names = {inp.name for inp in self._session.get_inputs()}
        self._input_ids_name = "input_ids" if "input_ids" in names else next(iter(names))
        self._attn_name = (
            "attention_mask" if "attention_mask" in names else None
        )
        self._output_name = self._session.get_outputs()[0].name
        self._warmed_up = False

    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_onnx_file(model_path: Path) -> Path:
        """Pick the best available ONNX file under ``model_path``.

        Preference order:
          1. ``onnx/model_fp16.onnx`` (the advertised file)
          2. ``onnx/model.onnx``
          3. ``model.onnx`` in the root
        """

        candidates = [
            model_path / "onnx" / "model_fp16.onnx",
            model_path / "onnx" / "model.onnx",
            model_path / "model.onnx",
        ]
        for c in candidates:
            if c.is_file():
                return c
        raise FileNotFoundError(
            f"No ONNX model found under {model_path} (looked for "
            f"onnx/model_fp16.onnx, onnx/model.onnx, model.onnx)."
        )

    # ------------------------------------------------------------------
    def warmup(self) -> None:
        if self._warmed_up:
            return
        self.detect([" "])
        self._warmed_up = True

    # ------------------------------------------------------------------
    def detect(self, texts: list[str]) -> list[list[Span]]:
        if not texts:
            return []

        all_spans: list[list[Span]] = [None] * len(texts)  # type: ignore[list-item]
        for batch_start in range(0, len(texts), self._max_batch):
            batch = texts[batch_start : batch_start + self._max_batch]
            input_ids, attn, char_spans = self._post.tokenize(batch)
            text_lens = [int(attn[i].sum()) for i in range(attn.shape[0])]
            feeds: dict[str, np.ndarray] = {self._input_ids_name: input_ids}
            if self._attn_name is not None:
                feeds[self._attn_name] = attn
            logits = self._session.run([self._output_name], feeds)[0]
            if logits.dtype != np.float32:
                logits = logits.astype(np.float32, copy=False)
            decoded = self._post.decode(logits, char_spans, text_lens)
            for i, spans in enumerate(decoded):
                all_spans[batch_start + i] = spans
        return all_spans  # type: ignore[return-value]


__all__ = ["ONNXDetector"]
