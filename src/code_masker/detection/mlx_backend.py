"""MLX backend for the privacy-filter detector.

The model is a token-classification encoder (``OpenAIPrivacyFilterForTokenClassification``),
not a causal LM, so it loads via ``mlx_embeddings.utils.load`` rather than
``mlx_lm.load``. That detail matters: ``mlx_lm`` only recognizes decoder
architectures from its built-in model registry, and silently fails with
``Model type openai_privacy_filter not supported`` on this one.

Expected model layout (``mlx-community/openai-privacy-filter-8bit``)::

    <model_path>/
        config.json          # has id2label
        tokenizer.json       # unused — we tokenize with tiktoken for ONNX parity
        model.safetensors    # quantized weights + classifier head

All MLX imports are lazy so this module can be imported on non-Mac
platforms without the MLX wheels installed.
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
    import mlx.core as mx


class MLXDetector:
    """Privacy-filter detector running on Apple Silicon via MLX."""

    def __init__(
        self,
        model_path: str | Path,
        calibration_path: str | Path,
        max_batch: int = 8,
    ) -> None:
        try:
            import mlx.core as mx
            from mlx_embeddings.utils import load as mlx_embed_load  # type: ignore
        except ImportError as exc:  # pragma: no cover - env-specific
            raise ImportError(
                "MLXDetector requires the 'mlx' and 'mlx-embeddings' packages. "
                "Install them (Apple Silicon only) via `pip install mlx mlx-embeddings`."
            ) from exc

        self._mx = mx
        self._max_batch = max_batch
        self._model_path = Path(model_path).expanduser()

        config = json.loads(
            (self._model_path / "config.json").read_text(encoding="utf-8")
        )
        id2label = extract_id2label(config)
        calibration = load_calibration(calibration_path)
        self._post = PostProcessor(id2label=id2label, calibration=calibration)

        # mlx-embeddings bundles a tokenizer, but we tokenize with tiktoken
        # (the model's reference tokenizer is o200k_base) to keep the MLX and
        # ONNX backends bit-identical in their post-processing inputs.
        self._model, _tokenizer = mlx_embed_load(str(self._model_path))
        self._warmed_up = False

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
            logits_np = self._forward_logits(input_ids, attn)
            decoded = self._post.decode(logits_np, char_spans, text_lens)
            for i, spans in enumerate(decoded):
                all_spans[batch_start + i] = spans
        return all_spans  # type: ignore[return-value]

    # ------------------------------------------------------------------
    def _forward_logits(
        self, input_ids: np.ndarray, attention_mask: np.ndarray
    ) -> np.ndarray:
        mx = self._mx
        ids = mx.array(input_ids)
        mask = mx.array(attention_mask)
        out = self._model(ids, attention_mask=mask)

        logits = getattr(out, "logits", out)
        # mx.eval forces evaluation of the lazy graph before the host copy.
        mx.eval(logits)
        arr = np.asarray(logits)
        if arr.ndim != 3:
            raise RuntimeError(
                f"Expected logits shape [B, S, num_labels], got {arr.shape}."
            )
        return arr.astype(np.float32, copy=False)


__all__ = ["MLXDetector"]
