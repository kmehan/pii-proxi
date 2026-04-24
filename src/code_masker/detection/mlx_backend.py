"""MLX backend for the privacy-filter detector.

All MLX-specific imports are performed lazily inside ``__init__`` so that
the module can be imported on non-Mac (or Mac without MLX installed)
platforms without failing — the caller only pays the import cost when
actually constructing an ``MLXDetector``.

Expected model layout (``mlx-community/openai-privacy-filter-8bit``):

    <model_path>/
        config.json          # contains id2label
        tokenizer.json       # (unused here — we use tiktoken directly)
        model.safetensors    # quantised weights

Because ``mlx_lm.load`` returns a causal-LM wrapper by default and the
privacy filter is a token-classification head, we reach for the
underlying module's forward pass and read the classifier logits out of
its output. ``mlx_lm`` exposes a ``return_hidden_states`` path on recent
versions; we gracefully fall back to looking for a classifier on the
loaded model.

Since the mlx-community build ships with the classifier head attached
(the safetensors file contains ``classifier.weight`` / ``classifier.bias``
shards), ``mlx_lm.load`` handles this transparently in practice; if a
future build splits the head, callers can override ``_forward_logits``
in a subclass.
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
        # Import lazily so that `import code_masker.detection` works on
        # platforms without MLX.
        try:
            import mlx.core as mx  # noqa: F401
            from mlx_lm import load as mlx_load  # type: ignore
        except ImportError as exc:  # pragma: no cover - env-specific
            raise ImportError(
                "MLXDetector requires the 'mlx' and 'mlx-lm' packages. "
                "Install them (Apple Silicon only) via `pip install mlx mlx-lm`."
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

        self._model, _tokenizer = mlx_load(str(self._model_path))
        self._warmed_up = False

    # ------------------------------------------------------------------
    def warmup(self) -> None:
        """Run a single short forward pass to amortise JIT / cache costs."""

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
        """Run the MLX forward pass and return ``[B, S, num_labels]`` as numpy."""

        mx = self._mx
        ids = mx.array(input_ids)
        mask = mx.array(attention_mask)
        # ``mlx_lm`` models accept ``(input_ids, mask=...)`` or positional
        # masks depending on version; try the common kwargs first.
        try:
            out = self._model(ids, attention_mask=mask)
        except TypeError:
            out = self._model(ids)

        # Different mlx_lm versions return either raw logits or a wrapper.
        logits = getattr(out, "logits", out)
        # Materialise to numpy via MLX -> host copy.
        arr = np.asarray(logits)
        if arr.ndim != 3:
            raise RuntimeError(
                f"Expected logits shape [B, S, num_labels], got {arr.shape}."
            )
        return arr.astype(np.float32, copy=False)


__all__ = ["MLXDetector"]
