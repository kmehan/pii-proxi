"""Detection subsystem for code-masker.

Exports the ``Detector`` protocol and ``Span`` dataclass plus the backend
implementations. Backend modules import their heavy dependencies
(``mlx``, ``onnxruntime``) lazily so the package can be imported on any
platform regardless of which extras are installed.
"""

from code_masker.detection.base import Detector, Span
from code_masker.detection.postprocess import PostProcessor

__all__ = ["Detector", "Span", "PostProcessor"]
