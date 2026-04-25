"""Detection subsystem for pii-proxi.

Exports the ``Detector`` protocol and ``Span`` dataclass plus the backend
implementations. Backend modules import their heavy dependencies
(``mlx``, ``onnxruntime``) lazily so the package can be imported on any
platform regardless of which extras are installed.
"""

from pii_proxi.detection.base import Detector, Span
from pii_proxi.detection.postprocess import PostProcessor

__all__ = ["Detector", "Span", "PostProcessor"]
