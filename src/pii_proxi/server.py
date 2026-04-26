"""FastAPI app factory + detector backend factory.

The app is wired together in :func:`create_app` so tests can inject a fake
detector or a fake placeholder map without monkey-patching module globals.
Lifespan owns the ``httpx.AsyncClient`` so we reuse connections across
requests (coding-assistant sessions are chatty; connection reuse matters).
"""

from __future__ import annotations

import contextlib
import logging
from typing import Any, AsyncIterator, Optional

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .config import Config
from .detection.base import Detector
from .masking.placeholder import PlaceholderMap
from .routes import make_anthropic_router, make_openai_router
from .session import new_session_key


log = logging.getLogger("pii_proxi.server")


def backend_factory(name: str, config: Config) -> Detector:
    """Instantiate the configured detector backend.

    Import is lazy so a user running the MLX build doesn't trip over missing
    ``onnxruntime`` (and vice versa).
    """
    if name == "mlx":
        from .detection.mlx_backend import MLXDetector  # type: ignore[attr-defined]

        return MLXDetector(
            model_path=config.model_path,
            calibration_path=config.calibration_path,
        )
    if name == "onnx":
        from .detection.onnx_backend import ONNXDetector  # type: ignore[attr-defined]

        return ONNXDetector(
            model_path=config.model_path,
            calibration_path=config.calibration_path,
        )
    raise ValueError(f"unknown backend: {name!r}")


def create_app(
    config: Optional[Config] = None,
    detector: Optional[Detector] = None,
    placeholder_map: Optional[PlaceholderMap] = None,
    http_client: Optional[httpx.AsyncClient] = None,
) -> FastAPI:
    """Build the FastAPI app.

    All collaborators are injectable so tests can stand up the app with a
    fake detector and a fake upstream (via ``respx``) without touching disk.
    """
    cfg = config or Config.load()

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.config = cfg

        if detector is not None:
            app.state.detector = detector
        else:
            app.state.detector = backend_factory(cfg.backend, cfg)
            # ``warmup`` is cheap for fakes and amortizes first-token latency
            # for real backends. Best-effort — log and continue on failure.
            try:
                app.state.detector.warmup()
            except Exception as e:  # pragma: no cover — backend-specific
                log.warning("detector warmup failed: %s", e)

        # Explicit ``is None`` — ``PlaceholderMap`` defines ``__len__``, so a
        # freshly-constructed (empty) map is falsy. ``or`` would silently
        # replace the caller's injected map with a new one.
        app.state.placeholder_map = (
            placeholder_map
            if placeholder_map is not None
            else PlaceholderMap(new_session_key())
        )

        owns_client = http_client is None
        # Explicit timeouts: streaming responses can legitimately hang for
        # minutes on a slow model; ``None`` for read gives us that patience
        # while still capping connect at a sane value.
        client = http_client or httpx.AsyncClient(
            timeout=httpx.Timeout(connect=10.0, read=None, write=60.0, pool=10.0),
            http2=False,
        )
        app.state.http_client = client
        try:
            yield
        finally:
            if owns_client:
                await client.aclose()

    app = FastAPI(title="pii-proxi", lifespan=lifespan)
    for name, p in cfg.providers.items():
        if p.format == "anthropic":
            app.include_router(make_anthropic_router(name, p.upstream))
        else:
            app.include_router(make_openai_router(name, p.upstream))
        log.info("mounted %s (%s) -> %s", name, p.format, p.upstream)

    @app.get("/healthz")
    async def healthz(request: Request) -> dict[str, Any]:
        det = getattr(request.app.state, "detector", None)
        return {
            "status": "ok",
            "backend": cfg.backend,
            "detector_loaded": det is not None,
            "session_entries": len(request.app.state.placeholder_map),
        }

    @app.post("/admin/clear-session")
    async def clear_session(request: Request) -> JSONResponse:
        # 127.0.0.1 bind is the authz boundary. If someone explicitly rebinds
        # to a LAN interface they've opted out of that protection; we don't
        # try to second-guess that here.
        request.app.state.placeholder_map = PlaceholderMap(new_session_key())
        return JSONResponse({"status": "ok", "message": "session cleared"})

    return app
