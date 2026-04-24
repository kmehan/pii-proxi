"""Anthropic Messages API adapter.

Mounted at ``/anthropic`` so clients configure ``ANTHROPIC_BASE_URL`` to
``http://127.0.0.1:{port}/anthropic``. The upstream path remains ``/v1/messages``
matching the public API shape.
"""

from __future__ import annotations

from fastapi import APIRouter, Request, Response

from ..masking.extractor import extract_anthropic
from ._common import proxy_roundtrip


router = APIRouter(prefix="/anthropic", tags=["anthropic"])


@router.post("/v1/messages")
async def messages(request: Request) -> Response:
    upstream_base = request.app.state.config.anthropic_upstream.rstrip("/")
    return await proxy_roundtrip(
        request,
        f"{upstream_base}/v1/messages",
        extract_anthropic,
    )
