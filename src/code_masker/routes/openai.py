"""OpenAI Chat Completions adapter.

Mounted at ``/openai`` so clients configure ``OPENAI_BASE_URL`` to
``http://127.0.0.1:{port}/openai/v1``. The upstream path remains
``/v1/chat/completions``; other tools (aider, continue.dev, Cursor BYO-key)
ride through this same route because they all speak the OpenAI-compat shape.
"""

from __future__ import annotations

from fastapi import APIRouter, Request, Response

from ..masking.extractor import extract_openai
from ._common import proxy_roundtrip


router = APIRouter(prefix="/openai", tags=["openai"])


@router.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Response:
    upstream_base = request.app.state.config.openai_upstream.rstrip("/")
    return await proxy_roundtrip(
        request,
        f"{upstream_base}/v1/chat/completions",
        extract_openai,
    )
