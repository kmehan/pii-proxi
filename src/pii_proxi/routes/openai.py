"""OpenAI Chat Completions adapter (factory: build a router for one provider)."""

from __future__ import annotations

from fastapi import APIRouter, Request, Response

from ..masking.extractor import extract_openai
from ._common import proxy_roundtrip


def make_router(name: str, upstream: str) -> APIRouter:
    router = APIRouter(prefix=f"/{name}", tags=[name])
    upstream_base = upstream.rstrip("/")

    @router.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        return await proxy_roundtrip(
            request,
            f"{upstream_base}/v1/chat/completions",
            extract_openai,
        )

    return router
