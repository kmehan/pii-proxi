"""Anthropic Messages API adapter (factory: build a router for one provider)."""

from __future__ import annotations

from fastapi import APIRouter, Request, Response

from ..masking.extractor import extract_anthropic
from ._common import proxy_roundtrip


def make_router(name: str, upstream: str) -> APIRouter:
    router = APIRouter(prefix=f"/{name}", tags=[name])
    upstream_base = upstream.rstrip("/")

    @router.post("/v1/messages")
    async def messages(request: Request) -> Response:
        return await proxy_roundtrip(
            request,
            f"{upstream_base}/v1/messages",
            extract_anthropic,
        )

    return router
