"""Integration tests for the OpenAI Chat Completions route."""

from __future__ import annotations

import json
import re

import httpx
import respx
from fastapi.testclient import TestClient

from pii_proxi.config import Config
from pii_proxi.masking.placeholder import PlaceholderMap
from pii_proxi.server import create_app
from pii_proxi.session import new_session_key

from .conftest import FakeDetector, FakeSpan


UPSTREAM = "https://api.openai.com"
SECRET = "sk-live-AAAABBBB"
PROMPT = f"here is my key {SECRET} — write a python snippet"


def _make_app(detector: FakeDetector, pmap: PlaceholderMap | None = None):
    cfg = Config(anthropic_upstream="https://api.anthropic.com", openai_upstream=UPSTREAM)
    return create_app(config=cfg, detector=detector, placeholder_map=pmap)


def test_non_streaming_masks_request_and_unmasks_response():
    secret_start = PROMPT.index(SECRET)
    detector = FakeDetector({
        PROMPT: [FakeSpan(secret_start, secret_start + len(SECRET), "SECRET")],
    })
    pmap = PlaceholderMap(new_session_key())
    app = _make_app(detector, pmap=pmap)

    placeholder_re = re.compile(r"⟦SECRET_[0-9a-f]{8}⟧")

    with respx.mock(assert_all_called=True) as mock:
        route = mock.post(f"{UPSTREAM}/v1/chat/completions").mock(
            return_value=httpx.Response(200, json={"ok": True})
        )

        with TestClient(app) as client:
            ph = pmap.mask(SECRET, "SECRET")
            route.mock(
                return_value=httpx.Response(
                    200,
                    json={
                        "id": "chatcmpl-x",
                        "choices": [
                            {
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": f"use {ph} in your code",
                                },
                                "finish_reason": "stop",
                            }
                        ],
                    },
                )
            )

            resp = client.post(
                "/openai/v1/chat/completions",
                headers={
                    "authorization": "Bearer sk-openai-fake",
                    "content-type": "application/json",
                },
                json={
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": PROMPT}],
                },
            )

    assert resp.status_code == 200

    sent = json.loads(route.calls.last.request.content)
    assert SECRET not in json.dumps(sent)
    assert placeholder_re.search(sent["messages"][0]["content"]) is not None

    assert route.calls.last.request.headers["authorization"] == "Bearer sk-openai-fake"

    body = resp.json()
    content_out = body["choices"][0]["message"]["content"]
    assert SECRET in content_out
    assert "⟦" not in content_out


def test_streaming_sse_unmasks_placeholders_back_to_plaintext():
    secret_start = PROMPT.index(SECRET)
    detector = FakeDetector({
        PROMPT: [FakeSpan(secret_start, secret_start + len(SECRET), "SECRET")],
    })
    pmap = PlaceholderMap(new_session_key())
    app = _make_app(detector, pmap=pmap)

    ph = pmap.mask(SECRET, "SECRET")

    sse_body = (
        b'data: {"choices":[{"delta":{"content":"use '
        + ph.encode()
        + b' now"}}]}\n\n'
        b"data: [DONE]\n\n"
    )

    with respx.mock(assert_all_called=True) as mock:
        route = mock.post(f"{UPSTREAM}/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=sse_body,
            )
        )

        with TestClient(app) as client:
            resp = client.post(
                "/openai/v1/chat/completions",
                headers={
                    "authorization": "Bearer sk-openai-fake",
                    "accept": "text/event-stream",
                    "content-type": "application/json",
                },
                json={
                    "model": "gpt-4o",
                    "stream": True,
                    "messages": [{"role": "user", "content": PROMPT}],
                },
            )

    assert resp.status_code == 200
    sent = json.loads(route.calls.last.request.content)
    assert SECRET not in json.dumps(sent)

    received = resp.content
    assert SECRET.encode() in received
    assert ph.encode() not in received


def test_detector_is_called_with_batched_texts():
    detector = FakeDetector({"alpha": [], "beta": []})
    app = _make_app(detector)

    with respx.mock() as mock:
        mock.post(f"{UPSTREAM}/v1/chat/completions").mock(
            return_value=httpx.Response(200, json={"ok": True})
        )

        with TestClient(app) as client:
            client.post(
                "/openai/v1/chat/completions",
                headers={"authorization": "Bearer k"},
                json={
                    "model": "gpt-4o",
                    "messages": [
                        {"role": "user", "content": "alpha"},
                        {"role": "user", "content": "beta"},
                    ],
                },
            )

    # Both texts batched into a single detector call (the point of the
    # batch API — one model forward pass for the whole request body).
    assert len(detector.calls) == 1
    assert detector.calls[0] == ["alpha", "beta"]


def test_upstream_500_passes_through():
    detector = FakeDetector()
    app = _make_app(detector)

    with respx.mock() as mock:
        mock.post(f"{UPSTREAM}/v1/chat/completions").mock(
            return_value=httpx.Response(500, json={"error": "boom"})
        )

        with TestClient(app) as client:
            resp = client.post(
                "/openai/v1/chat/completions",
                headers={"authorization": "Bearer k"},
                json={
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )

    assert resp.status_code == 500
    assert resp.json() == {"error": "boom"}
