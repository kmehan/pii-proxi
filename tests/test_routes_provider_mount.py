"""Custom-name provider mounts: confirm the route prefix follows the
provider key in ``cfg.providers`` rather than the format. Same round-trip
masking assertions as the legacy single-provider tests, just hitting a
different URL prefix.
"""

from __future__ import annotations

import json
import re

import httpx
import respx
from fastapi.testclient import TestClient

from pii_proxi.config import Config, ProviderConfig
from pii_proxi.masking.placeholder import PlaceholderMap
from pii_proxi.server import create_app
from pii_proxi.session import new_session_key

from .conftest import FakeDetector, FakeSpan


UPSTREAM = "https://upstream.example.com"
SAMPLE = "Ada Lovelace, ada@example.com"
PROMPT = f"please greet {SAMPLE} for me"


def test_anthropic_format_provider_under_custom_name():
    name_start = PROMPT.index(SAMPLE)
    detector = FakeDetector({
        PROMPT: [FakeSpan(name_start, name_start + len(SAMPLE), "private_person")],
    })
    pmap = PlaceholderMap(new_session_key())
    cfg = Config(
        providers={
            "claude": ProviderConfig(format="anthropic", upstream=UPSTREAM),
        }
    )
    app = create_app(config=cfg, detector=detector, placeholder_map=pmap)

    placeholder_re = re.compile(r"⟦private_person_[0-9a-f]{8}⟧")

    with respx.mock(assert_all_called=True) as mock:
        route = mock.post(f"{UPSTREAM}/v1/messages").mock(
            return_value=httpx.Response(200, json={"content": []})
        )

        with TestClient(app) as client:
            ph = pmap.mask(SAMPLE, "private_person")
            route.mock(
                return_value=httpx.Response(
                    200,
                    json={"content": [{"type": "text", "text": f"hi {ph}"}]},
                )
            )

            resp = client.post(
                "/claude/v1/messages",
                headers={
                    "x-api-key": "sk-ant-fake",
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": "claude-3-opus-20240229",
                    "max_tokens": 64,
                    "messages": [{"role": "user", "content": PROMPT}],
                },
            )

    assert resp.status_code == 200

    sent = json.loads(route.calls.last.request.content)
    assert SAMPLE not in json.dumps(sent)
    assert placeholder_re.search(sent["messages"][0]["content"]) is not None

    body = resp.json()
    text_out = body["content"][0]["text"]
    assert SAMPLE in text_out
    assert "⟦" not in text_out


def test_openai_format_provider_under_custom_name():
    name_start = PROMPT.index(SAMPLE)
    detector = FakeDetector({
        PROMPT: [FakeSpan(name_start, name_start + len(SAMPLE), "private_person")],
    })
    pmap = PlaceholderMap(new_session_key())
    cfg = Config(
        providers={
            "gpt": ProviderConfig(format="openai", upstream=UPSTREAM),
        }
    )
    app = create_app(config=cfg, detector=detector, placeholder_map=pmap)

    placeholder_re = re.compile(r"⟦private_person_[0-9a-f]{8}⟧")

    with respx.mock(assert_all_called=True) as mock:
        route = mock.post(f"{UPSTREAM}/v1/chat/completions").mock(
            return_value=httpx.Response(200, json={"ok": True})
        )

        with TestClient(app) as client:
            ph = pmap.mask(SAMPLE, "private_person")
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
                                    "content": f"hi {ph}",
                                },
                                "finish_reason": "stop",
                            }
                        ],
                    },
                )
            )

            resp = client.post(
                "/gpt/v1/chat/completions",
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
    assert SAMPLE not in json.dumps(sent)
    assert placeholder_re.search(sent["messages"][0]["content"]) is not None

    body = resp.json()
    content_out = body["choices"][0]["message"]["content"]
    assert SAMPLE in content_out
    assert "⟦" not in content_out
