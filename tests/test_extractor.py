from __future__ import annotations

import json
from pathlib import Path


from pii_proxi.masking.extractor import (
    JSON_STRING_SEP,
    extract_anthropic,
    extract_openai,
)


FIXTURES = json.loads(
    (Path(__file__).parent / "fixtures" / "sample_prompts.json").read_text()
)


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------


def test_anthropic_extract_string_system_and_user_content():
    body = {
        "system": "You are helpful.",
        "messages": [{"role": "user", "content": "Hi there."}],
    }
    got = extract_anthropic(body)
    assert ("/system", "You are helpful.") in got
    assert ("/messages/0/content", "Hi there.") in got


def test_anthropic_extract_block_system():
    body = {
        "system": [{"type": "text", "text": "block-form system"}],
        "messages": [],
    }
    got = extract_anthropic(body)
    assert got == [("/system/0/text", "block-form system")]


def test_anthropic_extract_nested_tool_result_blocks():
    body = FIXTURES["anthropic_tool_result_env_dump"]
    got = extract_anthropic(body)
    # The file dump must surface as one of the extracted leaves.
    env_leaf = next(
        (p, t) for (p, t) in got if "STRIPE_SECRET=sk_live" in t
    )
    ptr, text = env_leaf
    # Pointer shape: /messages/2/content/0/content/0/text
    assert ptr == "/messages/2/content/0/content/0/text"
    assert "alice@example.com" in text


def test_anthropic_tool_result_string_form():
    body = FIXTURES["anthropic_tool_result_string_content"]
    got = extract_anthropic(body)
    assert ("/messages/0/content/0/content",
            "quick string form with email bob@example.com") in got


def test_anthropic_tool_use_input_json_leaves():
    body = {
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "t1",
                        "name": "Bash",
                        "input": {
                            "command": "curl -H 'x: sk-live-X'",
                            "opts": {"url": "https://api.example.com"},
                            "count": 3,
                        },
                    }
                ],
            }
        ]
    }
    got = dict(extract_anthropic(body))
    assert got["/messages/0/content/0/input/command"] == "curl -H 'x: sk-live-X'"
    assert got["/messages/0/content/0/input/opts/url"] == "https://api.example.com"
    # Non-string leaves (the int 3) must not appear.
    assert all("count" not in p for p in got)


def test_anthropic_skips_empty_strings():
    body = {"system": "", "messages": [{"role": "user", "content": ""}]}
    assert extract_anthropic(body) == []


def test_anthropic_ignores_unknown_block_types():
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "source": {"data": "..."}},
                    {"type": "text", "text": "real text"},
                ],
            }
        ]
    }
    got = extract_anthropic(body)
    assert got == [("/messages/0/content/1/text", "real text")]


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------


def test_openai_mixed_content_forms():
    body = FIXTURES["openai_mixed_content_forms"]
    got = dict(extract_openai(body))
    assert got["/messages/0/content"] == "You write concise Python."
    assert got["/messages/1/content"].startswith("Ping foo@bar.com")
    assert got["/messages/2/content/0/text"] == "And also deal with baz@qux.io."
    assert got["/messages/2/content/1/text"].startswith("Key is sk-live")


def test_openai_tool_call_arguments_descent():
    body = FIXTURES["openai_tool_call_arguments_secret"]
    got = extract_openai(body)
    # Pointer for leaves inside the JSON-encoded arguments uses the '#'
    # sentinel.
    args_leaves = [
        (p, t)
        for (p, t) in got
        if JSON_STRING_SEP in p and p.startswith("/messages/1/tool_calls/0/function/arguments")
    ]
    leaves_by_inner = {p.split(JSON_STRING_SEP, 1)[1]: t for (p, t) in args_leaves}
    assert leaves_by_inner["/token"] == "sk-live-EMBEDDED99"
    assert leaves_by_inner["/notify"] == "ops@example.com"
    assert leaves_by_inner["/endpoint"] == "https://upload.example.com"


def test_openai_ignores_non_text_parts():
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "https://x/y"}},
                    {"type": "text", "text": "captioned"},
                ],
            }
        ]
    }
    got = extract_openai(body)
    assert got == [("/messages/0/content/1/text", "captioned")]


def test_openai_malformed_arguments_skipped():
    body = {
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "c",
                        "type": "function",
                        "function": {
                            "name": "f",
                            "arguments": "not-valid-json{{",
                        },
                    }
                ],
            }
        ]
    }
    # No crash, no leaves.
    assert extract_openai(body) == []


def test_openai_content_none_is_skipped():
    body = {"messages": [{"role": "assistant", "content": None}]}
    assert extract_openai(body) == []


def test_pointer_escaping_of_special_keys():
    # A key containing '/' and '~' must be escaped per RFC 6901.
    body = {
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "c",
                        "type": "function",
                        "function": {
                            "name": "f",
                            "arguments": json.dumps({"weird/key~name": "leaf-val"}),
                        },
                    }
                ],
            }
        ]
    }
    got = extract_openai(body)
    assert len(got) == 1
    ptr, value = got[0]
    assert value == "leaf-val"
    # The inner pointer component must escape / as ~1 and ~ as ~0.
    assert "/weird~1key~0name" in ptr
