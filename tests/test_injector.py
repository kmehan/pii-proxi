from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from pii_proxi.masking.extractor import extract_anthropic, extract_openai
from pii_proxi.masking.injector import inject, set_by_pointer
from pii_proxi.masking.placeholder import PlaceholderMap, Span, apply_spans


FIXTURES = json.loads(
    (Path(__file__).parent / "fixtures" / "sample_prompts.json").read_text()
)


def _pmap() -> PlaceholderMap:
    return PlaceholderMap(b"\xab" * 32)


def test_set_by_pointer_basic_list_and_dict():
    body = {"a": {"b": [10, 20, 30]}}
    set_by_pointer(body, "/a/b/1", "X")
    assert body == {"a": {"b": [10, "X", 30]}}


def test_set_by_pointer_escapes_roundtrip():
    body = {"weird/key~name": "old"}
    set_by_pointer(body, "/weird~1key~0name", "new")
    assert body["weird/key~name"] == "new"


def test_inject_preserves_original_body():
    body = FIXTURES["anthropic_tool_result_env_dump"]
    original = copy.deepcopy(body)
    # Mask one known leaf.
    pairs = [("/messages/0/content", "REDACTED")]
    out = inject(body, pairs)
    assert out["messages"][0]["content"] == "REDACTED"
    assert body == original  # deep-copied, caller's body untouched


def test_inject_into_openai_tool_arguments_string():
    body = FIXTURES["openai_tool_call_arguments_secret"]
    leaves = extract_openai(body)
    # Replace the embedded token and email with placeholders we pick.
    replacements: list[tuple[str, str]] = []
    for ptr, val in leaves:
        if val == "sk-live-EMBEDDED99":
            replacements.append((ptr, "[SECRET]"))
        elif val == "ops@example.com":
            replacements.append((ptr, "[EMAIL]"))
    out = inject(body, replacements)

    args_str = out["messages"][1]["tool_calls"][0]["function"]["arguments"]
    assert "sk-live-EMBEDDED99" not in args_str
    assert "ops@example.com" not in args_str
    parsed = json.loads(args_str)
    assert parsed["token"] == "[SECRET]"
    assert parsed["notify"] == "[EMAIL]"
    # Non-masked field should pass through verbatim.
    assert parsed["endpoint"] == "https://upload.example.com"


def test_full_roundtrip_anthropic_mask_and_unmask():
    body = FIXTURES["anthropic_tool_result_env_dump"]
    pmap = _pmap()

    leaves = extract_anthropic(body)
    masked_pairs: list[tuple[str, str]] = []
    originals: dict[str, str] = {}
    for ptr, text in leaves:
        if "sk_live_abcdef123456789" in text or "SG." in text:
            # Mask the first "line" as a SECRET.
            span = Span(0, len(text.split("\n")[0]), "SECRET")
            masked = apply_spans(text, [span], pmap)
            masked_pairs.append((ptr, masked))
            originals[ptr] = text
        if "alice@example.com" in text and "\n" not in text:
            # Mask entire question.
            span = Span(text.index("alice@example.com"),
                        text.index("alice@example.com") + len("alice@example.com"),
                        "EMAIL")
            masked_pairs.append((ptr, apply_spans(text, [span], pmap)))

    out = inject(body, masked_pairs)
    # After injection, the plaintext secret must be gone.
    serialized = json.dumps(out)
    assert "sk_live_abcdef123456789" not in serialized


def test_set_by_pointer_rejects_root():
    with pytest.raises(ValueError):
        set_by_pointer({}, "", "x")


def test_set_by_pointer_rejects_malformed_pointer():
    with pytest.raises(ValueError):
        set_by_pointer({}, "no-leading-slash", "x")


def test_inject_ordering_into_same_arguments_string():
    # Two leaves inside the same JSON-string should both land, not clobber.
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
                            "arguments": json.dumps({"a": "one", "b": "two"}),
                        },
                    }
                ],
            }
        ]
    }
    leaves = extract_openai(body)
    pairs = []
    for ptr, val in leaves:
        if val == "one":
            pairs.append((ptr, "ONE"))
        elif val == "two":
            pairs.append((ptr, "TWO"))
    out = inject(body, pairs)
    parsed = json.loads(out["messages"][0]["tool_calls"][0]["function"]["arguments"])
    assert parsed == {"a": "ONE", "b": "TWO"}
