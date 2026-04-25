"""Extract maskable text leaves from Anthropic and OpenAI request bodies.

Each extractor returns ``[(json_pointer, text), ...]`` using RFC 6901 pointer
syntax. The paired :mod:`injector` module writes masked strings back at those
same pointers, so pointer shape is part of the contract between the two.

Design notes
------------
* We only collect strings the detector should scan. Role strings, model
  identifiers, tool names, and IDs are intentionally skipped — masking those
  would break the upstream API.
* Anthropic ``tool_result`` blocks are walked recursively. Their ``content``
  may be a bare ``str`` or a list of sub-blocks (each a ``text`` block), and
  Claude Code's Bash/Read results embed file dumps there. This is the single
  highest-risk leak path, so it gets its own test.
* OpenAI ``tool_calls[].function.arguments`` is a JSON-encoded **string**, not
  a dict. We parse it, collect string leaves, and the injector reserializes.
  Pointers into the parsed arguments use the prefix ``.../arguments`` plus a
  ``#`` marker to signal "dive into the JSON string here"; the injector knows
  to re-parse, splice, and re-dump.
"""

from __future__ import annotations

import json
from typing import Any


# Sentinel that separates "outer" JSON pointer from "inner" pointer inside a
# JSON-encoded string. Chosen so it can't collide with a normal token — RFC
# 6901 pointers contain only ``/`` segments with ``~0``/``~1`` escapes.
JSON_STRING_SEP = "#"


def _escape_token(token: str) -> str:
    # RFC 6901: escape ~ then /
    return token.replace("~", "~0").replace("/", "~1")


def _join(prefix: str, *tokens: Any) -> str:
    out = prefix
    for t in tokens:
        out += "/" + _escape_token(str(t))
    return out


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------


def extract_anthropic(body: dict) -> list[tuple[str, str]]:
    """Walk an Anthropic ``POST /v1/messages`` request body.

    Covers:
      * ``system`` — may be a bare string or a list of blocks (each with
        ``text``).
      * ``messages[].content`` — bare string or list of blocks. Block types
        handled: ``text`` (``.text``), ``tool_result`` (recurse into
        ``.content`` which is str or list of sub-blocks), ``tool_use``
        (recurse into ``.input`` JSON-like structure, masking string leaves).
    """
    out: list[tuple[str, str]] = []

    sys = body.get("system")
    if isinstance(sys, str):
        if sys:
            out.append(("/system", sys))
    elif isinstance(sys, list):
        for i, block in enumerate(sys):
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                if block["text"]:
                    out.append((_join("/system", i, "text"), block["text"]))

    messages = body.get("messages")
    if isinstance(messages, list):
        for mi, msg in enumerate(messages):
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            msg_ptr = _join("/messages", mi)
            _walk_anthropic_content(content, _join(msg_ptr, "content"), out)

    return out


def _walk_anthropic_content(
    content: Any, ptr: str, out: list[tuple[str, str]]
) -> None:
    if isinstance(content, str):
        if content:
            out.append((ptr, content))
        return
    if not isinstance(content, list):
        return
    for bi, block in enumerate(content):
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        bptr = _join(ptr, bi)
        if btype == "text":
            text = block.get("text")
            if isinstance(text, str) and text:
                out.append((_join(bptr, "text"), text))
        elif btype == "tool_result":
            inner = block.get("content")
            _walk_anthropic_content(inner, _join(bptr, "content"), out)
        elif btype == "tool_use":
            inp = block.get("input")
            if inp is not None:
                _walk_json_leaves(inp, _join(bptr, "input"), out)
        # Other block types (image, document, thinking) carry no plaintext
        # we want to mask client-side, so we skip them.


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------


def extract_openai(body: dict) -> list[tuple[str, str]]:
    """Walk an OpenAI ``POST /v1/chat/completions`` request body.

    Covers:
      * ``messages[].content`` — bare string, or list of parts where each part
        has ``type == "text"`` and a ``text`` field.
      * ``messages[].tool_calls[].function.arguments`` — a JSON-encoded string.
        We parse it, collect every string leaf, and the injector reserializes.
    """
    out: list[tuple[str, str]] = []

    messages = body.get("messages")
    if not isinstance(messages, list):
        return out

    for mi, msg in enumerate(messages):
        if not isinstance(msg, dict):
            continue
        msg_ptr = _join("/messages", mi)

        content = msg.get("content")
        if isinstance(content, str):
            if content:
                out.append((_join(msg_ptr, "content"), content))
        elif isinstance(content, list):
            for pi, part in enumerate(content):
                if (
                    isinstance(part, dict)
                    and part.get("type") == "text"
                    and isinstance(part.get("text"), str)
                    and part["text"]
                ):
                    out.append(
                        (_join(msg_ptr, "content", pi, "text"), part["text"])
                    )

        tool_calls = msg.get("tool_calls")
        if isinstance(tool_calls, list):
            for ti, call in enumerate(tool_calls):
                if not isinstance(call, dict):
                    continue
                fn = call.get("function")
                if not isinstance(fn, dict):
                    continue
                args_str = fn.get("arguments")
                if not isinstance(args_str, str) or not args_str:
                    continue
                try:
                    parsed = json.loads(args_str)
                except json.JSONDecodeError:
                    # Upstream will reject a malformed arguments string on its
                    # own; masking a non-JSON blob wholesale would risk
                    # corrupting tool invocations, so we leave it alone.
                    continue
                base = _join(msg_ptr, "tool_calls", ti, "function", "arguments")
                # Sentinel prefix "#" tells the injector: from here on, the
                # pointer indexes the parsed JSON inside the arguments string.
                _walk_json_leaves(parsed, base + JSON_STRING_SEP, out)

    return out


# ---------------------------------------------------------------------------
# Shared: walk arbitrary JSON structures and emit pointer/string pairs.
# ---------------------------------------------------------------------------


def _walk_json_leaves(node: Any, ptr: str, out: list[tuple[str, str]]) -> None:
    if isinstance(node, str):
        if node:
            out.append((ptr, node))
        return
    if isinstance(node, dict):
        for k, v in node.items():
            _walk_json_leaves(v, _join(ptr, k), out)
        return
    if isinstance(node, list):
        for i, v in enumerate(node):
            _walk_json_leaves(v, _join(ptr, i), out)
        return
    # numbers, bools, null: nothing to mask
