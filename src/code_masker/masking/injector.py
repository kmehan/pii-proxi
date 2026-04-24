"""Splice masked strings back into request bodies at their JSON pointers.

Pairs with :mod:`extractor`. Pointers are RFC 6901 except that we use a ``#``
sentinel to signal "the remaining suffix indexes into a JSON-encoded string
value" — used for OpenAI ``tool_calls[].function.arguments``. The injector
detects that sentinel, parses the string, sets the leaf via the inner
pointer, and re-serializes the arguments string before writing it back.

Contract: :func:`inject` **deep-copies** the body before mutating. Callers
that care about throughput can still mutate in place via :func:`set_by_pointer`
directly, but ``inject`` is the safe default — the original body may still be
needed for logging shape/size without worrying about concurrent mutation.
"""

from __future__ import annotations

import copy
import json
from typing import Any

from .extractor import JSON_STRING_SEP


def _unescape_token(token: str) -> str:
    # RFC 6901: unescape ~1 then ~0 (order matters)
    return token.replace("~1", "/").replace("~0", "~")


def _split_pointer(pointer: str) -> list[str]:
    if pointer == "":
        return []
    if not pointer.startswith("/"):
        raise ValueError(f"invalid JSON pointer: {pointer!r}")
    return [_unescape_token(tok) for tok in pointer[1:].split("/")]


def _descend(node: Any, token: str) -> Any:
    if isinstance(node, list):
        idx = int(token)
        return node[idx]
    if isinstance(node, dict):
        return node[token]
    raise TypeError(f"cannot descend into {type(node).__name__} via {token!r}")


def _assign(node: Any, token: str, value: Any) -> None:
    if isinstance(node, list):
        node[int(token)] = value
    elif isinstance(node, dict):
        node[token] = value
    else:
        raise TypeError(f"cannot assign to {type(node).__name__} via {token!r}")


def set_by_pointer(body: dict, pointer: str, value: str) -> None:
    """Set ``value`` at ``pointer`` in ``body``. Mutates in place.

    Handles the ``#`` sentinel: if the pointer contains ``#``, the outer part
    addresses a JSON-encoded string in ``body``; we parse that string, set the
    leaf via the inner pointer, then re-dump it back at the outer location.
    """
    if JSON_STRING_SEP in pointer:
        outer, inner = pointer.split(JSON_STRING_SEP, 1)
        outer_tokens = _split_pointer(outer)
        if not outer_tokens:
            raise ValueError("JSON-string pointer must have an outer path")
        container, last = _walk_to_parent(body, outer_tokens)
        raw = _descend(container, last)
        if not isinstance(raw, str):
            raise TypeError(
                f"expected JSON-encoded string at {outer!r}, got {type(raw).__name__}"
            )
        parsed = json.loads(raw)
        inner_tokens = _split_pointer(inner)
        if not inner_tokens:
            # Whole arguments string is itself the leaf being replaced.
            _assign(container, last, value)
            return
        leaf_parent, leaf_last = _walk_nested(parsed, inner_tokens)
        _assign(leaf_parent, leaf_last, value)
        # ``ensure_ascii=False`` preserves non-ASCII chars (placeholders!) as
        # UTF-8 rather than \uXXXX escapes, which keeps the upstream body
        # byte-compatible with what the model produced in training.
        _assign(container, last, json.dumps(parsed, ensure_ascii=False))
        return

    tokens = _split_pointer(pointer)
    if not tokens:
        raise ValueError("cannot set root via pointer")
    parent, last = _walk_to_parent(body, tokens)
    _assign(parent, last, value)


def _walk_to_parent(root: Any, tokens: list[str]) -> tuple[Any, str]:
    *head, last = tokens
    node = root
    for t in head:
        node = _descend(node, t)
    return node, last


def _walk_nested(root: Any, tokens: list[str]) -> tuple[Any, str]:
    return _walk_to_parent(root, tokens)


def inject(body: dict, masked: list[tuple[str, str]]) -> dict:
    """Return a deep copy of ``body`` with each ``(pointer, masked_text)`` applied.

    Assignments are applied in the given order. For normal pointers the order
    doesn't matter; for JSON-string pointers that share an outer path we rely
    on the fact that each assignment re-parses, mutates, and re-dumps — so
    multiple leaves inside the same ``arguments`` string compose correctly
    across successive calls, even though each call round-trips through JSON.
    """
    out = copy.deepcopy(body)
    for pointer, value in masked:
        set_by_pointer(out, pointer, value)
    return out
