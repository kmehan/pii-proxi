"""Session-scoped state for the masking pipeline.

A *session key* is 32 random bytes used to key the blake2b hash that mints
placeholder suffixes. Because the key lives only in memory and is never
persisted, identical plaintexts across different proxy processes produce
different placeholders; this is by design — the key is effectively a salt that
prevents an attacker who captures masked traffic from pre-computing a
plaintext->placeholder rainbow table.
"""

from __future__ import annotations

import secrets


SESSION_KEY_BYTES = 32


def new_session_key() -> bytes:
    """Return a fresh 32-byte session key suitable for keyed blake2b."""
    return secrets.token_bytes(SESSION_KEY_BYTES)
