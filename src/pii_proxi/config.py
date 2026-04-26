"""Runtime configuration for the pii-proxi proxy.

Loads defaults, overlays optional TOML at ``~/.config/pii-proxi/config.toml``,
then lets environment variables (``PII_PROXI_*``) take the final word. Paths
are expanded with :func:`os.path.expanduser` at load time so downstream code
never has to think about ``~``.
"""

from __future__ import annotations

import os
import re
import tomllib
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


DEFAULT_CONFIG_PATH = Path("~/.config/pii-proxi/config.toml").expanduser()

_PROVIDER_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


class ProviderConfig(BaseModel):
    """Upstream provider entry for multi-provider proxy mode."""

    format: Literal["openai", "anthropic"]
    upstream: str

    @field_validator("upstream", mode="before")
    @classmethod
    def _strip_trailing_slash(cls, v: Any) -> Any:
        if isinstance(v, str):
            return v.rstrip("/")
        return v


def _load_toml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("rb") as f:
        return tomllib.load(f)


class Config(BaseSettings):
    """Proxy configuration.

    Field defaults mirror the documented config schema. The ``PII_PROXI_``
    env-var prefix lets users override any field without editing the TOML —
    useful for one-off runs, tests, and CI.
    """

    model_config = SettingsConfigDict(
        env_prefix="PII_PROXI_",
        extra="ignore",
    )

    port: int = 8787
    host: str = "127.0.0.1"
    backend: str = "mlx"  # "mlx" | "onnx"
    model_path: str = "~/.cache/pii-proxi/models/mlx-8bit"
    calibration_path: str = "~/.cache/pii-proxi/models/viterbi_calibration.json"
    disabled_labels: list[str] = Field(default_factory=list)
    log_path: str = "~/.local/state/pii-proxi/audit.log"
    anthropic_upstream: str = "https://api.anthropic.com"
    openai_upstream: str = "https://api.openai.com"
    # When true, each proxied request emits an INFO line containing the
    # detected plaintext alongside the label. Off by default — a privacy
    # tool shouldn't log secrets unless you ask it to. Safe for local
    # single-user use; do not enable on shared machines or in any context
    # where the proxy's stdout/log may be captured, uploaded, or shipped.
    log_entities: bool = False
    providers: dict[str, "ProviderConfig"] = Field(default_factory=dict)

    # matches FastAPI's built-in route paths
    _RESERVED_PROVIDER_NAMES: ClassVar[frozenset[str]] = frozenset(
        {"healthz", "admin", "docs", "redoc", "openapi.json"}
    )

    @field_validator("model_path", "calibration_path", "log_path", mode="before")
    @classmethod
    def _expand(cls, v: Any) -> Any:
        if isinstance(v, str):
            return os.path.expanduser(v)
        return v

    @field_validator("backend")
    @classmethod
    def _check_backend(cls, v: str) -> str:
        if v not in ("mlx", "onnx"):
            raise ValueError(f"backend must be 'mlx' or 'onnx', got {v!r}")
        return v

    @model_validator(mode="after")
    def _finalize_providers(self) -> "Config":
        for name in self.providers:
            if name in self._RESERVED_PROVIDER_NAMES:
                raise ValueError(
                    f"provider name {name!r} is reserved and cannot be used"
                )
            if not _PROVIDER_NAME_RE.match(name):
                raise ValueError(
                    f"provider name {name!r} is invalid: must match "
                    r"^[a-z0-9][a-z0-9_-]*$"
                )
        if "anthropic" not in self.providers:
            self.providers["anthropic"] = ProviderConfig(
                format="anthropic", upstream=self.anthropic_upstream
            )
        if "openai" not in self.providers:
            self.providers["openai"] = ProviderConfig(
                format="openai", upstream=self.openai_upstream
            )
        return self

    @classmethod
    def load(cls, path: Path | str | None = None) -> "Config":
        """Load config from TOML (if present), then overlay env vars."""
        toml_path = Path(path).expanduser() if path else DEFAULT_CONFIG_PATH
        toml_data = _load_toml(toml_path)
        # pydantic-settings' env-var source runs automatically when we
        # construct via the model; explicit fields from TOML take precedence
        # over defaults but env overrides both (BaseSettings init order).
        return cls(**toml_data)
