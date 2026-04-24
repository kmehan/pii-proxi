"""Runtime configuration for the code-masker proxy.

Loads defaults, overlays optional TOML at ``~/.config/code-masker/config.toml``,
then lets environment variables (``CODE_MASKER_*``) take the final word. Paths
are expanded with :func:`os.path.expanduser` at load time so downstream code
never has to think about ``~``.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


DEFAULT_CONFIG_PATH = Path("~/.config/code-masker/config.toml").expanduser()


def _load_toml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("rb") as f:
        return tomllib.load(f)


class Config(BaseSettings):
    """Proxy configuration.

    Field defaults mirror the documented config schema. The ``CODE_MASKER_``
    env-var prefix lets users override any field without editing the TOML —
    useful for one-off runs, tests, and CI.
    """

    model_config = SettingsConfigDict(
        env_prefix="CODE_MASKER_",
        extra="ignore",
    )

    port: int = 8787
    host: str = "127.0.0.1"
    backend: str = "mlx"  # "mlx" | "onnx"
    model_path: str = "~/.cache/code-masker/models/mlx-8bit"
    calibration_path: str = "~/.cache/code-masker/models/viterbi_calibration.json"
    disabled_labels: list[str] = Field(default_factory=list)
    log_path: str = "~/.local/state/code-masker/audit.log"
    anthropic_upstream: str = "https://api.anthropic.com"
    openai_upstream: str = "https://api.openai.com"

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

    @classmethod
    def load(cls, path: Path | str | None = None) -> "Config":
        """Load config from TOML (if present), then overlay env vars."""
        toml_path = Path(path).expanduser() if path else DEFAULT_CONFIG_PATH
        toml_data = _load_toml(toml_path)
        # pydantic-settings' env-var source runs automatically when we
        # construct via the model; explicit fields from TOML take precedence
        # over defaults but env overrides both (BaseSettings init order).
        return cls(**toml_data)
