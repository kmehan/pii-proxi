"""Schema-level tests for the multi-provider ``providers`` map.

These exercise pure pydantic behaviour: defaults, backfill from the legacy
``*_upstream`` fields, name validation, and TOML round-tripping. No FastAPI
or HTTP plumbing involved.
"""

from __future__ import annotations

import textwrap

import pytest
from pydantic import ValidationError

from pii_proxi.config import Config, ProviderConfig


def test_defaults_backfill_anthropic_and_openai():
    cfg = Config()
    assert set(cfg.providers.keys()) == {"anthropic", "openai"}
    assert cfg.providers["anthropic"].format == "anthropic"
    assert cfg.providers["anthropic"].upstream == "https://api.anthropic.com"
    assert cfg.providers["openai"].format == "openai"
    assert cfg.providers["openai"].upstream == "https://api.openai.com"


def test_explicit_extra_provider_does_not_remove_defaults():
    cfg = Config(
        providers={
            "deepseek": ProviderConfig(
                format="anthropic",
                upstream="https://api.deepseek.com/anthropic",
            ),
        }
    )
    assert set(cfg.providers.keys()) == {"anthropic", "deepseek", "openai"}
    assert (
        cfg.providers["deepseek"].upstream == "https://api.deepseek.com/anthropic"
    )


def test_user_redefined_anthropic_wins_over_legacy_field():
    cfg = Config(
        providers={
            "anthropic": ProviderConfig(
                format="anthropic", upstream="https://custom.example.com"
            ),
        }
    )
    assert cfg.providers["anthropic"].upstream == "https://custom.example.com"


def test_reserved_provider_name_rejected():
    with pytest.raises(ValidationError) as excinfo:
        Config(
            providers={
                "healthz": ProviderConfig(
                    format="openai", upstream="https://x.example.com"
                ),
            }
        )
    msg = str(excinfo.value)
    assert "healthz" in msg
    assert "reserved" in msg


def test_invalid_format_rejected():
    with pytest.raises(ValidationError):
        ProviderConfig(format="bogus", upstream="https://x.example.com")  # type: ignore[arg-type]


def test_trailing_slash_stripped_from_upstream():
    pc = ProviderConfig(format="openai", upstream="https://api.example.com/")
    assert pc.upstream == "https://api.example.com"


def test_bad_provider_name_regex_rejected():
    with pytest.raises(ValidationError) as excinfo:
        Config(
            providers={
                "Bad.Name": ProviderConfig(
                    format="openai", upstream="https://x.example.com"
                ),
            }
        )
    msg = str(excinfo.value)
    assert "Bad.Name" in msg


def test_load_from_toml_round_trips_provider_block(tmp_path):
    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text(
        textwrap.dedent(
            """
            backend = "onnx"

            [providers.deepseek]
            format = "anthropic"
            upstream = "https://api.deepseek.com/anthropic"
            """
        ).strip()
        + "\n"
    )

    cfg = Config.load(cfg_file)
    assert "deepseek" in cfg.providers
    assert cfg.providers["deepseek"].format == "anthropic"
    assert (
        cfg.providers["deepseek"].upstream
        == "https://api.deepseek.com/anthropic"
    )
    assert {"anthropic", "openai"}.issubset(cfg.providers.keys())
