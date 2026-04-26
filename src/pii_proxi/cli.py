"""pii-proxi CLI entrypoints.

Commands:
    setup         One-shot bootstrap: pick backend, write config, fetch model.
    serve         Start the proxy.
    test TEXT     One-shot detection on a string, print spans and masked form.
    status        Probe the running proxy's /healthz endpoint.
    clear-session Ask the running proxy to drop its placeholder map.

The CLI is deliberately thin — the heavy lifting (model loading, HTTP
plumbing) lives in :mod:`pii_proxi.server`.
"""

from __future__ import annotations

import copy
import json
import platform
from pathlib import Path
from typing import Any, Optional

import httpx
import typer
import uvicorn
from uvicorn.config import LOGGING_CONFIG as _UVICORN_LOGGING_CONFIG

from .config import Config
from .masking.placeholder import PlaceholderMap, apply_spans
from .server import backend_factory, create_app
from .session import new_session_key


# Repo paths that the `setup` command writes into. Resolved against ``~`` so
# the values are absolute by the time they hit disk or the TOML file.
_CACHE_ROOT = Path("~/.cache/pii-proxi/models").expanduser()
_CONFIG_DIR = Path("~/.config/pii-proxi").expanduser()
_CONFIG_FILE = _CONFIG_DIR / "config.toml"
_MLX_DIR = _CACHE_ROOT / "mlx-8bit"
_ONNX_DIR = _CACHE_ROOT / "onnx-fp16"
_CALIB_FILE = _CACHE_ROOT / "viterbi_calibration.json"

# HF repos. The calibration JSON ships in the ONNX repo for both backends.
_MLX_REPO = "mlx-community/openai-privacy-filter-8bit"
_ONNX_REPO = "yasserrmd/privacy-filter-ONNX"
_CALIB_REPO = "yasserrmd/privacy-filter-ONNX"
_CALIB_FILENAME = "viterbi_calibration.json"


def _download_model(backend: str) -> None:
    """Fetch model weights and the Viterbi calibration JSON for ``backend``.

    ``snapshot_download`` is idempotent — already-present files are skipped.
    Kept as a helper so the setup command stays linear and readable.
    """
    # Imported lazily so `pii-proxi --help` doesn't pay the import cost.
    from huggingface_hub import hf_hub_download, snapshot_download

    if backend == "mlx":
        typer.echo(f"  fetching MLX weights → {_MLX_DIR}")
        snapshot_download(repo_id=_MLX_REPO, local_dir=str(_MLX_DIR))
    else:
        typer.echo(f"  fetching ONNX weights → {_ONNX_DIR}")
        snapshot_download(repo_id=_ONNX_REPO, local_dir=str(_ONNX_DIR))

    typer.echo(f"  fetching calibration → {_CALIB_FILE}")
    hf_hub_download(
        repo_id=_CALIB_REPO,
        filename=_CALIB_FILENAME,
        local_dir=str(_CACHE_ROOT),
    )


def _build_log_config() -> dict[str, Any]:
    """Extend uvicorn's default logging config to cover app loggers.

    Uvicorn only wires its own ``uvicorn.*`` loggers. The proxy's masking
    summary emits on ``pii_proxi.mask`` (and related ``pii_proxi.*``
    loggers), so without this extension those records have no handler and
    are dropped silently — which looked to users like masking wasn't
    happening at all.
    """
    cfg = copy.deepcopy(_UVICORN_LOGGING_CONFIG)
    # Include the logger name so ``pii_proxi.mask`` lines are self-
    # identifying in the shared stderr stream — matches the format in
    # README "Observing masking activity".
    cfg["formatters"]["pii_proxi"] = {
        "()": "uvicorn.logging.DefaultFormatter",
        "fmt": "%(levelprefix)s %(name)s: %(message)s",
        "use_colors": None,
    }
    cfg["handlers"]["pii_proxi"] = {
        "formatter": "pii_proxi",
        "class": "logging.StreamHandler",
        "stream": "ext://sys.stderr",
    }
    cfg["loggers"]["pii_proxi"] = {
        "handlers": ["pii_proxi"],
        "level": "INFO",
        "propagate": False,
    }
    return cfg


app = typer.Typer(
    name="pii-proxi",
    add_completion=False,
    no_args_is_help=True,
    help="Local privacy-filter proxy for coding assistants.",
)


def _proxy_base_url(cfg: Config) -> str:
    return f"http://{cfg.host}:{cfg.port}"


def _print_provider_hints(cfg: Config, prefix: str = "    ") -> None:
    base = f"http://{cfg.host}:{cfg.port}"
    for name, p in cfg.providers.items():
        if p.format == "anthropic":
            typer.echo(f"{prefix}export ANTHROPIC_BASE_URL={base}/{name}")
        else:
            typer.echo(f"{prefix}export OPENAI_BASE_URL={base}/{name}/v1")


def _write_default_config(backend: str) -> None:
    """Write a starter ``config.toml`` for the chosen backend.

    Prefers copying the in-repo ``config.example.toml`` (works when the
    package is installed editable from a clone) so users get the full set of
    documented optional fields. Falls back to a minimal hardcoded stub for
    wheel installs where the example file is not on disk.
    """
    # parents[2] from src/pii_proxi/cli.py == repo root in editable installs.
    example = Path(__file__).resolve().parents[2] / "config.example.toml"
    model_path = str(_MLX_DIR if backend == "mlx" else _ONNX_DIR)
    calib_path = str(_CALIB_FILE)

    if example.is_file():
        text = example.read_text()
        # Rewrite the first three lines to match the chosen backend and the
        # absolute (expanded) paths so users don't need to think about ``~``.
        new_lines = [
            f'backend = "{backend}"',
            f'model_path = "{model_path}"',
            f'calibration_path = "{calib_path}"',
        ]
        rest = text.splitlines()[3:]
        _CONFIG_FILE.write_text("\n".join(new_lines + rest) + "\n")
        return

    # Fallback: minimal but valid TOML covering the required fields. Not as
    # nice as the commented example, but sufficient to boot the proxy.
    _CONFIG_FILE.write_text(
        f'backend = "{backend}"\n'
        f'model_path = "{model_path}"\n'
        f'calibration_path = "{calib_path}"\n'
    )


@app.command()
def setup() -> None:
    """Bootstrap a fresh install: pick backend, write config, fetch model."""
    # 1. Detect the best backend for this machine. Apple Silicon gets MLX
    #    (native Metal accel); everything else falls back to ONNX runtime.
    is_apple_silicon = (
        platform.system() == "Darwin" and platform.machine() == "arm64"
    )
    backend = "mlx" if is_apple_silicon else "onnx"
    typer.echo(f"detected platform → backend: {backend}")

    # 2. Config dir + file. Idempotent: leave any existing config alone so
    #    re-running setup never clobbers user edits.
    _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    if _CONFIG_FILE.exists():
        typer.echo(f"config already exists at {_CONFIG_FILE}, leaving as-is")
    else:
        _write_default_config(backend)
        typer.echo(f"wrote config → {_CONFIG_FILE}")

    # 3. Cache dir for model weights.
    _CACHE_ROOT.mkdir(parents=True, exist_ok=True)

    # 4. Pull model weights + calibration. snapshot_download skips files
    #    that are already on disk, so re-running is cheap.
    typer.echo("downloading model artifacts (this can take a while on first run)…")
    _download_model(backend)

    # 5. Load the actual config (which may pre-date this run) and warm up
    #    the detector to confirm everything wired up cleanly. We swallow the
    #    stack trace here — a wall of traceback after a fresh install is
    #    intimidating, and the actionable next step is just "re-run setup".
    cfg = Config.load()
    typer.echo("warming up detector…")
    try:
        detector = backend_factory(cfg.backend, cfg)
        detector.warmup()
    except Exception as e:  # pragma: no cover - exercised manually
        typer.echo(f"warmup failed: {e}", err=True)
        typer.echo(
            "the model may be incomplete or incompatible — "
            "try re-running `pii-proxi setup`",
            err=True,
        )
        raise typer.Exit(code=1)

    typer.echo("")
    typer.echo("setup complete. Next steps:")
    _print_provider_hints(cfg)
    typer.echo("    pii-proxi serve")


@app.command()
def serve(
    config_path: Optional[str] = typer.Option(
        None, "--config", "-c", help="Override config TOML path."
    ),
    port: Optional[int] = typer.Option(None, "--port", "-p"),
    host: Optional[str] = typer.Option(None, "--host"),
) -> None:
    """Start the proxy (loads detector, binds 127.0.0.1 by default)."""
    cfg = Config.load(config_path)
    if port is not None:
        cfg.port = port
    if host is not None:
        cfg.host = host

    application = create_app(config=cfg)

    typer.echo(f"  pii-proxi listening on {cfg.host}:{cfg.port}")
    _print_provider_hints(cfg)
    typer.echo("")
    typer.echo("  Backend: " + cfg.backend)
    typer.echo("")

    uvicorn.run(
        application,
        host=cfg.host,
        port=cfg.port,
        log_config=_build_log_config(),
    )


@app.command("test")
def test_cmd(
    text: str = typer.Argument(..., help="Text to run through the detector."),
    config_path: Optional[str] = typer.Option(None, "--config", "-c"),
) -> None:
    """One-shot: run the detector and print spans + masked form."""
    cfg = Config.load(config_path)
    detector = backend_factory(cfg.backend, cfg)
    detector.warmup()

    disabled = frozenset(cfg.disabled_labels)
    spans_list = detector.detect([text])
    spans = spans_list[0] if spans_list else []
    if disabled:
        spans = [s for s in spans if s.label not in disabled]

    typer.echo(f"Input: {text!r}")
    typer.echo(f"Detected {len(spans)} span(s):")
    for s in spans:
        typer.echo(f"  [{s.start:>4}:{s.end:<4}] {s.label}  {text[s.start:s.end]!r}")

    pmap = PlaceholderMap(new_session_key())
    # apply_spans expects its own Span type, but it duck-types on the three
    # fields, so detector spans work here without a conversion step.
    masked = apply_spans(text, list(spans), pmap)  # type: ignore[arg-type]
    typer.echo("")
    typer.echo(f"Masked: {masked!r}")


@app.command()
def status(
    port: int = typer.Option(8787, "--port", "-p"),
    host: str = typer.Option("127.0.0.1", "--host"),
) -> None:
    """Probe the running proxy's /healthz endpoint."""
    url = f"http://{host}:{port}/healthz"
    try:
        resp = httpx.get(url, timeout=3.0)
    except httpx.HTTPError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(code=1)
    typer.echo(f"{resp.status_code} {json.dumps(resp.json(), indent=2)}")
    if resp.status_code != 200:
        raise typer.Exit(code=1)


@app.command("clear-session")
def clear_session(
    port: int = typer.Option(8787, "--port", "-p"),
    host: str = typer.Option("127.0.0.1", "--host"),
) -> None:
    """Tell the running proxy to drop its placeholder map."""
    url = f"http://{host}:{port}/admin/clear-session"
    try:
        resp = httpx.post(url, timeout=3.0)
    except httpx.HTTPError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(code=1)
    typer.echo(f"{resp.status_code} {resp.text}")
    if resp.status_code != 200:
        raise typer.Exit(code=1)


@app.command()
def providers(
    config_path: Optional[str] = typer.Option(
        None, "--config", "-c", help="Override config TOML path."
    ),
) -> None:
    """List registered providers and their local URLs."""
    cfg = Config.load(config_path)
    base = f"http://{cfg.host}:{cfg.port}"
    rows = [("NAME", "FORMAT", "UPSTREAM", "LOCAL URL")]
    for name, p in cfg.providers.items():
        local = f"{base}/{name}/v1" if p.format == "openai" else f"{base}/{name}"
        rows.append((name, p.format, p.upstream, local))
    widths = [max(len(r[i]) for r in rows) for i in range(4)]
    for r in rows:
        typer.echo("  ".join(c.ljust(w) for c, w in zip(r, widths)))


if __name__ == "__main__":  # pragma: no cover
    app()
