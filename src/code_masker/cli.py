"""code-masker CLI entrypoints.

Commands:
    serve         Start the proxy.
    test TEXT     One-shot detection on a string, print spans and masked form.
    status        Probe the running proxy's /healthz endpoint.
    clear-session Ask the running proxy to drop its placeholder map.

The CLI is deliberately thin — the heavy lifting (model loading, HTTP
plumbing) lives in :mod:`code_masker.server`.
"""

from __future__ import annotations

import json
import sys
from typing import Optional

import httpx
import typer
import uvicorn

from .config import Config
from .masking.placeholder import PlaceholderMap, apply_spans
from .server import backend_factory, create_app
from .session import new_session_key


app = typer.Typer(
    name="code-masker",
    add_completion=False,
    no_args_is_help=True,
    help="Local privacy-filter proxy for coding assistants.",
)


def _proxy_base_url(cfg: Config) -> str:
    return f"http://{cfg.host}:{cfg.port}"


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

    typer.echo(f"  code-masker listening on {cfg.host}:{cfg.port}")
    typer.echo(f"    export ANTHROPIC_BASE_URL=http://{cfg.host}:{cfg.port}/anthropic")
    typer.echo(f"    export OPENAI_BASE_URL=http://{cfg.host}:{cfg.port}/openai/v1")
    typer.echo("")
    typer.echo("  Backend: " + cfg.backend)
    typer.echo("")

    uvicorn.run(application, host=cfg.host, port=cfg.port, log_level="info")


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


if __name__ == "__main__":  # pragma: no cover
    app()
