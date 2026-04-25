# pii-proxi

Drop-in local proxy that strips PII and secrets from prompts before they reach any LLM API and seamlessly restores them in responses, so you can use any cloud model worry-free without changing your workflow. Detection runs entirely on-device using OpenAI's open-weight privacy-filter model, and the proxy is transparent at the auth layer — it works with API keys and OAuth (Claude Pro/Max, Sign in with ChatGPT) alike.

## Quick start

Requires Python 3.11+. Pick `[mlx]` on Apple Silicon (fastest), `[onnx]` on Linux / Intel / Windows.

Install — option A (single command, recommended):

```bash
pipx install "git+https://github.com/<owner>/pii-proxi.git#egg=pii-proxi[mlx]"
```

Install — option B (clone for development):

```bash
git clone https://github.com/<owner>/pii-proxi.git && cd pii-proxi
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[mlx]"   # or ".[onnx]"
```

Then:

```bash
pii-proxi setup     # detects backend, fetches model + calibration, writes default config
pii-proxi serve
```

In a separate shell, point your client at the proxy:

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:8787/anthropic
export OPENAI_BASE_URL=http://127.0.0.1:8787/openai/v1
```

Sanity-check the detector without wiring a client:

```bash
pii-proxi test "my key is sk-live-AAAABBBBCCCCDDDD and email foo@bar.com"
```

## Run it persistently

| Platform | Command |
|---|---|
| macOS (launchd) | `./scripts/install-launchd.sh` |
| Linux (systemd user unit) | `./scripts/install-systemd.sh` |
| Manual / dev | `pii-proxi serve` under tmux or screen |

Uninstall with `./scripts/uninstall.sh`.

## Point your client at the proxy

**Claude Code** — works with `ANTHROPIC_API_KEY` and Pro/Max OAuth; the OAuth bearer rides through `Authorization` the same way an API key does, so `ANTHROPIC_BASE_URL` is honored either way:

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:8787/anthropic
claude
```

**Codex CLI / aider / continue.dev** (any OpenAI-compatible client):

```bash
export OPENAI_BASE_URL=http://127.0.0.1:8787/openai/v1
codex   # or: aider, continue, ...
```

**Cursor (BYO key only):** Settings → Models → OpenAI Base URL → `http://127.0.0.1:8787/openai/v1`. Cursor Pro subscriptions are not supported (see table).

## Supported clients

| Client | Auth mode | Supported | Notes |
|---|---|---|---|
| Claude Code | `ANTHROPIC_API_KEY` | yes | Standard path. |
| Claude Code | Claude Pro/Max login (OAuth) | yes | `ANTHROPIC_BASE_URL` is still honored; the OAuth bearer token rides through `Authorization` the same way an API key does. No proxy-side changes needed. |
| Codex CLI | `OPENAI_API_KEY` | yes | Standard path. |
| Codex CLI | "Sign in with ChatGPT" (OAuth) | warn / verify | Likely works if Codex honors `OPENAI_BASE_URL` in OAuth mode and the token's audience isn't pinned to a ChatGPT-specific backend. Verify locally before relying on it for sensitive prompts. |
| Any OpenAI-compat tool (aider, continue.dev) w/ BYO key | API key | yes | Same `OPENAI_BASE_URL` route. |
| Cursor | BYO API key | yes | Uses Cursor's custom-base-URL setting. |
| Cursor | Cursor Pro subscription | no | Cursor Pro routes through `api.cursor.sh` with vendor-managed auth; no client-side base-URL override. Would require TLS MITM, which is out of scope. |

## Configuration

`pii-proxi setup` writes a working default to `~/.config/pii-proxi/config.toml`. You only need to edit it to override the defaults:

```toml
port = 8787
backend = "mlx"                    # "mlx" | "onnx"
model_path = "~/.cache/pii-proxi/models/mlx-8bit"
calibration_path = "~/.cache/pii-proxi/models/viterbi_calibration.json"
disabled_labels = []               # e.g. ["private_email"] to skip a class
log_path = "~/.local/state/pii-proxi/audit.log"
anthropic_upstream = "https://api.anthropic.com"
openai_upstream = "https://api.openai.com"
log_entities = false               # see Observability — off by default
```

Every field also reads from `PII_PROXI_<NAME>` env vars for one-off overrides.

## Observability

Counts are always logged to stdout — no plaintext, safe to leave on:

```
INFO:     pii_proxi.mask: masked 2 span(s) across 1 text(s): private_email=1, secret=1
```

Plaintext logging is opt-in (`log_entities = true`) and emits the detected strings alongside the count line:

```
INFO:     pii_proxi.mask:   secret: ' sk-live-AAAABBBBCCCCDDDD'
INFO:     pii_proxi.mask:   private_email: ' alice@example.com'
```

Do **not** enable `log_entities` on a shared machine, in CI, or anywhere stdout could be captured or shipped off-box — that defeats the point of the proxy.

## CLI

| Command | Purpose |
|---|---|
| `pii-proxi setup` | One-time: detect backend, write config, fetch model, warm up. |
| `pii-proxi serve` | Start the proxy. |
| `pii-proxi test "text"` | One-shot detection on a string. |
| `pii-proxi status` | Probe the running proxy's `/healthz`. |
| `pii-proxi clear-session` | Drop the in-memory placeholder map. |

## Threat model

Local-only (binds `127.0.0.1` by default). Forwards `x-api-key` / `Authorization` headers verbatim — the proxy is local-only and transparent at the auth layer, so your credentials never get inspected. The placeholder map is process-scoped and lives only in memory; nothing about the plaintext of flagged spans is logged or persisted. Clearing the map (`pii-proxi clear-session`) or restarting the process rolls a fresh session key.

## Roadmap

- Universal proxy mode: config-driven extractors so any upstream API can be PII-masked, not just Anthropic + OpenAI.
