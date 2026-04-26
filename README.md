<div align="center">

# 🔒 pii-proxi

### Local PII-masking proxy for any LLM API

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-3776ab.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![Backend: MLX | ONNX](https://img.shields.io/badge/backend-MLX%20%7C%20ONNX-4ecdc4.svg?style=for-the-badge)](#configuration)
[![CI](https://img.shields.io/github/actions/workflow/status/kmehan/pii-proxi/ci.yml?branch=main&style=for-the-badge&label=tests)](https://github.com/kmehan/pii-proxi/actions/workflows/ci.yml)
[![Code style: Ruff](https://img.shields.io/badge/code%20style-ruff-f5a623.svg?style=for-the-badge)](https://github.com/astral-sh/ruff)

Strip PII and secrets from prompts **before** they leave your machine, and seamlessly restore them in responses. Works with **Anthropic**, **OpenAI**, **DeepSeek**, **Moonshot/Kimi**, **Groq**, **OpenRouter**, **Together**, **LM Studio** — anything OpenAI- or Anthropic-compatible. Detection runs entirely on-device using OpenAI's open-weight privacy-filter model.

The proxy preserves **referential integrity**: the same value always gets the same `⟦label_hex8⟧` placeholder within a session, so the model can reason about and refer back to masked spans by stable IDs, and your client never sees them in the response.

[Quick Start](#quick-start) · [How It Works](#how-it-works) · [Providers](#providers) · [Clients](#supported-clients) · [Configuration](#configuration) · [Threat Model](#threat-model) · [Development](#development)

---

</div>

## Features

| Feature | What it does |
|---|---|
| **On-device detection** | Runs OpenAI's open-weight privacy-filter model locally (MLX on Apple Silicon, ONNX elsewhere). No text leaves your box for detection. |
| **Referential integrity** | Same `(label, value)` always mints the same `⟦label_hex8⟧` placeholder within a session, so models reason consistently and can reference masked spans by ID. |
| **Any OpenAI-compatible upstream** | Override `PII_PROXI_OPENAI_UPSTREAM` to route to DeepSeek, Moonshot/Kimi, Groq, OpenRouter, Together, LM Studio — no code changes. |
| **First-class Anthropic Messages** | `/v1/messages` with full recursion into `tool_result` and `tool_use.input`. |
| **Transparent auth** | `Authorization` and `x-api-key` forwarded verbatim. Works with API keys **and** Claude Pro/Max OAuth. |
| **SSE-aware streaming** | Reconstructs placeholders fragmented across `text_delta` / `input_json_delta` events before unmasking, so streaming clients see clean output. |
| **No-leak by default** | Masked-span counts logged; plaintext logging is strictly opt-in (`log_entities = false` is the default and the recommended setting). |
| **Drop-in CLI** | `pii-proxi setup && pii-proxi serve`, then point your client at `http://127.0.0.1:8787`. |

## Quick Start

Requires Python 3.11+. Pick `[mlx]` on Apple Silicon (fastest), `[onnx]` on Linux / Intel / Windows.

> Works with API keys **and** Claude Pro/Max OAuth — the auth header is forwarded verbatim, so the proxy is transparent at the auth layer.

### 1. Install

<details>
<summary><b>Option A — pipx (recommended)</b></summary>

```bash
pipx install "git+https://github.com/kmehan/pii-proxi.git#egg=pii-proxi[mlx]"
# or [onnx] on non-Apple-Silicon
```

</details>

<details>
<summary><b>Option B — clone for development</b></summary>

```bash
git clone https://github.com/kmehan/pii-proxi.git && cd pii-proxi
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[mlx]"        # or ".[onnx]"
```

</details>

### 2. First-run setup

```bash
pii-proxi setup     # detects backend, fetches model + calibration (~2 GB), writes default config
pii-proxi serve     # binds 127.0.0.1:8787 by default
```

`setup` writes `~/.config/pii-proxi/config.toml` and downloads weights to `~/.cache/pii-proxi/models/`. It's idempotent — safe to re-run.

### 3. Point your client at the proxy

#### Claude Code → Anthropic (default)

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:8787/anthropic
claude
```

Works with `ANTHROPIC_API_KEY` and Pro/Max login alike — the auth bearer rides through `Authorization` either way.

#### Codex CLI / aider / continue.dev → OpenAI (default)

```bash
export OPENAI_BASE_URL=http://127.0.0.1:8787/openai/v1
codex   # or: aider, continue
```

#### …with **DeepSeek** instead of OpenAI

Tell the proxy to forward `/openai/*` to DeepSeek, then point your client at the proxy:

```bash
# Shell A — start the proxy with DeepSeek as the upstream
export PII_PROXI_OPENAI_UPSTREAM=https://api.deepseek.com
pii-proxi serve

# Shell B — your client
export OPENAI_BASE_URL=http://127.0.0.1:8787/openai/v1
export OPENAI_API_KEY=sk-deepseek-your-key   # forwarded verbatim to DeepSeek
aider --model deepseek-chat
```

Your name, your teammates' emails, the `sk-…` keys in your shell history — none of it ever reaches DeepSeek. The model sees `⟦private_person_a1b2c3d4⟧` and friends; the response is unmasked locally before it lands in your terminal.

#### …with **Moonshot / Kimi**, **Groq**, **OpenRouter**, **Together**

Same recipe, different upstream:

| Upstream | `PII_PROXI_OPENAI_UPSTREAM` | Docs |
|---|---|---|
| DeepSeek | `https://api.deepseek.com` | <https://api-docs.deepseek.com> |
| Moonshot / Kimi | `https://api.moonshot.cn` | <https://platform.moonshot.cn/docs> |
| Groq | `https://api.groq.com/openai` | <https://console.groq.com/docs> |
| OpenRouter | `https://openrouter.ai/api` | <https://openrouter.ai/docs> |
| Together | `https://api.together.xyz` | <https://docs.together.ai> |
| LM Studio (local) | `http://127.0.0.1:1234` | <https://lmstudio.ai/docs> |

The proxy POSTs to `{upstream}/v1/chat/completions`, so set the upstream to the host root (without the `/v1` suffix unless the provider's docs already include a path prefix like Groq's `/openai`).

### 4. Sanity-check the detector

```bash
pii-proxi test "my key is sk-live-AAAABBBBCCCCDDDD and email foo@example.com"
```

You should see the spans flagged and the masked form printed.

---

## How It Works

```
┌──────────────┐   plaintext    ┌──────────────┐   masked     ┌──────────────┐
│  Your CLI    │ ─────────────▶ │  pii-proxi   │ ───────────▶ │   Upstream   │
│ Claude Code, │                │  127.0.0.1:  │              │  Anthropic,  │
│ aider, codex │ ◀───────────── │     8787     │ ◀─────────── │  DeepSeek,   │
└──────────────┘  unmasked SSE  └──────────────┘  masked SSE  │  Moonshot…   │
                                                              └──────────────┘
       Auth header forwarded verbatim. Plaintext never leaves the box.
```

- **Extract → detect → mask.** Per-API extractors pull text fields out of the request body (Anthropic recurses into `tool_result` and `tool_use.input`; OpenAI handles `tool_calls[].function.arguments` as nested JSON). The on-device detector tags spans. Each span is replaced with a `⟦label_hex8⟧` placeholder. The masked body is forwarded.
- **Same value → same placeholder.** A process-scoped `PlaceholderMap` keyed by a per-session blake2b key gives every distinct `(label, original)` pair a stable ID. The model sees consistent IDs across turns, so it can refer back to a masked span by its placeholder and the response unmasks correctly. Restart the process (or `pii-proxi clear-session`) to roll a fresh key.
- **Forward upstream.** `Authorization` and `x-api-key` are relayed unchanged. Pro/Max OAuth bearers, `sk-…` keys, NIM keys — the proxy doesn't inspect them.
- **Unmask response.** A byte-stream scanner handles buffered responses; an SSE-aware reconstructor (`masking/sse_unmask.py`) handles streaming, so placeholders fragmented across event boundaries still resolve cleanly before reaching your client.

## Providers

The proxy exposes two routes — pick the one that matches your client's wire format. Both routes accept any backend that speaks the matching format.

| Route | Upstream override env | Speaks | Default |
|---|---|---|---|
| `/anthropic/v1/messages` | `PII_PROXI_ANTHROPIC_UPSTREAM` | Anthropic Messages API | `https://api.anthropic.com` |
| `/openai/v1/chat/completions` | `PII_PROXI_OPENAI_UPSTREAM` | OpenAI Chat Completions API | `https://api.openai.com` |

> The proxy is a **PII-masking** proxy, not a format translator: it does not convert between Anthropic and OpenAI wire formats. Use the route that matches what your client speaks.

## Supported Clients

| Client | Auth mode | Supported | Notes |
|---|---|---|---|
| Claude Code | `ANTHROPIC_API_KEY` | yes | Standard path. |
| Claude Code | Claude Pro/Max login (OAuth) | yes | `ANTHROPIC_BASE_URL` is honored; the OAuth bearer rides through `Authorization` the same way an API key does. |
| Codex CLI | `OPENAI_API_KEY` | yes | Standard path. |
| Codex CLI | "Sign in with ChatGPT" (OAuth) | warn / verify | Likely works if Codex honors `OPENAI_BASE_URL` in OAuth mode and the token isn't pinned to a ChatGPT-specific backend. Verify locally before relying on it for sensitive prompts. |
| aider, continue.dev (any OpenAI-compat tool) | API key | yes | Same `OPENAI_BASE_URL` route. |
| Cursor | BYO API key | yes | Uses Cursor's custom-base-URL setting. |
| Cursor | Cursor Pro subscription | no | Cursor Pro routes through `api.cursor.sh` with vendor-managed auth; no client-side base-URL override. Would require TLS MITM, which is out of scope. |

---

## Configuration

`pii-proxi setup` writes a working default to `~/.config/pii-proxi/config.toml`. Every field also reads from a `PII_PROXI_<NAME>` env var for one-off overrides (env wins over TOML).

| TOML key | Env var | Description | Default |
|---|---|---|---|
| `port` | `PII_PROXI_PORT` | Bind port | `8787` |
| `host` | `PII_PROXI_HOST` | Bind host | `127.0.0.1` |
| `backend` | `PII_PROXI_BACKEND` | `mlx` or `onnx` | auto-detected by `setup` |
| `model_path` | `PII_PROXI_MODEL_PATH` | Detector weights directory | `~/.cache/pii-proxi/models/...` |
| `calibration_path` | `PII_PROXI_CALIBRATION_PATH` | Viterbi calibration JSON | `~/.cache/pii-proxi/models/viterbi_calibration.json` |
| `disabled_labels` | `PII_PROXI_DISABLED_LABELS` | Skip these labels (e.g. `["private_email"]`) | `[]` |
| `anthropic_upstream` | `PII_PROXI_ANTHROPIC_UPSTREAM` | Where `/anthropic/*` is forwarded | `https://api.anthropic.com` |
| `openai_upstream` | `PII_PROXI_OPENAI_UPSTREAM` | Where `/openai/*` is forwarded | `https://api.openai.com` |
| `log_path` | `PII_PROXI_LOG_PATH` | Audit log destination | `~/.local/state/pii-proxi/audit.log` |
| `log_entities` | `PII_PROXI_LOG_ENTITIES` | **Opt-in** plaintext span logging — leave `false` in any shared environment | `false` |

Detection labels are read from the model's `config.json:id2label` and emitted in lowercase (e.g. `private_person`, `private_email`, `secret`, …). Use `disabled_labels` to skip a class without disabling the proxy.

### Run it as a service

| Platform | Command |
|---|---|
| macOS (launchd) | `./scripts/install-launchd.sh` |
| Linux (systemd user unit) | `./scripts/install-systemd.sh` |
| Manual / dev | `pii-proxi serve` under tmux or screen |

Uninstall with `./scripts/uninstall.sh`.

---

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

Do **not** enable `log_entities` on a shared machine, in CI, or anywhere stdout could be captured or shipped off-box — it defeats the entire point of the proxy. It exists for local debugging only.

## Threat Model

Local-only — binds `127.0.0.1` by default. Forwards `x-api-key` and `Authorization` headers verbatim, so credentials are never inspected by the proxy. The placeholder map is process-scoped and lives only in memory; nothing about the plaintext of flagged spans is logged or persisted unless you opt into `log_entities`. Clearing the map (`pii-proxi clear-session`) or restarting the process rolls a fresh per-session blake2b key, so placeholder hashes can't be correlated across runs.

The only supported mechanism that can leak plaintext is `log_entities = true`. It's off by default.

Vulnerabilities go through [`SECURITY.md`](SECURITY.md) — please don't open public issues for security reports.

## CLI

| Command | Purpose |
|---|---|
| `pii-proxi setup` | One-time: detect backend, write config, fetch model, warm up. |
| `pii-proxi serve` | Start the proxy. |
| `pii-proxi test "text"` | One-shot detection on a string. |
| `pii-proxi status` | Probe the running proxy's `/healthz`. |
| `pii-proxi clear-session` | Drop the in-memory placeholder map and rotate the session key. |

---

## Development

```bash
git clone https://github.com/kmehan/pii-proxi.git && cd pii-proxi
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[onnx,dev]"        # or ".[mlx,dev]" on Apple Silicon

ruff check .
pytest
```

- Tests use a `FakeDetector` from `tests/conftest.py` — CI never downloads the real model.
- CI matrix: `ubuntu-latest` + `macos-latest` × Python `3.11` + `3.12`.
- The `[mlx]` extra is **never** installed in CI (it pulls a git dep and requires Apple Silicon). Use `[onnx,dev]`.
- Per-API logic split: `routes/<api>.py` + `masking/extractor.py` per format; `routes/_common.py:proxy_roundtrip` is the generic transport. Adding a new upstream API means adding one route file and one extractor.

## Roadmap

- Universal proxy mode: config-driven extractors so any upstream API shape can be PII-masked, not just Anthropic + OpenAI.
- Per-label detection thresholds.
- Persistent placeholder maps across restarts (opt-in, encrypted at rest).

## Contributing

Bugs and feature requests via [Issues](https://github.com/kmehan/pii-proxi/issues). PRs welcome — please run `ruff check . && pytest` before opening one. Any change that affects what leaves the local machine (logging, telemetry, header handling, upstream forwarding) is security-sensitive — call it out in the PR description.

## License

[MIT](LICENSE).
