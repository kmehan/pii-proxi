<div align="center">

# 🔒 pii-proxi

### Local PII-masking proxy for any LLM API

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-3776ab.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![Backend: MLX | ONNX](https://img.shields.io/badge/backend-MLX%20%7C%20ONNX-4ecdc4.svg?style=for-the-badge)](#configuration)
[![CI](https://img.shields.io/github/actions/workflow/status/kmehan/pii-proxi/ci.yml?branch=main&style=for-the-badge&label=tests)](https://github.com/kmehan/pii-proxi/actions/workflows/ci.yml)
[![Code style: Ruff](https://img.shields.io/badge/code%20style-ruff-f5a623.svg?style=for-the-badge)](https://github.com/astral-sh/ruff)

Strips names, emails, phones, addresses, and secrets out of your prompts **before** they leave the machine — then puts them back in the response. Powered by [OpenAI's Privacy Filter](https://cdn.openai.com/pdf/c66281ed-b638-456a-8ce1-97e9f5264a90/OpenAI-Privacy-Filter-Model-Card.pdf), running on your CPU. Works with **Anthropic**, **OpenAI**, **DeepSeek**, **Moonshot/Kimi**, **Groq**, **OpenRouter**, **Together**, **LM Studio** — anything OpenAI- or Anthropic-compatible.

[Why pii-proxi](#why-pii-proxi) · [How it works](#how-it-works) · [Detection quality](#detection-quality) · [Quick start](#quick-start) · [Configuration](#configuration) · [Threat model](#threat-model)

---

</div>

## Why pii-proxi

- **Runs locally.** No detection service, no telemetry, no cloud round-trip. Detection lands in ~50–100 ms per request on a laptop CPU (`latency_ms=65.1` is the typical figure from `pii-proxi status`).
- **Referentially intact.** The same `(label, value)` always mints the same `⟦label_hex8⟧` placeholder within a session, so the upstream model can reason about masked spans by stable IDs across multi-turn conversations and tool calls.
- **Lightweight.** 1.5B total / **50M active** parameters (sparse MoE, banded attention with effective window 257, 128k context). Runs in 4–8 GB of RAM on CPU, ~3 GB VRAM in FP16 on GPU. No GPU required.
- **Accurate where it matters.** 96% F1 on PII-Masking-300k (97.43% on the corrected variant); 0.91–0.93 F1 across English, French, German, Spanish, Italian, and Dutch. Dedicated `secret` label for API keys, OAuth bearers, and JWTs in code.

## How it works

```
                                ┌──────────────────────────────────────┐
                                │  pii-proxi  (127.0.0.1:8787)         │
   plaintext                    │   ┌────────────────────────────────┐ │   masked
   request  ─── extract ───────▶│──▶│  OpenAI Privacy Filter         │─▶│──▶  Anthropic / OpenAI /
                                │   │  on-device (MLX or ONNX)       │  │     DeepSeek / Groq /
                                │   │  1.5B total · 50M active · MoE │  │     Moonshot / OpenRouter /
                                │   └──────────────┬─────────────────┘  │     Together / LM Studio
                                │                  ▼                    │
                                │   PlaceholderMap  ⟶  ⟦label_hex8⟧     │
                                │   (per-session blake2b key)           │
   unmasked ◀── SSE/buffered ◀──│── unmask ◀── upstream stream ◀────────│
   response                     └──────────────────────────────────────┘

       Authorization / x-api-key forwarded verbatim. Plaintext never leaves the box.
```

- **Extract.** Per-API extractors pull text fields out of the request body — Anthropic recurses into `tool_result` and `tool_use.input`; OpenAI handles `tool_calls[].function.arguments` as nested JSON.
- **Detect & mask.** The Privacy Filter tags spans (BIOES + Viterbi over `tiktoken o200k_base`); each span becomes a `⟦label_hex8⟧` placeholder keyed by a per-session blake2b. The masked body is forwarded; `Authorization` and `x-api-key` ride through unchanged.
- **Unmask.** A buffered scanner handles complete responses; an SSE-aware reconstructor handles streams, so placeholders fragmented across `text_delta` / `input_json_delta` events still resolve cleanly before reaching your client.

## Detection quality

All numbers below are from the [OpenAI Privacy Filter Model Card](https://cdn.openai.com/pdf/c66281ed-b638-456a-8ce1-97e9f5264a90/OpenAI-Privacy-Filter-Model-Card.pdf) (April 22, 2026).

### Headline accuracy

| Benchmark | Precision | Recall | F1 |
|---|---|---|---|
| PII-Masking-300k | 94.04% | 98.04% | **96.0%** |
| PII-Masking-300k (corrected) | 96.79% | 98.08% | **97.43%** |

### Multilingual

F1 on PII-Masking-300k, broken down by language (model card, Table 6):

| English | Spanish | French | German | Italian | Dutch |
|---|---|---|---|---|---|
| 0.934 | 0.933 | 0.927 | 0.926 | 0.921 | 0.914 |

Strong on synthetic multilingual data too — Portuguese 0.933, Mandarin 0.917, Korean 0.895, Russian 0.895, Japanese 0.881 (model card, Table 7). Eleven more languages benchmarked there.

### Code & secrets

The `secret` label catches API keys, OAuth bearers, JWTs, and DB URIs. With surrounding code as context (the realistic regime — model card Table 5, "Clue + PII"), Privacy Filter posts **0.99 precision** on `secret`, so it's safe to wrap a coding assistant without false positives that scramble unrelated tokens. Recall on `secret` is 0.71 — high, but not 1.0, so still review prompts that touch production credentials.

### vs. Presidio / GLiNER

[Presidio](https://github.com/microsoft/presidio) is regex + classical NER — strong on IBANs and email shapes, brittle on ambiguous names and code. [GLiNER](https://github.com/urchade/GLiNER) is a general-purpose extractor that degrades sharply outside its training distribution. The Privacy Filter is an end-to-end transformer trained for this exact taxonomy with a bidirectional 257-token effective window — it disambiguates "Hopper" the person from "hopper" the verb because it sees the whole sentence.

## Quick start

Requires Python 3.11+. Pick `[mlx]` on Apple Silicon, `[onnx]` everywhere else.

```bash
# 1. Install
pipx install "git+https://github.com/kmehan/pii-proxi.git#egg=pii-proxi[mlx]"   # Apple Silicon
pipx install "git+https://github.com/kmehan/pii-proxi.git#egg=pii-proxi[onnx]"  # Linux / Intel / Windows

# 2. One-time setup — fetches model + calibration (~2 GB) into ~/.cache/pii-proxi/models/
pii-proxi setup
pii-proxi serve   # binds 127.0.0.1:8787

# 3. Sanity-check the detector
pii-proxi test "my key is sk-live-AAAABBBBCCCCDDDD and email foo@example.com"
```

### Point your client at the proxy

```bash
# Anthropic (Claude Code, Claude Pro/Max OAuth, anything that honors ANTHROPIC_BASE_URL)
export ANTHROPIC_BASE_URL=http://127.0.0.1:8787/anthropic

# OpenAI-compatible (Codex CLI, aider, continue.dev, Cursor BYOK, …)
export OPENAI_BASE_URL=http://127.0.0.1:8787/openai/v1
```

The proxy is transparent at the auth layer: `Authorization` and `x-api-key` are forwarded verbatim, so API keys **and** Pro/Max OAuth bearers ride through unchanged.

### Multiple providers

Register as many upstreams as you want in `~/.config/pii-proxi/config.toml` under `[providers.<name>]`. Each entry is mounted at `/<name>` on the proxy, with `format = "openai"` exposing `/<name>/v1/chat/completions` and `format = "anthropic"` exposing `/<name>/v1/messages`.

```toml
[providers.openai]
format = "openai"
upstream = "https://api.openai.com"

[providers.deepseek]
format = "anthropic"
upstream = "https://api.deepseek.com/anthropic"
```

Then point each SDK at its own mount:

```bash
export OPENAI_BASE_URL=http://127.0.0.1:8787/openai/v1
export ANTHROPIC_BASE_URL=http://127.0.0.1:8787/deepseek
```

Run `pii-proxi providers` to print the registered set with their local URLs. The legacy `anthropic_upstream` / `openai_upstream` keys still work and seed default `anthropic` / `openai` providers when no `[providers]` table is defined. System-wide / browser interception (no per-tool base-URL config) is planned but requires installing a local CA — tracked separately.

### Swap the upstream

Set `PII_PROXI_OPENAI_UPSTREAM` to send `/openai/*` anywhere that speaks Chat Completions:

| Upstream | `PII_PROXI_OPENAI_UPSTREAM` | Docs |
|---|---|---|
| DeepSeek | `https://api.deepseek.com` | <https://api-docs.deepseek.com> |
| Moonshot / Kimi | `https://api.moonshot.cn` | <https://platform.moonshot.cn/docs> |
| Groq | `https://api.groq.com/openai` | <https://console.groq.com/docs> |
| OpenRouter | `https://openrouter.ai/api` | <https://openrouter.ai/docs> |
| Together | `https://api.together.xyz` | <https://docs.together.ai> |
| LM Studio (local) | `http://127.0.0.1:1234` | <https://lmstudio.ai/docs> |

Use the route that matches your client's wire format: `/anthropic/v1/messages` for Anthropic, `/openai/v1/chat/completions` for OpenAI-compatible. The proxy is a PII-masking proxy, not a format translator.

DeepSeek also exposes an **Anthropic-compatible** endpoint, so you can route the `/anthropic` side through it too:

| Upstream | `PII_PROXI_ANTHROPIC_UPSTREAM` | Docs |
|---|---|---|
| DeepSeek (Anthropic format) | `https://api.deepseek.com/anthropic` | <https://api-docs.deepseek.com/guides/anthropic_api> |

Pick any DeepSeek model name in the request body — unsupported names fall back to `deepseek-v4-flash` server-side.

### Supported clients

| Client | Auth mode | Supported | Notes |
|---|---|---|---|
| Claude Code | `ANTHROPIC_API_KEY` | yes | Standard path. |
| Claude Code | Claude Pro/Max login (OAuth) | yes | OAuth bearer rides through `Authorization` like an API key. |
| Codex CLI | `OPENAI_API_KEY` | yes | Standard path. |
| Codex CLI | "Sign in with ChatGPT" (OAuth) | warn / verify | Likely works if Codex honors `OPENAI_BASE_URL` in OAuth mode. Verify locally first. |
| aider, continue.dev, any OpenAI-compat tool | API key | yes | Same `OPENAI_BASE_URL` route. |
| Cursor | BYO API key | yes | Uses Cursor's custom-base-URL setting. |
| Cursor | Cursor Pro subscription | no | Vendor-managed auth via `api.cursor.sh`; no client-side base-URL override. |

## Configuration

`pii-proxi setup` writes a working default to `~/.config/pii-proxi/config.toml`. Every field also reads from a `PII_PROXI_<NAME>` env var (env wins over TOML).

| TOML key | Env var | Description | Default |
|---|---|---|---|
| `port` | `PII_PROXI_PORT` | Bind port | `8787` |
| `host` | `PII_PROXI_HOST` | Bind host | `127.0.0.1` |
| `backend` | `PII_PROXI_BACKEND` | `mlx` or `onnx` | auto-detected by `setup` |
| `model_path` | `PII_PROXI_MODEL_PATH` | Detector weights directory | `~/.cache/pii-proxi/models/...` |
| `calibration_path` | `PII_PROXI_CALIBRATION_PATH` | Viterbi calibration JSON | `~/.cache/pii-proxi/models/viterbi_calibration.json` |
| `disabled_labels` | `PII_PROXI_DISABLED_LABELS` | Skip these labels | `[]` |
| `anthropic_upstream` | `PII_PROXI_ANTHROPIC_UPSTREAM` | Where `/anthropic/*` is forwarded | `https://api.anthropic.com` |
| `openai_upstream` | `PII_PROXI_OPENAI_UPSTREAM` | Where `/openai/*` is forwarded | `https://api.openai.com` |
| `log_path` | `PII_PROXI_LOG_PATH` | Audit log destination | `~/.local/state/pii-proxi/audit.log` |
| `log_entities` | `PII_PROXI_LOG_ENTITIES` | **Opt-in** plaintext span logging — leave `false` in any shared environment | `false` |

Detection labels come from the model's `config.json:id2label` and are emitted in lowercase: `private_person`, `private_email`, `private_phone`, `private_address`, `private_url`, `private_date`, `account_number`, `secret`.

### Customize

- **Disable a label** — `PII_PROXI_DISABLED_LABELS='["private_date"]'` (e.g. if dates are noisy in your domain).
- **Swap the upstream** — `PII_PROXI_OPENAI_UPSTREAM=https://api.deepseek.com pii-proxi serve`.
- **Reset session keys** — `pii-proxi clear-session` rotates the per-session blake2b key, so placeholder hashes can't be correlated across runs.

### Run it as a service

| Platform | Command |
|---|---|
| macOS (launchd) | `./scripts/install-launchd.sh` |
| Linux (systemd user unit) | `./scripts/install-systemd.sh` |
| Manual / dev | `pii-proxi serve` under tmux or screen |

Uninstall with `./scripts/uninstall.sh`.

### CLI

| Command | Purpose |
|---|---|
| `pii-proxi setup` | One-time: detect backend, write config, fetch model. |
| `pii-proxi serve` | Start the proxy. |
| `pii-proxi test "text"` | One-shot detection on a string. |
| `pii-proxi status` | Probe the running proxy's `/healthz`. |
| `pii-proxi clear-session` | Drop the in-memory placeholder map and rotate the session key. |

## Threat model

Local-only — binds `127.0.0.1` by default. Detection runs on-device; no third-party detection service is contacted, ever. `Authorization` and `x-api-key` are forwarded verbatim, so credentials are never inspected. The placeholder map is process-scoped and lives only in memory — nothing about the plaintext of flagged spans is logged or persisted unless you opt into `log_entities`. Restarting the process or running `pii-proxi clear-session` rolls a fresh per-session blake2b key.

Counts are always logged to stdout — no plaintext, safe to leave on:

```
INFO:     pii_proxi.mask: masked 2 span(s) across 1 text(s): private_email=1, secret=1
```

Plaintext logging is **opt-in** (`log_entities = true`) and emits the detected strings:

```
INFO:     pii_proxi.mask:   secret: ' sk-live-AAAABBBBCCCCDDDD'
INFO:     pii_proxi.mask:   private_email: ' alice@example.com'
```

Do not enable `log_entities` on a shared machine, in CI, or anywhere stdout could be captured. It exists for local debugging only and defeats the entire point of the proxy if left on.

Vulnerabilities go through [`SECURITY.md`](SECURITY.md) — please don't open public issues for security reports.

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

Roadmap: universal config-driven extractors so any upstream API shape can be masked, per-label detection thresholds, encrypted persistent placeholder maps across restarts.

## Contributing

Bugs and feature requests via [Issues](https://github.com/kmehan/pii-proxi/issues). PRs welcome — please run `ruff check . && pytest` before opening one. Any change that affects what leaves the local machine (logging, telemetry, header handling, upstream forwarding) is security-sensitive — call it out in the PR description.

## License

[MIT](LICENSE).
