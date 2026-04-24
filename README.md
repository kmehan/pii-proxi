# code-masker

`code-masker` is a local HTTP proxy that sits between coding assistants and their upstream APIs. It classifies outbound prompts, swaps detected secrets / PII / credentials for reversible placeholders, forwards the sanitized request, and unmasks placeholders in the streamed response so the assistant's suggested code still references your real identifiers. The goal: work with cloud coding assistants without constantly second-guessing what ended up in the prompt.

Detection runs on-device using OpenAI's open-weight privacy-filter model (~1.5 GB MLX 8-bit on Apple Silicon, ~2.6 GB ONNX FP16 elsewhere). Nothing about what was flagged leaves the machine.

## Install

Apple Silicon (recommended):

```bash
pip install -e ".[mlx,dev]"
```

Everything else:

```bash
pip install -e ".[onnx,dev]"
```

## Download the model

MLX build:

```bash
huggingface-cli download mlx-community/openai-privacy-filter-8bit \
    --local-dir ~/.cache/code-masker/models/mlx-8bit
```

ONNX build — point `model_path` in your config at wherever you downloaded the FP16 ONNX export.

## Run

```bash
code-masker serve
```

On startup the proxy prints the two `*_BASE_URL` lines to export. Example:

```
  code-masker listening on 127.0.0.1:8787
    export ANTHROPIC_BASE_URL=http://127.0.0.1:8787/anthropic
    export OPENAI_BASE_URL=http://127.0.0.1:8787/openai/v1
```

Paste those into the shell that'll launch your client, then fire up Claude Code / Codex CLI / aider / etc. as usual.

## Config

Optional TOML at `~/.config/code-masker/config.toml`:

```toml
port = 8787
backend = "mlx"                   # "mlx" | "onnx"
model_path = "~/.cache/code-masker/models/mlx-8bit"
calibration_path = "~/.cache/code-masker/models/viterbi_calibration.json"
disabled_labels = []              # e.g. ["EMAIL"] to skip that entity class
log_path = "~/.local/state/code-masker/audit.log"
anthropic_upstream = "https://api.anthropic.com"
openai_upstream = "https://api.openai.com"
```

Every field also reads from `CODE_MASKER_<NAME>` env vars for one-off overrides.

## CLI

| Command | Purpose |
|---|---|
| `code-masker serve` | Start the proxy. |
| `code-masker test "some text"` | One-shot detection, prints spans + masked form. |
| `code-masker status` | Probe the running proxy's `/healthz`. |
| `code-masker clear-session` | Drop the in-memory placeholder map. |

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

## Threat model

Local-only (binds `127.0.0.1` by default). Forwards `x-api-key` / `Authorization` headers verbatim — the proxy is transparent at the auth layer. The placeholder map is process-scoped and lives only in memory; nothing about the plaintext of flagged spans is logged or persisted. Clearing the map (`code-masker clear-session`) or restarting the process rolls a fresh session key.
