# code-masker

`code-masker` is a local HTTP proxy that sits between coding assistants and their upstream APIs. It classifies outbound prompts, swaps detected secrets / PII / credentials for reversible placeholders, forwards the sanitized request, and unmasks placeholders in the streamed response so the assistant's suggested code still references your real identifiers. The goal: work with cloud coding assistants without constantly second-guessing what ended up in the prompt.

Detection runs on-device using OpenAI's open-weight privacy-filter model (~1.5 GB MLX 8-bit on Apple Silicon, ~2.6 GB ONNX FP16 elsewhere). Nothing about what was flagged leaves the machine.

## Setup

Requires Python 3.11+.

### 1. Clone and create a virtualenv

```bash
git clone <this repo> code-masker
cd code-masker
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install

Apple Silicon (recommended — uses MLX, fastest on M-series):

```bash
pip install -e ".[mlx,dev]"
```

> **Note:** the `[mlx]` extra pulls `mlx-embeddings` from git (the `openai_privacy_filter` architecture isn't in its 0.1.0 PyPI release yet). This will be swapped back to a PyPI pin once the maintainer cuts a release containing it.

Linux / Windows / Intel Mac (ONNX backend):

```bash
pip install -e ".[onnx,dev]"
```

After install, `code-masker --help` should work *while the venv is active*. The binary lives at `.venv/bin/code-masker` — if you want it on your global `$PATH`, install with `pipx install -e ".[mlx]"` instead of the commands above.

### 3. Download the model

```bash
pip install huggingface_hub           # if you don't have it already; ships the `hf` CLI

# MLX weights (Apple Silicon build)
hf download mlx-community/openai-privacy-filter-8bit \
    --local-dir ~/.cache/code-masker/models/mlx-8bit

# Viterbi calibration file — lives in the ONNX repo but is needed by both backends
hf download yasserrmd/privacy-filter-ONNX viterbi_calibration.json \
    --local-dir ~/.cache/code-masker/models
```

> The older `huggingface-cli` command is deprecated; use `hf` instead. If you see the deprecation warning, you're on a recent `huggingface_hub` — `hf` is already installed.

ONNX users also download the FP16 export from `yasserrmd/privacy-filter-ONNX` and point `model_path` at that directory (see Config below).

### 4. Create the config

Copy the example config:

```bash
mkdir -p ~/.config/code-masker
cp config.example.toml ~/.config/code-masker/config.toml
```

The example is set up for the MLX backend with the paths from step 3. Edit it if you're on ONNX or put the model somewhere else.

<details>
<summary>Prefer to create it by hand?</summary>

```bash
mkdir -p ~/.config/code-masker
nano ~/.config/code-masker/config.toml     # paste the three lines, Ctrl-O, Ctrl-X
```

Minimum contents:

```toml
backend = "mlx"
model_path = "~/.cache/code-masker/models/mlx-8bit"
calibration_path = "~/.cache/code-masker/models/viterbi_calibration.json"
```

(Avoid the `cat > file <<'EOF' … EOF` heredoc form interactively — the closing `EOF` has to be at column 0 on its own line, and it's easy to get stuck in an unterminated heredoc.)

</details>

### 5. Start the proxy

```bash
code-masker serve
```

You should see:

```
  code-masker listening on 127.0.0.1:8787
    export ANTHROPIC_BASE_URL=http://127.0.0.1:8787/anthropic
    export OPENAI_BASE_URL=http://127.0.0.1:8787/openai/v1
```

Leave this terminal running.

### 6. Point your coding assistant at the proxy

In a **separate terminal**, export the base URL for whichever client you use, then launch it normally:

**Claude Code** (API key *or* Pro/Max OAuth — both work):

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:8787/anthropic
claude
```

**Codex CLI / aider / continue.dev / any OpenAI-compatible client:**

```bash
export OPENAI_BASE_URL=http://127.0.0.1:8787/openai/v1
codex    # or: aider, continue, etc.
```

**Cursor (BYO API key mode only):** Settings → Models → OpenAI Base URL → `http://127.0.0.1:8787/openai/v1`. Cursor Pro subscriptions are *not* supported — see the table below.

### Smoke test

Before wiring a real client, sanity-check the detector on a throwaway string:

```bash
code-masker test "my key is sk-live-AAAABBBBCCCCDDDD and email foo@bar.com"
```

This prints detected spans and the masked form. Loading the MLX model takes a few seconds the first time.

## Observing masking activity

Every proxied request emits a one-line summary to the proxy's stdout showing how many spans were masked, broken down by label:

```
INFO:     code_masker.mask: masked 2 span(s) across 1 text(s): private_email=1, secret=1
INFO:     127.0.0.1:65155 - "POST /anthropic/v1/messages HTTP/1.1" 200 OK
```

Counts only — no plaintext. Safe to leave on anywhere.

For local debugging you can also log the detected plaintext alongside each label. This is **off by default** (a privacy tool shouldn't log secrets unless you ask it to). Turn it on by adding one line to your config:

```bash
echo 'log_entities = true' >> ~/.config/code-masker/config.toml
```

Then restart the proxy. You'll additionally see:

```
INFO:     code_masker.mask:   secret: ' sk-live-AAAABBBBCCCCDDDD'
INFO:     code_masker.mask:   private_email: ' alice@example.com'
```

**Do not enable `log_entities` on a shared machine, in a CI job, or anywhere the proxy's stdout could be captured, uploaded, or shipped off-box** — the whole point of the proxy is to keep those strings off the wire, and logging them locally defeats that if the logs aren't private.

## Config

Optional TOML at `~/.config/code-masker/config.toml`:

```toml
port = 8787
backend = "mlx"                   # "mlx" | "onnx"
model_path = "~/.cache/code-masker/models/mlx-8bit"
calibration_path = "~/.cache/code-masker/models/viterbi_calibration.json"
disabled_labels = []              # e.g. ["private_email"] to skip that entity class
log_path = "~/.local/state/code-masker/audit.log"
anthropic_upstream = "https://api.anthropic.com"
openai_upstream = "https://api.openai.com"
log_entities = false              # see "Observing masking activity" — off by default
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
