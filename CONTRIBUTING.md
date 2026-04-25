# Contributing to pii-proxi

Thanks for your interest in improving pii-proxi. This document covers the basics for getting set up and submitting changes.

## Dev setup

```bash
git clone https://github.com/kmehan/pii-proxi.git
cd pii-proxi
python -m venv .venv
source .venv/bin/activate
pip install -e ".[onnx,dev]"
```

On Apple Silicon, swap `[onnx,dev]` for `[mlx,dev]` to use the MLX backend.

## Running tests

```bash
pytest
```

Tests must use synthetic identities only — Ada Lovelace, Grace Hopper, addresses at `example.com`, and the like. Never check in real PII. This is a privacy tool; PII in fixtures is a leak, not a test artifact.

## Linting

```bash
ruff check .
```

## Pull requests

- Branch off `main`.
- CI must be green before merge.
- One approving review is required.
- Conventional-style commit subjects (`feat:`, `fix:`, `docs:`, ...) are encouraged.
- Link related issues in the PR description.

## Reporting bugs

Use the issue templates under `.github/ISSUE_TEMPLATE/`. Pick "bug report" or "feature request" and fill in the prompts.

## Reporting security issues

Do not file a public issue for a vulnerability. See [`SECURITY.md`](SECURITY.md) for the private reporting process.
