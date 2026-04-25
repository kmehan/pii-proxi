---
name: Bug report
about: Report something that doesn't work as documented
title: ""
labels: ["bug"]
---

> If this is a security or privacy concern (e.g., a placeholder that leaked, plaintext that reached the upstream), see [`SECURITY.md`](../../SECURITY.md) instead — do not file a public issue.

## What happened

A clear, concise description of the actual behavior.

## What you expected

A clear, concise description of the expected behavior.

## Reproduction steps

1. Run `pii-proxi serve` with config `...`
2. Send request `...`
3. Observe `...`

A minimal masked input (use synthetic data — Ada Lovelace, `*@example.com`, etc.) is the most useful thing you can include.

## Environment

- OS + version:
- Python version (`python --version`):
- Backend (`mlx` / `onnx`):
- Install method (`pipx`, `pip install -e .`, ...):
- `pii-proxi` version or commit SHA:
