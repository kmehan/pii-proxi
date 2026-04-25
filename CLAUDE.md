# pii-proxi — project rules for Claude

Local PII-masking proxy for LLM APIs. Privacy tool: every rule below is load-bearing.

## Commits

- Branch off `main`. Never push directly to `main` (branch protection blocks it).
- Conventional prefixes: `feat:`, `fix:`, `docs:`, `chore:`, `refactor:`, `test:`, `ci:`.
- One logical change per commit. Bug fixes get their own commit, separate from refactors.
- Every commit message ends with the `Co-Authored-By:` trailer for Claude.
- Squash or rebase merge only. No merge commits.
- No `--no-verify`, no `--force` to `main`.

## Tests

- Synthetic identities only: Ada Lovelace, Grace Hopper, Alan Turing, `*@example.com`.
- Never check in real names, real emails, real API keys, real secrets — even in fixtures or commit messages.
- Run before pushing: `pytest` and `ruff check .`. Both must be green.
- Tests use `FakeDetector` (see `tests/conftest.py`). Don't add tests that download the real model — CI has no GPU and no network for HuggingFace.
- Detector emits **lowercase** labels (`private_email`, `private_person`, `secret`...). Use lowercase in any new test that exercises the masking pipeline.

## Code

- Package: `pii_proxi` (underscore). CLI: `pii-proxi` (hyphen). Env-var prefix: `PII_PROXI_*`.
- Logger root: `pii_proxi.*`.
- Config: `~/.config/pii-proxi/config.toml`. Model cache: `~/.cache/pii-proxi/models/`.
- Placeholder format: `⟦{label}_{hex8}⟧`. Don't change the delimiters or the hex length.
- `routes/_common.py:proxy_roundtrip` is the generic transport. Per-API logic lives in `routes/<api>.py` + `masking/extractor.py`. Keep the split.
- Default to no comments. Only comment WHY, never WHAT. No multi-paragraph docstrings.

## CI

- The `[mlx]` extra is **never** installed in CI. It pulls from a git URL and requires Apple Silicon. Use `[onnx,dev]`.
- CI matrix: `ubuntu-latest` + `macos-latest` × Python `3.11` + `3.12`. All four checks are required by branch protection.

## Don't

- Don't enable `log_entities = true` in any committed config or example.
- Don't log secrets to stdout/stderr in new code paths.
- Don't add `mlx` to CI.
- Don't run `pii-proxi setup` to test — it downloads ~2 GB. Use `FakeDetector` in tests instead.
- Don't publish to PyPI without explicit ask.
- Don't rename `pii-proxi` again. The previous rename from `code-masker` is documented in PR #1.

## Security

- Vulnerabilities go through `SECURITY.md`, never a public issue.
- Any change that affects what leaves the local machine (logging, telemetry, upstream forwarding, header handling) is security-sensitive — call it out in the PR description.
