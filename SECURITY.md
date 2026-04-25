# Security policy

`pii-proxi` is a privacy tool, so security and correctness are the same thing here. If you find a way prompts can leak past the proxy, treat it as a vulnerability, not a bug.

## Reporting a vulnerability

Please do **not** open a public GitHub issue for security reports.

Two private channels:

- **Preferred:** GitHub's private vulnerability reporting — go to the repo's **Security** tab and click **Report a vulnerability**.
- **Fallback:** email `kunal.mehan@gmail.com` with subject prefix `[pii-proxi security]`.

Include enough detail to reproduce: a minimal input, the expected behavior, and what the proxy actually did.

## What counts

Examples of in-scope vulnerabilities:

- A PII span the detector misses for a known-good input class (e.g., a standard email format that bypasses masking).
- A placeholder that fails to round-trip through `UnmaskStream` and reaches the client unmasked.
- Any path where prompt plaintext leaves the local machine when masking is enabled (logs shipped off-box, telemetry, accidental upstream leakage).
- Auth-header leakage or modification (the proxy is supposed to be transparent at the auth layer).

Out of scope: model accuracy on adversarial or out-of-distribution inputs, performance issues, missing detector classes that require a model retrain.

## Response timeline

- Acknowledgment within **7 days** of receiving a report (best-effort, solo maintainer).
- Aim to ship a fix or documented workaround within **30 days** for confirmed issues.

## Supported versions

Only the current `main` branch is supported during pre-1.0 development. Once a version is tagged, this section will be updated with a support matrix.
