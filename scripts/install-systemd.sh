#!/usr/bin/env bash
# Description: Install the pii-proxi local proxy as a per-user systemd service on Linux.
set -euo pipefail

BIN="$(command -v pii-proxi || true)"
if [[ -z "${BIN}" ]]; then
  echo "error: pii-proxi binary not found on PATH." >&2
  echo "Install it first, for example:" >&2
  echo "  pipx install pii-proxi" >&2
  echo "  # or, from a checkout of this repo:" >&2
  echo "  pip install -e ." >&2
  exit 1
fi

UNIT_DIR="${HOME}/.config/systemd/user"
UNIT_PATH="${UNIT_DIR}/pii-proxi.service"

mkdir -p "${UNIT_DIR}"

cat > "${UNIT_PATH}" <<UNIT
[Unit]
Description=pii-proxi local PII-masking LLM proxy

[Service]
ExecStart=${BIN} serve
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
UNIT

systemctl --user daemon-reload
systemctl --user enable --now pii-proxi.service

echo "Installed systemd user unit: ${UNIT_PATH}"
echo "Tail logs:   journalctl --user -u pii-proxi -f"
echo "Status:      systemctl --user status pii-proxi"
