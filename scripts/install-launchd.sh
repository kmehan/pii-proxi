#!/usr/bin/env bash
# Description: Install the pii-proxi local proxy as a per-user launchd agent on macOS.
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

LABEL="com.pii-proxi"
PLIST_DIR="${HOME}/Library/LaunchAgents"
PLIST_PATH="${PLIST_DIR}/${LABEL}.plist"
LOG_DIR="${HOME}/Library/Logs"
LOG_PATH="${LOG_DIR}/pii-proxi.log"

mkdir -p "${PLIST_DIR}" "${LOG_DIR}"

# Best-effort unload of any previously installed instance; ignore if not loaded.
if [[ -f "${PLIST_PATH}" ]]; then
  launchctl unload "${PLIST_PATH}" 2>/dev/null || true
fi

cat > "${PLIST_PATH}" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>${LABEL}</string>
    <key>ProgramArguments</key>
    <array>
        <string>${BIN}</string>
        <string>serve</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>${LOG_PATH}</string>
    <key>StandardErrorPath</key>
    <string>${LOG_PATH}</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>${PATH}</string>
        <key>HOME</key>
        <string>${HOME}</string>
    </dict>
</dict>
</plist>
PLIST

launchctl load "${PLIST_PATH}"

echo "Installed launchd agent: ${PLIST_PATH}"
echo "Logs: ${LOG_PATH}"
echo "Tail logs:   tail -f \"${LOG_PATH}\""
echo "Status:      launchctl list | grep pii-proxi"
