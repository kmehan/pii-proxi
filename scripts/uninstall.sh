#!/usr/bin/env bash
# Description: Uninstall the pii-proxi local proxy supervisor entry on macOS or Linux.
set -euo pipefail

PLATFORM="$(uname -s)"

case "${PLATFORM}" in
  Darwin)
    PLIST_PATH="${HOME}/Library/LaunchAgents/com.pii-proxi.plist"
    if [[ -f "${PLIST_PATH}" ]]; then
      launchctl unload "${PLIST_PATH}" 2>/dev/null || true
      rm -f "${PLIST_PATH}"
      echo "Removed launchd agent: ${PLIST_PATH}"
    else
      echo "No launchd agent found at ${PLIST_PATH}; nothing to do."
    fi
    ;;
  Linux)
    UNIT_PATH="${HOME}/.config/systemd/user/pii-proxi.service"
    if [[ -f "${UNIT_PATH}" ]]; then
      systemctl --user disable --now pii-proxi.service 2>/dev/null || true
      rm -f "${UNIT_PATH}"
      systemctl --user daemon-reload || true
      echo "Removed systemd user unit: ${UNIT_PATH}"
    else
      echo "No systemd user unit found at ${UNIT_PATH}; nothing to do."
    fi
    ;;
  *)
    echo "error: unsupported platform '${PLATFORM}'." >&2
    exit 1
    ;;
esac
