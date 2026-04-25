#!/usr/bin/env bash
# Description: Run the pii-proxi local proxy in the foreground for use under tmux/screen/foreman.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -f "${REPO_ROOT}/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.venv/bin/activate"
fi

if ! command -v pii-proxi >/dev/null 2>&1; then
  echo "error: pii-proxi binary not found on PATH." >&2
  echo "Install it first, for example:" >&2
  echo "  pipx install pii-proxi" >&2
  echo "  # or, from a checkout of this repo:" >&2
  echo "  pip install -e ." >&2
  exit 1
fi

exec pii-proxi serve "$@"
