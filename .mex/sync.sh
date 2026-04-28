#!/usr/bin/env bash
set -euo pipefail

if ! command -v mex >/dev/null 2>&1; then
  echo "mex CLI not found. Install it first, or run via 'npx promexeus'." >&2
  exit 1
fi

exec mex sync --warnings "$@"
