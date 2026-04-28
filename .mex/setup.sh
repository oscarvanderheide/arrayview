#!/usr/bin/env bash
set -euo pipefail

if ! command -v mex >/dev/null 2>&1; then
  echo "mex CLI not found. Install it first, or run via 'npx promexeus'." >&2
  exit 1
fi

if [[ ! -f ".mex/ROUTER.md" || $# -gt 0 ]]; then
  mex setup "$@"
else
  echo "Scaffold already present; skipping 'mex setup'."
fi

echo "Installing post-commit drift hook..."
mex watch

echo
echo "Scaffold automation ready."
echo "Quick check: mex check --quiet"
echo "Resync:      .mex/sync.sh"
