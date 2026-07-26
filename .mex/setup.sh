#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/mex-bin.sh"
MEX="$(resolve_mex_bin)" || exit 1

if [[ ! -f ".mex/ROUTER.md" || $# -gt 0 ]]; then
  "$MEX" setup "$@"
else
  echo "Scaffold already present; skipping 'mex setup'."
fi

echo "Installing post-commit drift hook..."
"$MEX" watch

echo
echo "Scaffold automation ready."
echo "Quick check: .mex/check.sh"
echo "Resync:      .mex/sync.sh"
