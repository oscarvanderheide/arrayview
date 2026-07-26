#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/mex-bin.sh"
MEX="$(resolve_mex_bin)" || exit 1

exec "$MEX" check "${@:---quiet}"
