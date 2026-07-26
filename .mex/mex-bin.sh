#!/usr/bin/env bash
# Resolve the scaffold CLI, refusing impostors.
#
# TeX Live's texlive-lang-polish package installs /usr/bin/mex — a symlink to
# pdftex.  A plain `command -v mex` finds it, so the obvious guard passes and
# `mex check` then hands "check" to a TeX engine, which fails obscurely and
# drops a texput.log in the repo.  Probe what the binary actually is.
#
# Override with MEX_BIN if the real CLI lives somewhere unusual.

resolve_mex_bin() {
  local candidate="${MEX_BIN:-mex}"

  if ! command -v "$candidate" >/dev/null 2>&1; then
    echo "mex CLI not found. Install it, run via 'npx promexeus', or set MEX_BIN." >&2
    return 1
  fi

  if "$candidate" --version 2>&1 | grep -qiE 'pdftex|tex live|kpathsea'; then
    echo "'$(command -v "$candidate")' is TeX's mex (pdftex), not the scaffold CLI." >&2
    echo "Install promexeus and set MEX_BIN, or use 'npx promexeus'." >&2
    return 1
  fi

  printf '%s\n' "$candidate"
}
