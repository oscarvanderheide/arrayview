#!/usr/bin/env bash
# dev_run.sh — clean slate + run arrayview for debugging
set -e

ARRAY="${1:-small_array.npy}"
EXT_DIR="$HOME/.vscode-server/extensions"
SIGNAL_DIR="$HOME/.arrayview"
PORT=8000

echo "=== 1. Kill any arrayview server on port $PORT ==="
if lsof -ti tcp:$PORT &>/dev/null; then
    lsof -ti tcp:$PORT | xargs kill -9 && echo "Killed process on :$PORT"
else
    echo "Port $PORT is free"
fi

echo ""
echo "=== 2. Remove stale signal/cache files ==="
rm -f "$SIGNAL_DIR/open-request.json" "$SIGNAL_DIR/tunnel-url.json"
echo "Cleared ~/.arrayview/open-request.json and tunnel-url.json"

echo ""
echo "=== 3. Extension versions installed ==="
ls "$EXT_DIR" | grep arrayview || echo "(none)"

echo ""
echo "=== 4. Remove all but the newest arrayview extension version ==="
# Find newest by sort (version strings sort lexicographically for semver x.y.z)
NEWEST=$(ls "$EXT_DIR" | grep "^arrayview.arrayview-opener" | sort -V | tail -1)
if [[ -z "$NEWEST" ]]; then
    echo "No arrayview extension installed — installing now..."
    VSIX="$(dirname "$0")/src/arrayview/arrayview-opener.vsix"
    "$HOME/.vscode-server/bin/"*/bin/code-server --install-extension "$VSIX"
    NEWEST=$(ls "$EXT_DIR" | grep "^arrayview.arrayview-opener" | sort -V | tail -1)
fi

for dir in "$EXT_DIR"/arrayview.arrayview-opener-*; do
    name=$(basename "$dir")
    if [[ "$name" != "$NEWEST" ]]; then
        echo "Removing old: $name"
        rm -rf "$dir"
    fi
done
echo "Keeping: $NEWEST"

echo ""
echo "=== 5. Version string in installed extension.js ==="
EXT_JS="$EXT_DIR/$NEWEST/extension.js"
if [[ -f "$EXT_JS" ]]; then
    head -1 "$EXT_JS"
    grep "Extension activated" "$EXT_JS" || true
else
    echo "WARNING: extension.js not found at $EXT_JS"
fi

echo ""
echo "=== 6. Tail of extension log (last 10 lines) ==="
if [[ -f "$SIGNAL_DIR/extension.log" ]]; then
    tail -10 "$SIGNAL_DIR/extension.log"
else
    echo "(no log yet)"
fi

echo ""
echo "=== 7. Running: uv run arrayview $ARRAY ==="
echo "(watch ~/.arrayview/extension.log for extension activity)"
cd "$(dirname "$0")"
uv run arrayview "$ARRAY"
