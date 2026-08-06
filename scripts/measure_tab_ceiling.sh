#!/usr/bin/env bash
# Does the number of open viewer tabs have a ceiling, and where?
#
#   scripts/measure_tab_ceiling.sh [max-opens] [gap-seconds] [array]
#
# Opens arrays back to back with a short gap, so the idle-path failure is out of
# play (an open within ~10 s of the previous one has never lost a page load).
# Anything that fails here is about tab count, not a stale path. Prints the tab
# count the opener saw for every launch, so a cliff is visible rather than
# inferred.
#
# Leaves every tab open by design — that is the variable. Close them afterwards.
set -uo pipefail

MAX="${1:-24}"
GAP="${2:-5}"
ARRAY="${3:-/localscratch/oheide/tmp/initial_PD_masked.npy}"
LOG="$HOME/.arrayview/extension.log"

[ -f "$ARRAY" ] || { echo "array not found: $ARRAY" >&2; exit 1; }
[ -f "$LOG" ] || { echo "opener log not found: $LOG" >&2; exit 1; }

echo "max=$MAX gap=${GAP}s array=$ARRAY"
echo
printf "%-6s %-10s %-10s %s\n" "open" "tabs" "swallowed" "result"

first_failure=0
for i in $(seq 1 "$MAX"); do
    before=$(wc -l < "$LOG")
    uv run arrayview "$ARRAY" >/dev/null 2>&1
    sleep 2
    new=$(tail -n +$((before + 1)) "$LOG")

    swallowed=$(grep -c "blank tab at attempt" <<<"$new")
    rendered=$(grep -c "frame-rendered" <<<"$new")
    tabs=$(grep -oE "browserTabsOpen=[0-9]+" <<<"$new" | tail -1 | cut -d= -f2)
    [ -z "$tabs" ] && tabs="$i"

    if [ "$rendered" -eq 0 ]; then
        printf "%-6s %-10s %-10s %s\n" "$i" "$tabs" "$swallowed" "NO VIEWER"
        [ "$first_failure" -eq 0 ] && first_failure="$i"
    else
        printf "%-6s %-10s %-10s %s\n" "$i" "$tabs" "$swallowed" "ok"
    fi
    sleep "$GAP"
done

echo
if [ "$first_failure" -eq 0 ]; then
    echo "no ceiling reached in $MAX opens"
else
    echo "first failure at open $first_failure"
fi
