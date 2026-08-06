#!/usr/bin/env bash
# Measure the row-2 failure: how many page loads are swallowed on the first
# open after the client path has been idle.
#
#   scripts/measure_idle_launch.sh <trials> <idle-seconds> [array]
#
# Each trial waits <idle-seconds> with a viewer already open, opens the array,
# and counts the blank-tab records the opener wrote for that launch. Prints one
# line per trial and a summary. Run the same command before and after a change;
# a single trial proves nothing, the drops come in runs.
set -uo pipefail

TRIALS="${1:-10}"
IDLE="${2:-100}"
ARRAY="${3:-/localscratch/oheide/tmp/initial_PD_masked.npy}"
LOG="$HOME/.arrayview/extension.log"

command -v uv >/dev/null || { echo "uv not found" >&2; exit 1; }
[ -f "$ARRAY" ] || { echo "array not found: $ARRAY" >&2; exit 1; }
[ -f "$LOG" ] || { echo "opener log not found: $LOG" >&2; exit 1; }

echo "trials=$TRIALS idle=${IDLE}s array=$ARRAY"
echo

swallowed_total=0
clean=0
failed=0

for i in $(seq 1 "$TRIALS"); do
    sleep "$IDLE"
    before=$(wc -l < "$LOG")
    uv run arrayview "$ARRAY" >/dev/null 2>&1
    rc=$?
    sleep 2
    new=$(tail -n +$((before + 1)) "$LOG")
    swallowed=$(grep -c "blank tab at attempt" <<<"$new")
    rendered=$(grep -c "frame-rendered" <<<"$new")

    if [ "$rendered" -eq 0 ]; then
        failed=$((failed + 1))
        printf "trial %2d: NO VIEWER (exit %d, %d swallowed)\n" "$i" "$rc" "$swallowed"
    else
        swallowed_total=$((swallowed_total + swallowed))
        [ "$swallowed" -eq 0 ] && clean=$((clean + 1))
        printf "trial %2d: %d swallowed%s\n" "$i" "$swallowed" \
            "$([ "$swallowed" -eq 0 ] && echo '  (clean, no flicker)')"
    fi
done

echo
echo "clean (no flicker): $clean/$TRIALS"
echo "failed (no viewer): $failed/$TRIALS"
echo "swallowed loads, total across rendered trials: $swallowed_total"
