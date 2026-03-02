#!/usr/bin/env bash
# Kill stale arrayview server processes.
# Works on macOS and Linux.

set -euo pipefail

killed=0

while IFS= read -r line; do
    pid=$(echo "$line" | awk '{print $1}')
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        echo "Killing PID $pid: $line"
        kill "$pid"
        killed=$((killed + 1))
    fi
done < <(ps aux 2>/dev/null | grep -E '[a]rrayview|[_]app\.py' | grep -v grep | awk '{print $2, $0}')

if [ "$killed" -eq 0 ]; then
    echo "No stale arrayview processes found."
else
    echo "Killed $killed process(es)."
fi
