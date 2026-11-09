"""How often a VS Code tab opened blank, from ~/.arrayview/extension.log.

Reads the opener's own log, so it reports real use on this host and opens
nothing. Launches are grouped by how long their viewer port had sat unused,
because that is what the failure depends on (see plans/tunnel/LOG.md,
2026-11-09).

    uv run python scripts/blank_tab_rate.py              # everything in the log
    uv run python scripts/blank_tab_rate.py 2026-11-10   # that day
    uv run python scripts/blank_tab_rate.py 2026-11      # that month
    uv run python scripts/blank_tab_rate.py 2026-11-10 -v   # list the bad ones
"""

from __future__ import annotations

import collections
import datetime
import re
import sys
from pathlib import Path

LOG = Path.home() / ".arrayview" / "extension.log"
LINE = re.compile(r"^\[(\d{4}-\d\d-\d\dT[\d:.]+)Z\] (?:\[([0-9a-f]+)\] )?(.*)")
SIGNAL = re.compile(r"SIGNAL: requestId=(\w+) url=http://localhost:(\d+)/")
BUCKETS = ("first use", "< 20 s", "20-60 s", "60-180 s", "180-600 s", "> 600 s")


def _bucket(idle: float | None) -> str:
    if idle is None:
        return BUCKETS[0]
    for label, limit in zip(BUCKETS[1:], (20, 60, 180, 600)):
        if idle < limit:
            return label
    return BUCKETS[-1]


def main(argv: list[str]) -> int:
    verbose = "-v" in argv
    prefixes = [arg for arg in argv if arg != "-v"]
    last_frame_on_port: dict[int, float] = {}
    current: dict[str | None, str] = {}
    launches: dict[str, dict] = {}

    for line in LOG.open(errors="replace"):
        match = LINE.match(line)
        if not match:
            continue
        stamp, window, message = match.groups()
        if prefixes and not any(stamp.startswith(prefix) for prefix in prefixes):
            continue
        when = datetime.datetime.fromisoformat(stamp).timestamp()
        signal = SIGNAL.search(message)
        if signal:
            request_id, port = signal.group(1), int(signal.group(2))
            if request_id not in launches:
                previous = last_frame_on_port.get(port)
                launches[request_id] = {
                    "stamp": stamp[:19],
                    "started": when,
                    "port": port,
                    "idle": None if previous is None else when - previous,
                    "blank": 0,
                    "fresh_tab": False,
                    "rendered": False,
                    "integrated": False,
                    "took": None,
                }
            current[window] = request_id
            continue
        launch = launches.get(current.get(window, ""))
        if launch is None:
            continue
        if "openInIntegratedBrowser(" in message:
            launch["integrated"] = True
        elif "blank tab at attempt" in message:
            launch["blank"] += 1
        elif "opening this array in a fresh tab" in message:
            launch["fresh_tab"] = True
        elif "viewer-phase frame-rendered" in message and not launch["rendered"]:
            launch["rendered"] = True
            launch["took"] = when - launch["started"]
            last_frame_on_port[launch["port"]] = when

    totals = collections.OrderedDict((label, [0, 0]) for label in BUCKETS)
    bad_launches = []
    for launch in launches.values():
        if not launch["integrated"]:
            continue
        label = _bucket(launch["idle"])
        totals[label][0] += 1
        if launch["blank"] or launch["fresh_tab"] or not launch["rendered"]:
            totals[label][1] += 1
            bad_launches.append(launch)

    print("viewer port unused for   launches   blank / retried / failed")
    for label, (count, bad) in totals.items():
        share = f"{100 * bad / count:3.0f} %" if count else "   -"
        print(f"{label:>22}   {count:8d}   {bad:5d}  {share}")
    if verbose:
        for launch in bad_launches:
            idle = "first use" if launch["idle"] is None else f"idle {launch['idle']:.0f} s"
            outcome = f"frame after {launch['took']:.1f} s" if launch["rendered"] else "no frame"
            print(
                f"  {launch['stamp']}Z port {launch['port']} {idle}: "
                f"{launch['blank']} blank"
                f"{', fresh tab' if launch['fresh_tab'] else ''}, {outcome}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
