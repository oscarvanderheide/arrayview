"""Where a VS Code open spent its time, from this host's own logs.

For each launch the opener handled, two waits that code changes were aimed at
(plans/tunnel/LOG.md, 2026-11-10):

- click -> tab: from the Explorer click to VS Code being asked to open the tab
  (Explorer clicks only; the opener's own handling).
- asked -> image: from the viewer's first frame request to the server sending
  that frame (needs the launch trace: `touch ~/.arrayview/enable-launch-trace`
  before the server starts).

Reads logs only; opens nothing.

    uv run python scripts/first_image_wait.py 2026-11-10        # that day
    uv run python scripts/first_image_wait.py 2026-11-10 -v     # every launch
"""

from __future__ import annotations

import json
import re
import statistics
import sys
from pathlib import Path

HOME = Path.home() / ".arrayview"
LINE = re.compile(r"^\[(\d{4}-\d\d-\d\dT[\d:.]+)Z\] (?:\[([0-9a-f]+)\] )?(.*)")
SIGNAL = re.compile(r"SIGNAL: requestId=(\w+) url=\S+(?: title=(.*))?")


def _seconds(stamp: str) -> float:
    hours, minutes, seconds = stamp[11:].split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def _summary(label: str, values: list[float]) -> None:
    if not values:
        print(f"{label}: none")
        return
    values = sorted(values)
    print(
        f"{label}: {len(values)} launches, median {statistics.median(values):.0f} ms, "
        f"90 % under {values[int(0.9 * (len(values) - 1))]:.0f} ms, "
        f"worst {values[-1]:.0f} ms"
    )


def main(argv: list[str]) -> int:
    verbose = "-v" in argv
    prefixes = [arg for arg in argv if arg != "-v"]

    launches: dict[str, dict] = {}
    click_at: dict[str | None, float] = {}
    current: dict[str | None, str] = {}
    for line in (HOME / "extension.log").open(errors="replace"):
        match = LINE.match(line)
        if not match:
            continue
        stamp, window, message = match.groups()
        if prefixes and not any(stamp.startswith(prefix) for prefix in prefixes):
            continue
        if "CUSTOM-EDITOR: resolveCustomEditor for" in message:
            click_at[window] = _seconds(stamp)
            continue
        signal = SIGNAL.search(message)
        if signal and signal.group(1) not in launches:
            launches[signal.group(1)] = {
                "stamp": stamp[:23],
                "title": signal.group(2) or "",
                "click": click_at.pop(window, None),
                "to_tab": None,
                "image": None,
            }
            current[window] = signal.group(1)
            continue
        launch = launches.get(current.get(window, ""))
        if (
            launch is not None
            and launch["to_tab"] is None
            and launch["click"] is not None
            and "openInIntegratedBrowser(" in message
        ):
            launch["to_tab"] = 1000 * (_seconds(stamp) - launch["click"])

    trace = HOME / "launch-trace.jsonl"
    asked: dict[str, int] = {}
    if trace.exists():
        for line in trace.open(errors="replace"):
            if '"viewer.first_' not in line:
                continue
            event = json.loads(line)
            request_id = (event.get("attrs") or {}).get("request_id")
            launch = launches.get(request_id)
            if launch is None or launch["image"] is not None:
                continue
            if event["event"] == "viewer.first_request":
                asked.setdefault(request_id, event["wall_ns"])
            elif request_id in asked:
                launch["image"] = (event["wall_ns"] - asked[request_id]) / 1e6

    _summary(
        "click -> tab   ",
        [v["to_tab"] for v in launches.values() if v["to_tab"] is not None],
    )
    _summary(
        "asked -> image ",
        [v["image"] for v in launches.values() if v["image"] is not None],
    )
    if verbose:
        for launch in launches.values():
            to_tab = "      -" if launch["to_tab"] is None else f"{launch['to_tab']:5.0f} ms"
            image = "      -" if launch["image"] is None else f"{launch['image']:5.0f} ms"
            print(f"  {launch['stamp']}Z  click->tab {to_tab}  asked->image {image}  {launch['title']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
