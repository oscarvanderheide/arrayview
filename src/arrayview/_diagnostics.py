"""Turn a launch failure into a concise, actionable terminal message.

One line saying what went wrong, one line saying what to do about it. The
severity colour is the signal that matters:

    orange  something you can clear yourself right now — reload the window,
            run it again, kill a stale server
    red     something that needs a real change — the file, the arrays, or a
            bug worth a traceback

`SETUP` is the exception to all of that: the run before it *worked*, and the
only thing left is a step the user has to take by hand. Nothing went wrong, so
it is not written as though something did — no "failed to open" headline, no
traceback hint, and none of the raw internal text.

The patterns below are matched against the message text the opener and the
backend actually produce; they are taken from observed failures rather than
invented, so a new failure string falls through to the red catch-all with a
`--trace` hint rather than being mislabelled as harmless.
"""

from __future__ import annotations

import os
import re
import sys

ACTION = "action"   # orange — user can clear it
FATAL = "fatal"     # red — needs a real fix
SETUP = "setup"     # orange — nothing failed; one manual step is outstanding

_RESET = "\033[0m"
_RED = "\033[1;31m"
_ORANGE = "\033[1;38;5;208m"
_DIM = "\033[2m"

_RELOAD = 'Reload the VS Code window (Ctrl/Cmd+Shift+P → "Developer: Reload Window").'
_RETRY = "Run the same command again."

# (pattern, severity, what happened, what to do)
_DIAGNOSES: list[tuple[str, str, str, str]] = [
    # ── First run after the opener is installed — not a failure ───────────
    # Must stay ahead of the stale-opener pattern below: the local wording
    # ("reload this VS Code window once, then retry") matches both, and a
    # just-installed opener is not a stale one.
    (r"opener was installed|installed its vs ?code opener|"
     r"updated its vs ?code opener", SETUP,
     "The VS Code opener is installed.",
     'Reload this VS Code window once (Ctrl/Cmd+Shift+P → "Developer: Reload '
     'Window"), then run the same command again.'),

    # ── Opener/extension state — all clearable ────────────────────────────
    (r"stale arrayview opener|reload this vs ?code window", ACTION,
     "This VS Code window is running an older ArrayView opener than the one "
     "installed.",
     _RELOAD),
    (r"another arrayview backend now owns this port", ACTION,
     "A newer launch took over the port, so the session this tab wanted is "
     "gone.",
     _RETRY),
    (r"required extension|extension version|opener v\d", ACTION,
     "The installed opener and the Python package disagree on version.",
     _RELOAD + "  If it persists: uv tool install --reinstall arrayview"),

    # ── Transport — the route, not the data ───────────────────────────────
    (r"backend stopped answering", ACTION,
     "The route to the backend stopped answering before the viewer was ready "
     "(usually a cold tunnel, not a dead server).",
     _RETRY + "  If it repeats, reload the VS Code window."),
    (r"could not connect to the arrayview backend|ws-open", ACTION,
     "The viewer loaded but its WebSocket never connected.",
     _RELOAD + "  If it persists, the port is blocked or the tunnel is down."),
    (r"viewer route stopped answering|route .*not answering", ACTION,
     "The forwarded route stopped answering; the backend itself is likely "
     "fine.",
     _RETRY),

    # ── A window that is alive but not answering ──────────────────────────
    # Its process exists, so nothing treats it as dead and no other window may
    # take the request from it — targeted requests are invisible to everyone
    # else. The user sees nothing happen at all, and reloading that one window
    # is the only recovery, so say so rather than suggesting a plain retry:
    # retrying sends the next launch to the same stuck window.
    (r"integrated browser open timeout|command discovery timeout|"
     r"browser command did not return", ACTION,
     "The VS Code window handling this request has stopped responding. It is "
     "still running, so nothing else will take the request from it.",
     _RELOAD + "  Retrying without reloading sends the next launch to the "
     "same window."),
    (r"claimed but never|no progress after being claimed", ACTION,
     "A VS Code window took this request and then went quiet, so the launch is "
     "stuck with it.",
     _RELOAD),

    # ── Timing — slow, not broken ─────────────────────────────────────────
    (r"did not start the viewer script", ACTION,
     "VS Code opened an integrated-browser tab, but the tab did not navigate "
     "to the ArrayView page.",
     _RETRY + "  If it repeats, reload the VS Code window."),
    (r"did not render a frame|did not become ready",
     ACTION,
     "The viewer opened but showed no frame before the deadline — usually a "
     "slow first load.",
     _RETRY + "  If the same file always does this, it may be too large to "
     "render in one go."),
    (r"panel closed before|closed before its first frame", ACTION,
     "The viewer tab was closed (or replaced by a preview tab) before it "
     "rendered.",
     _RETRY + "  Leave the tab open while it loads."),
    (r"signal expired|expired before a panel|hard timeout after", ACTION,
     "The opener was still busy with an earlier request and this one aged "
     "out.",
     _RETRY),

    # ── Ports and stale processes ─────────────────────────────────────────
    (r"could not find a free port|no free arrayview port|address already in use",
     ACTION,
     "No free port was available — earlier ArrayView servers are probably "
     "still running.",
     "See them with `arrayview instances`, clear them with `arrayview stop`, "
     "then retry."),
    (r"could not register the session|registration", ACTION,
     "The server started but the session could not be registered.",
     _RETRY + "  If it repeats: arrayview stop, then try again."),

    # ── The data itself — a real change is needed ─────────────────────────
    (r"pickled \(object\) data|allow_pickle", FATAL,
     "This .npy holds pickled Python objects, not a plain array.",
     "Load it yourself and pass the array, or re-save with "
     "np.save(..., arr) on a numeric array."),
    (r"could not open this file|no data left in file|failed to load|"
     r"cannot load|unsupported file", FATAL,
     "The file could not be read as an array.",
     "Check the path and that the file is complete and not truncated."),
    (r"dtype mismatch", FATAL,
     "The arrays you passed have different dtypes.",
     "Compare needs matching dtypes — convert one before loading."),
    (r"shape mismatch|expected \(|spatial shape|shapes? .*differ", FATAL,
     "The arrays you passed have incompatible shapes.",
     "Compare needs matching spatial shapes — check both files."),
    (r"cannot start server", FATAL,
     "The backend server could not start at all.",
     "Re-run with --trace for the underlying error."),

    # ── Bad arguments, caught before anything starts ──────────────────────
    (r"file not found", FATAL,
     "That file does not exist.",
     "Check the path — and that the share it lives on is mounted."),
    (r"not a directory", FATAL,
     "That path is not a directory.",
     "Point --stack at a directory of arrays."),
    (r"invalid vector ?field", FATAL,
     "The vector field file could not be used with this array.",
     "It needs the array's spatial shape plus a trailing components axis."),
    (r"could not discover masks|could not match collection", FATAL,
     "The pattern matched no files.",
     "Check the pattern and run with --dry-run to see what it resolves to."),
    (r"invalid launch request", FATAL,
     "The launch arguments did not form a valid request.",
     "Re-run with --diagnose to see how the launch was planned."),
]

_COMPILED = [(re.compile(p, re.I), sev, what, fix) for p, sev, what, fix in _DIAGNOSES]


def _colour_enabled(stream) -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("ARRAYVIEW_FORCE_COLOR"):
        return True
    return bool(getattr(stream, "isatty", lambda: False)())


def diagnose(message: str) -> tuple[str, str, str]:
    """Classify a failure message → (severity, what happened, what to do)."""
    text = (message or "").strip()
    for pattern, severity, what, fix in _COMPILED:
        if pattern.search(text):
            return severity, what, fix
    return (
        FATAL,
        "ArrayView failed to open the display.",
        "Re-run with --trace (or ARRAYVIEW_TRACE=1) for the full traceback.",
    )


def format_failure(message: str, *, colour: bool = True) -> str:
    """Render the diagnosis as the block printed to stderr."""
    severity, what, fix = diagnose(message)
    detail = (message or "").strip().removeprefix("[ArrayView] ").rstrip()
    head = (_RED if severity == FATAL else _ORANGE) if colour else ""
    dim = _DIM if colour else ""
    end = _RESET if colour else ""
    lines = [
        f"{head}[ArrayView] {what}{end}",
        f"{head}  → {fix}{end}",
    ]
    # A pending setup step is not a failure report: the instruction is the
    # whole message. Echoing the internal wording underneath it is what made
    # "the opener is installed, reload once" read like something broke.
    if severity == SETUP:
        return "\n".join(lines)
    # Keep the raw text when it says more than the headline does.
    if detail and detail.lower() not in what.lower():
        lines.append(f"{dim}  ({detail}){end}")
    return "\n".join(lines)


def print_failure(message: str, stream=None) -> None:
    stream = stream if stream is not None else sys.stderr
    print(format_failure(message, colour=_colour_enabled(stream)), file=stream)


def format_notice(message: str, *, colour: bool = True) -> str:
    """Render a one-line ACTION notice.

    Same orange as an ACTION failure, because it carries the same meaning —
    something the user asked for is not happening — but the launch continues,
    so there is no cause/fix block to print.
    """
    head = _ORANGE if colour else ""
    end = _RESET if colour else ""
    return f"{head}[ArrayView] {message}{end}"


def print_notice(message: str, stream=None) -> None:
    stream = stream if stream is not None else sys.stdout
    print(
        format_notice(message, colour=_colour_enabled(stream)),
        file=stream,
        flush=True,
    )
