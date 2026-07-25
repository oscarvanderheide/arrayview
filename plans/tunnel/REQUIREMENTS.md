# ArrayView in VS Code tunnel windows — requirements

Status: drafted 2026-07-25 from the maintainer's description. This is the
definition of "works" that the tunnel repair effort is measured against.
Everything here is user-observable; none of it is about internals.

## Setting

- Arrays live on the Linux host (`roodnoot`). The ArrayView server runs there.
- The maintainer views them on a macOS laptop through a VS Code tunnel window.
- Storage tiers differ by a large factor:
  - `/localscratch/...` — local disk, fast
  - `$HOME` — network drive, slower
  - `/smb/user/...` — bulk storage, slowest
- Several VS Code windows are typically open at once: local and tunnel,
  attached to different repositories.

## Requirements

R1. **Port forwarding is automatic.** Port 8000 reaches public visibility
without the user opening the Ports panel or changing privacy by hand.

R2. **Both entry points work equally.** Running `arrayview <file>` in a VS Code
terminal, and clicking an array in the Explorer, both open a working viewer.

R3. **The viewer opens in the window it was requested from.** Never a different
window, never an arbitrary one. *(Believed working today — protect it.)*

R4. **Windows do not interfere.** Multiple VS Code windows — tunnel and local,
different repositories — may be open simultaneously. One window's request must
never be claimed, blocked, expired, or cleaned up by another window.

R5. **Several arrays open at once,** each in its own tab, regardless of which
entry point opened it.

R6. **It keeps working indefinitely.** Open three, close two, open two more,
close everything, open again — repeatedly, in any order. No degradation with
repetition. No state that accumulates until it jams.

R7. **A failure never wedges the window.** If one request fails, the next
request starts clean. Recovery is bounded in seconds, not minutes. No queued
request may die waiting behind a failed one.

R8. **Slow storage is not failure.** An array on bulk storage may take a long
time to load. Waiting is visibly waiting (spinner/progress). Readiness deadlines
must not fire because the disk is slow, and a slow load must not block anything
else.

R9. **Success means a rendered frame.** Not "a tab opened", not "the port
answered", not "a WebSocket connected". The requested array is visible.

R10. **Failure is visible and actionable.** A clear message in the terminal or
the tab. Silent nothing is the worst possible outcome and is the current
behaviour.

R11. **Version skew is impossible to miss.** A window running an older extension
build than its peers must say so, rather than failing intermittently.

## Acceptance matrix

Each row must pass in a real tunnel window. `PASS` requires a visible first
frame (R9), not a process or socket event.

| # | Scenario | Entry point | Status |
|---|----------|-------------|--------|
| T1 | Single array, cold start (no daemon) | Explorer click | **PASS** 2026-07-26 |
| T2 | Single array, cold start | terminal `arrayview` | not yet run |
| T3 | Second array while first is open | Explorer click | **PASS** 2026-07-26 |
| T4 | Third and fourth array, mixed entry points | both | **PASS** (Explorer only) |
| T5 | Close all tabs, then open again | both | **PASS** (Explorer only) |
| T6 | Repeat T1–T5 a second and third time | both | **PASS** (Explorer only) |
| T7 | Two tunnel windows, one request each | both | not yet run |
| T8 | Request while another window is mid-launch | both | not yet run |
| T9 | Array on `/smb` bulk storage | both | partial — small files pass; a genuinely slow load still blocks others (see R8 gap) |
| T10 | Deliberately failed request, then a good one | both | **PASS** 2026-07-26 |
| T11 | Mixed extension versions across windows warns | n/a | implemented, not yet triggered in anger |
| T12 | Released `uvx arrayview` after the fix | terminal | not yet run |

Open gaps, in the order they should be closed:

1. **R8 — a slow load still blocks other arrays.** The readiness wait holds the
   global `isProcessingSignal` lock, so one slow array stalls every request
   behind it. Only the *failure* case is fast. Fixing this means making the lock
   per-request instead of global.
2. **T2 — the terminal entry point** (`arrayview <file>`) has not been retested
   since the fixes; only Explorer clicks have.
3. **T7/T8 — multi-window behaviour** has not been retested since the fixes.
4. **T12 — a release**, after which `uvx arrayview` carries all of this.

## Non-goals for this effort

- Changing the viewer UI.
- Local (non-remote) VS Code behaviour beyond not regressing it.
- Jupyter, Julia, MATLAB, and plain-SSH paths beyond not regressing them.
