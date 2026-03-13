# Tunnel Fix Log

## Attempt 1: Initial analysis

**Date**: 2026-03-13
**Hypothesis**: Adding `_configure_vscode_port_preview(parsed_port)` to the 
remote `_open_browser` path is the primary cause. This function was NEVER called
in the remote path before (only local VS Code + CLI new-server path). Writing
VS Code settings files during every `_open_browser` call in a tunnel session
could corrupt settings or confuse VS Code's port forwarding.

Secondary suspect: port default change from 8000 → 8123 means old Public port
config no longer matches.

**Change**: 
1. Revert the `_configure_vscode_port_preview(parsed_port)` call from inside 
   the `is_remote` branch of `_open_browser._do()`.
2. Keep it in the `--serve` path (runs once, pre-server, appropriate).
3. Update `parsed_port` fallback from 8000 to 8123.

**Result**: NOT sufficient. User still got connection refused. Symptom: server reported
"opening port 8321" but VS Code Ports tab showed 8000 (stale from before our changes).
Setting 8000 to Public had no effect because the server was on 8321. Manually changing
the URL to port 8000 also failed for the same reason.

**Root cause of Attempt 1 failure**: Auto-port scanning (Fix 3A) is fundamentally
incompatible with the tunnel workflow. When the default port is busy, the scan picks
an unpredictable port (e.g. 8321). The user's Ports tab still shows the stale old port
(8000 or 8123). The user can't know which port to set to Public, and VS Code may not
auto-forward the new port correctly even with `_configure_vscode_port_preview`.

**Decision**: Keep the revert (no `_configure_vscode_port_preview` in `_open_browser`
remote path). Proceed to Attempt 2.

---

## Attempt 2: Disable auto-scan in tunnel mode + fix message + fix timing

**Date**: 2026-03-13

**Hypothesis**: Three compounding problems:
1. Auto-scan picks an unpredictable port → user doesn't know which port to set Public
2. The message says "ensure port is Public" without specifying WHICH port → user sets the
   wrong (stale) port
3. `_configure_vscode_port_preview` in `--serve` was called AFTER the server started →
   VS Code may have already forwarded the port at default (Private) privacy before
   the settings file was written

**Changes**:
1. In tunnel mode (`_is_vscode_remote()`), disable auto-scan in both `--serve` path
   and CLI new-server path. If the port is busy with a non-ArrayView process, print a
   clear error: "Port X is in use. Run 'arrayview --kill --port X' to free it, or
   use --port to specify a different port." Exit with code 1.
2. Changed the remote message in `_open_browser._do()` to include the exact port:
   "[ArrayView] Remote tunnel session on port PORT.\n  VS Code Ports tab: right-click
   port PORT → Port Visibility → Public."
3. Moved `_configure_vscode_port_preview(args.port)` in the `--serve` path to BEFORE
   `subprocess.Popen(...)` starts the server. This gives VS Code a chance to read the
   settings before the port starts listening, so it can auto-forward as Public.

**What da69007 had**: `_configure_vscode_port_preview` was imported but NEVER called.
The user had to manually set the port to Public. Our additions call it to try to
automate that step, but timing matters.

**Result**: Still broken. The message showed "port 8123" correctly but VS Code Ports tab
showed port 8000 as "Auto Forwarded / Public". Port 8123 was Private → ERR_CONNECTION_REFUSED.

**Root cause confirmed**: The CLI default port changed from 8000 → 8123.
The user's VS Code already had port 8000 configured as Public (from settings.json written
by `_configure_vscode_port_preview` in a prior run). The new server started on 8123 (different
port), which VS Code auto-forwarded as Private. `_configure_vscode_port_preview(8123)` was
called before the server started, but VS Code must have already read settings with port 8000 as
Public and not immediately re-read for 8123, or the settings write race was lost.

**Decision**: Revert CLI default port from 8123 → 8000. The user's Ports tab already has 8000
as Public. The `view()` default stays 8123 (Python API, unaffected). All other improvements
from Attempt 2 (better message, always-write signal file, blocking=force_vscode) are kept.

---

## Attempt 3: Revert CLI default port to 8000

**Date**: 2026-03-13

**Change**: `--port default=8000` in the CLI argparser; `parsed_port` fallback in `_open_browser`
also reverted to 8000. The `--kill` help text updated to say 8000.

**Result**: (pending — needs user to test on tunnel)

---

## Attempt 4 – 8 (relay feature work, 2026-03-13)

Multiple changes were made to add the relay feature (gpu-server → tunnel-remote).
In the process, the `@app.get("/")` decorator was accidentally dropped from `get_ui()`
in `_server.py`, making the viewer return 404 for all sessions. Fixed by restoring
the decorator. `_schedule_remote_open_retries` was also incorrectly added to
`/load_bytes` causing multiple Simple Browser tabs to open.

After those fixes, the user reports: normal tunnel (`arrayview test_data.npy` without
GPU server) shows the message but nothing opens — "nothing happens".

**Status**: CLOSED. Relay implementation caused two bugs:
1. `@app.get("/")` decorator dropped from `get_ui()` in `_server.py` → 404 on all sessions (fixed)
2. `_schedule_remote_open_retries` erroneously added to `/load_bytes` → multiple browser tabs (removed)
After these fixes, the normal tunnel message prints correctly but nothing opens.

---

## Attempt 9: Diagnostic — extension log inspection

**Date**: 2026-03-13
**Hypothesis**: The VS Code extension is either not installed, not running (no active extension host),
or encountering a silent error calling `simpleBrowser.show`. The signal file IS written
(message "Remote tunnel session on port …" proves we reach the correct branch and write the file).
The blocking question is: does the extension see the signal?

**Change**: None — pure diagnostic. No code changed.

**Regression risk**: None.

**Test procedure**:
```
# Kill old server first
uv run arrayview --kill

# Re-start persistent server
uv run arrayview --serve

# Load an array (triggers signal file write)
uv run arrayview test_data.npy

# Then immediately in a second terminal:
cat ~/.arrayview/extension.log | tail -40
ls ~/.vscode-server/extensions/ | grep arrayview
# Also useful:
cat ~/.arrayview/open-request-v0900.json 2>/dev/null && echo "signal file exists" || echo "signal file already consumed or missing"
```

**What to look for**:
- If extension.log shows `SIGNAL: requestId=… url=…` → extension IS running and processed it. Then the issue is what happens next (asExternalUri, simpleBrowser.show).
- If extension.log shows `=== ACTIVATE ===` but no `SIGNAL:` line after the arrayview run → extension running but not seeing the file (wrong path / filename mismatch).
- If extension.log has no recent entries or doesn't exist → extension NOT running. Must check install.
- If `ls ~/.vscode-server/extensions/ | grep arrayview` shows nothing → extension never installed.

**Result**: SUCCESS. Normal tunnel flow is working end-to-end.
Extension log shows: SIGNAL processed → asExternalUri → simpleBrowser.show done.
The ENOENT error is a harmless race: Python deletes old signal files before writing
new ones; the extension occasionally tries to unlink a file already gone by Python.
It logs the error but continues and processes the signal correctly.

### Current path status after Attempt 9
- ✅ Normal tunnel: `arrayview --serve` + `arrayview file.npy` → Simple Browser opens
- ❓ Relay mode: `arrayview file.npy --relay PORT` (GPU server → reverse SSH) — untested

---

## Attempt 10: Relay mode end-to-end test (GPU server → reverse SSH → tunnel-remote)

**Date**: 2026-03-13
**Hypothesis**: Relay mode works end-to-end given correct setup.
**Change**: None — pure test.
**Regression risk**: None.

**Test procedure**:
1. Tunnel-remote: `uv run arrayview --serve` (port 8000, set to Public)
2. SSH from tunnel-remote: `ssh -R 8765:localhost:8000 oheide@<gpu-host>`
3. On GPU server: `av <file> --relay 8765`

**Result**: SUCCESS. Relay mode works end-to-end.
- GPU server sends array bytes to `localhost:8765` → reverse SSH tunnel →
  `localhost:8000/load_bytes` on tunnel-remote → session created → signal file written →
  VS Code extension opens Simple Browser with devtunnel URL including correct `?sid=`.

### Final path status
- ✅ Normal tunnel: `arrayview --serve` + `arrayview file.npy` → Simple Browser opens
- ✅ Relay mode: on GPU server `av file.npy --relay 8765` → Simple Browser opens on local VS Code

### Remaining UX friction (next tasks)
1. Tunnel: user has to run `--serve` separately before `arrayview file.npy`. Could be combined.
2. Relay: user has to specify `--relay 8765` manually. Auto-detect is almost already implemented
   (see `_launcher.py` hostname check) — works if user SSHes with same port on both sides
   (`ssh -R 8000:localhost:8000`) so `_server_alive(8000)` returns True and hostname differs.

---

## Attempt 11: Auto-relay detection fix (SSH tunnel HTTP timeout)

**Date**: 2026-03-13
**Hypothesis**: `_server_alive(args.port)` uses a 0.5s HTTP timeout. When port 8000 is bound
by a reverse SSH tunnel, the full HTTP round-trip (TCP connect through SSH mux + GET + response)
exceeds 0.5s even on a LAN. This causes `_server_alive(8000)` to return False, the code auto-scans
to port 8001, and relay detection never fires (`is_arrayview_server` is False before the hostname check).

**Change**: `_launcher.py` only:
1. Added `timeout: float = 0.5` param to `_server_alive` and `_server_hostname`.
2. Moved `_is_ssh` detection before the port scan block.
3. Added retry: if SSH + port in use + fast check failed → retry `_server_alive(port, timeout=3.0)`.
4. Used 3s timeout for `_server_hostname` in relay detection (same tunnel latency applies).

**Regression risk**: None for non-SSH paths (default timeout unchanged). In SSH mode with a
non-ArrayView process on port 8000: max 3s extra wait before auto-scanning. Acceptable.

**Test procedure (for relay auto-detect)**:
```bash
# On tunnel-remote (if not already running):
uv run arrayview --serve

# SSH from tunnel-remote to GPU server WITH SAME PORT on both sides:
ssh -R 8000:localhost:8000 oheide@rtrspla06

# On GPU server — NO --relay flag needed:
av sense_images.npy
```
Expected: arrayview detects the reverse tunnel automatically, relays the array, Simple Browser opens.

**Result**: (pending — awaiting user test)

---

## Attempt 12: Enter-prompt UX for tunnel without --serve

**Date**: 2026-03-13
**Hypothesis**: When `arrayview file.npy` starts a fresh server in tunnel mode, the signal file
is written immediately (port still Private), Simple Browser opens but hits auth challenge. The
7-second retry lottery is unreliable UX. Better: prompt user to set port Public, then press Enter,
then write signal file. Port is definitely Public by then.

**Change**: `_launcher.py` only: in the new-server remote path (after `_wait_for_port` succeeds,
before `_open_browser`), if `is_remote and sys.stdin.isatty()`: print Enter prompt, wait, then
set `_vscode._remote_message_shown = True` to suppress duplicate message in `_open_browser`.

**Regression risk**: None — only fires when `is_remote=True` AND stdin is a TTY AND this is the
new-server path (not the existing-server path). The retry loop in `_open_browser` still runs as
a safety net after the user presses Enter.

**Test procedure (for Enter prompt)**:
```bash
# Kill any running server first
uv run arrayview --kill

# Run directly with a file (no --serve first)
uv run arrayview test_data.npy
```
Expected: server starts, prompt appears "right-click port 8000 → Public, press Enter",
after Enter Simple Browser opens immediately.

**Result**: (pending — awaiting user test)
