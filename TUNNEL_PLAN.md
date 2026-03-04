# ArrayView — Tunnel Support Plan

## Goal
Make `arrayview` auto-open in VS Code's **Simple Browser** tab when running over a
VS Code **Dev Tunnel** connection (remote Linux ← macOS/Windows laptop client).
No new windows, no manual URL clicking.

---

## Architecture

```
 LAPTOP (VS Code client)                REMOTE MACHINE (tunnel server)
 ──────────────────────                  ─────────────────────────────
 VS Code UI ◄─── Dev Tunnel ───►  code-tunnel daemon
                                         │
                                         ├─ Extension Host (runs workspace extensions)
                                         │    └─ arrayview-opener v0.2.0
                                         │         ├─ fs.watch(~/.arrayview/)
                                         │         ├─ 2s polling fallback
                                         │         └─ IS_REMOTE_SERVER targeting
                                         │
                                         └─ Terminal
                                              └─ uv run arrayview file.npy
                                                   │
                                                   ├─ Start uvicorn on localhost:PORT
                                                   ├─ _ensure_vscode_extension()
                                                   └─ _open_via_signal_file(url)
                                                        │
                                                        └─ Write ~/.arrayview/open-request.json
                                                             { "url": "...", "remote": true }
                                                             │
                RESULT:                                      ▼
                SimpleBrowser tab ◄─── auto-forward ◄── Extension reads signal file
                shows arrayview    (SimpleBrowser auto-      │
                                    forwards localhost)      └─ simpleBrowser.show(localhostUrl)
```

**Key insight:** `simpleBrowser.show(localhostUrl)` auto-forwards ports through
tunnels — no need for `asExternalUri`, `openTunnel`, or devtunnel URL construction.

## Components

### 1. VS Code Extension: `arrayview-opener` v0.2.0

- **Location**: `vscode-extension/extension.js` + `vscode-extension/package.json`
- **Bundled VSIX**: `src/arrayview/arrayview-opener.vsix`
- **Installed to**: `~/.vscode-server/extensions/arrayview.arrayview-opener-0.2.0/`
  (shared by both SSH-remote and tunnel connections)
- **Also in**: `~/.vscode/extensions/arrayview.arrayview-opener-0.2.0/` (local desktop)
- **extensionKind**: `["workspace"]` — runs on the remote side
- **activationEvents**: `["onStartupFinished", "onUri"]`

**What it does:**
1. Detects `IS_REMOTE_SERVER` at load time: `__dirname.includes('.vscode-server')` or `.vscode/cli/servers`
2. On activation, starts `fs.watch` on `~/.arrayview/` + 2s polling fallback
3. When `open-request.json` appears:
   - Reads and parses the signal file
   - **Targeting**: if `remote=true` but running locally → skip (leave for remote instance); vice versa
   - Deletes the file
   - Calls `simpleBrowser.show(localhostUrl)` — Simple Browser auto-forwards through tunnels

**Log file**: `~/.arrayview/extension.log`

**IMPORTANT**: Must be installed in the tunnel server via the server CLI:
```bash
~/.vscode/cli/servers/Stable-<commit>/server/bin/code-server --install-extension <path>.vsix
```
Manually copying files to `~/.vscode-server/extensions/` may NOT be recognized.

### 2. Python Backend (`_app.py`)

Key functions for tunnel support:

| Function | Purpose |
|----------|---------|
| `_is_vscode_remote()` | Detects tunnel/remote: `VSCODE_IPC_HOOK_CLI` set + not macOS/Windows |
| `_find_code_cli()` | Finds `code` binary, preferring remote-cli in `~/.vscode/cli/servers/` |
| `_find_vscode_ipc_hook()` | Walks process tree to recover IPC hook stripped by `uv run` |
| `_ensure_vscode_extension()` | Installs bundled `.vsix` via `code --install-extension` if missing/outdated |
| `_open_via_signal_file(url)` | Writes `~/.arrayview/open-request.json` for the extension |
| `_open_browser(url)` | For remote: uses signal file. For local: tries `code --open-url`, `xdg-open`, etc. |

All user-facing URLs use `localhost` (not `127.0.0.1`) — VS Code only auto-forwards `localhost:PORT`.

### 3. Tunnel Configuration

- Config file: `~/.vscode/cli/code_tunnel.json`
  ```json
  {"name":"roodnoot","id":"quick-horse-ztth1dg","cluster":"euw"}
  ```
- Tunnel URL pattern: `https://roodnoot-<PORT>.euw.devtunnels.ms`

---

## Diagnostic Checklist

Run `diagnostics.py` (in project root) from a **tunnel terminal** to check everything:
```bash
uv run python diagnostics.py
```

### Manual Checks

1. **Extension log** — should show `activate` entry with today's date:
   ```bash
   cat ~/.arrayview/extension.log
   ```

2. **Extension installed for tunnel?**
   ```bash
   ls ~/.vscode-server/extensions/ | grep arrayview
   ```

3. **Environment in tunnel terminal** — must have `VSCODE_IPC_HOOK_CLI`:
   ```bash
   echo $VSCODE_IPC_HOOK_CLI
   ```

4. **Signal file consumed?** After running arrayview, check:
   ```bash
   ls -la ~/.arrayview/open-request.json  # should NOT exist (consumed by extension)
   ```

---

## Test Procedure

### Step 0: Reload Window (first time after install)
On the **laptop** VS Code tunnel window:
- `Ctrl+Shift+P` → "Developer: Reload Window"
- Wait for reconnection

### Step 1: Verify Extension Active
In a **tunnel terminal**:
```bash
cat ~/.arrayview/extension.log | tail -5
# Should show: activate (remoteName=..., appHost=...)
```

### Step 2: Test Signal File Manually
```bash
mkdir -p ~/.arrayview
echo '{"url":"http://localhost:12345/test"}' > ~/.arrayview/open-request.json
# Wait 2 seconds
cat ~/.arrayview/extension.log | tail -5
# Should show: Signal file consumed: http://localhost:12345/test
ls ~/.arrayview/open-request.json
# Should say "No such file" (consumed by extension)
```

### Step 3: Test ArrayView
```bash
cd /localscratch/oheide/projects/arrayview
uv run arrayview small_array.npy --browser
```
Should open Simple Browser tab with the array viewer.

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| No `activate` in extension log | Extension not activated after install | Reload VS Code window |
| `VSCODE_IPC_HOOK_CLI` empty in tunnel terminal | Env var stripped by `uv run` | `_find_vscode_ipc_hook()` walks process tree; check `diagnostics.py` output |
| Signal file not consumed | Extension not watching / NFS issue | Check extension log for `fs.watch OK`; polling fallback should handle NFS |
| Signal says "targets remote but I'm local" | Signal has `remote:true` but only local ext running | Tunnel ext host not activated. Reload tunnel window. Check `code-server --install-extension` was used |
| SimpleBrowser shows white screen | Port not forwarded through tunnel | Verify tunnel is connected; try `curl localhost:PORT` from tunnel terminal |
| Manual ext install not recognized | Files copied but not registered with code-server | Use `~/.vscode/cli/servers/Stable-xxx/server/bin/code-server --install-extension <vsix>` |

---

## File Locations

| File | Purpose |
|------|---------|
| `src/arrayview/_app.py` | Python backend (all tunnel logic) |
| `vscode-extension/extension.js` | Extension source |
| `vscode-extension/package.json` | Extension manifest |
| `src/arrayview/arrayview-opener.vsix` | Bundled vsix (included in wheel) |
| `~/.arrayview/extension.log` | Extension runtime log |
| `~/.arrayview/open-request.json` | Signal file (transient) |
| `~/.vscode/cli/code_tunnel.json` | Tunnel name/cluster config |
| `~/.vscode-server/extensions/` | Where tunnel extensions are installed |

---

## Attempt Log

### Attempt 1 — 2026-03-02 (Session with Copilot, v0.1.3)

**Starting state:**
- Extension v0.0.2 local, v0.1.3 in `~/.vscode-server/extensions/` (manually copied)
- Stale signal file unconsumed, tunnel daemon running

**Actions taken:**
1. Upgraded local extension v0.0.2 → v0.1.3
2. Cleaned stale signal file
3. Created `diagnostics.py` and rewrote this `TUNNEL_PLAN.md`

**Result:**
- Browser tab opened **on the REMOTE machine's physical desktop** (not in VS Code on laptop)
- Got 404 at `https://roodnoot-8000.euw.devtunnels.ms` (port not actually forwarded)
- White screen in Simple Browser in local VS Code

**Root cause (from extension log):**
- `remoteName=undefined, appHost=desktop` → the **local desktop** VS Code's extension instance consumed the signal, not the tunnel's
- `openTunnel(8000) failed: Extension CANNOT use API proposal: tunnels` — proposed API
- `asExternalUri` returned URL unchanged (no forwarding from local desktop)
- Constructed devtunnel URL manually → 404 (port not forwarded that way)
- `openExternal()` opened browser on remote's physical desktop

**Lesson:** Both local and tunnel extension instances share `~/.arrayview/` — need **targeting** to ensure only the correct instance consumes the signal.

---

### Attempt 2 — 2026-03-02 (Session with Copilot, v0.2.0)

**Starting state:**
- Extension v0.2.0 rewritten: removed all broken APIs (openTunnel, asExternalUri, devtunnel URL, openExternal)
- Simplified to just `simpleBrowser.show(localhostUrl)` — auto-forwards through tunnels
- Added `IS_REMOTE_SERVER` targeting: signal has `"remote": true`, local extension skips it
- v0.2.0 manually installed into `~/.vscode-server/extensions/` AND `~/.vscode/extensions/`

**Test result:**
- Local desktop VS Code v0.2.0 correctly logged: `Signal targets remote but I'm local — leaving for remote instance` ✅
- But the **tunnel extension host NEVER activated v0.2.0** — no log entry from the tunnel side
- Signal file left unconsumed; timed out

**Root cause:**
- Extension was manually copied to `~/.vscode-server/extensions/` and manually added to `extensions.json`
- The tunnel's code-server process did NOT recognize the manually-installed extension
- The tunnel server uses `~/.vscode-server/` (confirmed: `product.json` → `serverDataFolderName: .vscode-server`)
- But proper registration requires using the server CLI: `code-server --install-extension <vsix>`

**Fix applied:**
- Re-installed via `~/.vscode/cli/servers/Stable-xxx/server/bin/code-server --install-extension <vsix>`
- CLI reported: `Extension 'arrayview-opener.vsix' was successfully installed.`
- `extensions.json` now has proper metadata: `{source: "vsix", installedTimestamp: ..., pinned: true}`

**Next step:** User reconnects tunnel from laptop, verifies extension activates with `isRemoteServer=true`, tests arrayview

---

### Attempt 3 — 2026-03-02 (Session with Copilot, code --open-url approach)

**Starting state:**
- Attempt 2 fix (code-server --install-extension) was tried — extension properly installed to `~/.vscode-server/extensions/` with correct metadata
- User reconnected tunnel window and tested
- Result: same as before — tunnel extension host at `15:42` loaded copilot-chat but NOT arrayview-opener. Only the local desktop extension logged.

**Root cause:**
- For tunnel connections, the **CLIENT** (laptop VS Code) controls which extensions activate on the remote side
- Installing extensions server-side via `code-server --install-extension` puts files on disk but the client never picks them up unless it also registers them
- The only built-in non-copilot extensions that activated were all built-in (`vscode.git`, `vscode.merge-conflict`, etc.)
- Extension must be installed through the tunnel client UI (Extensions panel → "Install from VSIX...")

**Fix: Bypass extension entirely for tunnels — use `code --open-url` from remote CLI**

Changed `_open_browser()` in `_app.py` to:
1. **Primary**: Use `code --open-url http://localhost:PORT` via the tunnel's remote CLI + recovered IPC hook
   - Opens external browser on the laptop
   - VS Code auto-forwards the port through the tunnel
2. **Secondary**: Fall back to signal file (for when extension IS properly installed)
3. Always print the URL as clickable text

Also improved:
- `_is_vscode_remote()`: Now recovers IPC hook from process tree (catches `uv run` env stripping)
- `_find_code_cli()`: Now also checks `_is_vscode_remote()` to find remote-cli without env var
- `_find_vscode_ipc_hook()`: Added result caching to avoid repeated process tree walks

**Next step:** User tests from tunnel terminal: `uv run arrayview small_array.npy --browser`

**Attempt 3 Result (user test from tunnel):**
- `code --open-url http://localhost:8000/shell?init_sid=<sid>&init_name=<name>` ran successfully
- A **new VS Code window** opened on the laptop (NOT Simple Browser in the tunnel window)
- The arrayviewer HTML loaded! → Port auto-forwarding **WORKS** through the tunnel
- BUT: viewer displayed **"Connecting... Session not found or expired"**
- This means `/metadata/<sid>` returned 404 after 15 retries from the viewer JavaScript
- Server is confirmed alive (`/ping` returns OK, `/sessions` lists sessions)
- Root cause unclear — the session IS in the daemon's SESSIONS dict (verified via curl)

**Diagnosis of "Session not found":**
- The daemon subprocess creates the session and serves HTTP correctly (confirmed locally)
- Port forwarding works (shell HTML loads on the laptop)
- But somehow `/metadata/<sid>` fails through the tunnel
- Possible causes: (a) URL query params lost/mangled by `code --open-url`, (b) iframe fetch goes to wrong origin through VS Code proxy, (c) race condition between session creation and request

**Remaining issues:**
1. `code --open-url` opens a new VS Code window instead of Simple Browser in the tunnel window
2. Session not found through tunnel (despite working locally)

---

### Attempt 4 — 2026-03-03 (Session with Copilot, robust fallback)

**Starting state:**
- Attempt 3 showed `code --open-url` gets the viewer to load on the laptop (port forwarding works!)
- Session not found through the tunnel — unknown root cause

**Fixes applied:**

1. **Shell HTML session fallback** (`_shell.html`):
   - When `init_sid` is present in URL, now verifies it exists via `GET /metadata/<init_sid>` before creating the tab
   - If the session doesn't exist (404), falls back to `GET /sessions` and loads all available sessions
   - This handles stale/wrong sids and URL mangling scenarios

2. **Viewer HTML enhanced diagnostics** (`_viewer.html`):
   - If `/metadata/<sid>` fails after retries, fetches `/sessions` to see what's available
   - If sessions exist with different sids, auto-redirects to the first available session
   - Shows diagnostic info: what sid was tried, what sessions the server has
   - Enables self-healing: even if the sid is wrong, the viewer finds and loads the right session

3. **Daemon logging** (`_app.py`):
   - Daemon subprocess now logs to `~/.arrayview/daemon.log` instead of /dev/null
   - Logs session creation (sid, shape, count), startup, and shutdown events
   - Helps diagnose session lifecycle issues

4. **Enhanced `_open_browser` logging** (`_app.py`):
   - Logs exact `code --open-url` command and its stdout/stderr return
   - Always prints clickable URL in terminal as fallback

**Next step:** User tests from tunnel terminal: `uv run arrayview small_array.npy --browser`
