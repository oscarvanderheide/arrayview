# Tunnel Fix Plan

## Problem

After implementing fixes for issues 1-4, the VS Code tunnel workflow is broken
with `ERR_CONNECTION_REFUSED`. It was working (albeit buggy sometimes) in the
previous commit (`da69007`).

## Working tunnel workflow (pre-changes)

1. User runs `arrayview --serve` on remote (default port 8000)
2. User manually sets port 8000 to Public in VS Code Ports tab
3. User runs `arrayview file.npy` — server detects existing ArrayView on 8000,
   POSTs `/load`, extension opens Simple Browser via signal file
4. Simple Browser connects to `http://localhost:8000/?sid=...` which VS Code
   auto-forwards through the tunnel

## What changed (our diff)

### _vscode.py changes
1. Added `_remote_message_shown` flag (cosmetic — OK)
2. **Added `_configure_vscode_port_preview(parsed_port)` to the `is_remote` branch** — this writes VS Code Machine/User settings files to set `"privacy": "public"`. This was NOT called in the remote path before. It was only called in:
   - The local VS Code branch (line 410)
   - The CLI new-server path (line 1357)
3. Changed signal file write to always happen even if extension install fails — this is fine
4. Changed local VS Code path to always write signal file — this is fine

### _launcher.py changes
1. CLI default port changed 8000 → 8123
2. Auto-port scanning added to view(), _view_subprocess(), CLI
3. `_configure_vscode_port_preview(args.port)` added to `--serve` path
4. `blocking=force_vscode` added to `_view_subprocess` — only affects Julia

### _server.py changes
1. Session() wrapped in asyncio.to_thread() — shouldn't affect tunnel

## Root cause analysis

### Suspect #1: Port default change 8000 → 8123 (HIGH)

If the user had previously done `arrayview --serve` on port 8000 and manually
set port 8000 to Public, then after our changes `arrayview --serve` would start
on 8123 instead. The port 8123 would NOT be set to Public yet. The user would
need to set 8123 to Public in the Ports tab.

However the user says it "no longer works" — implying they've tried the full
workflow fresh. But 8123 vs 8000 could easily cause confusion if the user's
muscle memory assumes port 8000.

### Suspect #2: _configure_vscode_port_preview in remote branch (HIGH)

The original code NEVER called `_configure_vscode_port_preview` in the remote
`_open_browser` path. We added it. This function writes to:
- `~/.vscode-server/data/Machine/settings.json`
- `~/.vscode-server/data/User/settings.json`
- `~/.vscode/cli/data/Machine/settings.json`
- `~/.vscode/cli/data/User/settings.json`

It sets `"privacy": "public"` programmatically. But VS Code may not honor
file-based privacy changes for ports that are ALREADY forwarded. Worse: writing
these settings files at runtime could corrupt or overwrite other VS Code
settings, or VS Code could detect the change and reload settings in a way that
disrupts active port forwarding.

The `_configure_vscode_port_preview` function also strips JSON comments before
rewriting — if the settings had comments, they'd be lost. And the function
might create settings files that don't normally exist, confusing VS Code.

### Suspect #3: parsed_port fallback is 8000 (LOW)

In the `_open_browser` function, `parsed_port` defaults to 8000 on parse
failure. After changing the CLI default to 8123, if `_configure_vscode_port_preview`
is called with 8000 instead of the actual port, VS Code would configure the
wrong port as Public.

Actually: `parsed_port` parses from the URL, which should be correct. This is
a non-issue unless the URL format changes.

### Suspect #4: Auto-port scanning could pick a different port (MEDIUM)

If port 8123 is busy, the auto-scan picks the next free port (e.g. 8124). The
user's manually-public port is 8123 (or 8000 if they remember the old default).
The viewer would be on 8124 which is NOT public → ERR_CONNECTION_REFUSED.

But the user didn't mention port conflicts, and they said it WAS working before.

## Conclusion

Most likely culprits in order of probability:

1. **Adding `_configure_vscode_port_preview` to the remote `_open_browser` path**
   — this function was never called there before and may cause settings file
   corruption or VS Code confusion
2. **Port default change** — if user has old port 8000 cached/manual-public

## Fix plan

### Step 1: Revert the `_configure_vscode_port_preview` call from the remote `_open_browser` path

This was the most invasive change to the tunnel code path and was NOT in the
working version. The `_configure_vscode_port_preview` function was designed for:
- Local VS Code terminals (workspace-level `.vscode/settings.json`)
- Pre-server setup in the CLI new-server path (line 1357)

It should NOT be called inside the `_do()` callback that runs on every
`_open_browser` invocation in the remote path — that's too late and too
frequent.

Keep it in the `--serve` path (that's pre-server, runs once, and is appropriate).

### Step 2: Keep port default 8123 but update parsed_port fallback

Change the `parsed_port` fallback from 8000 to 8123 to match the new default.

### Step 3: Verify all other changes are safe for tunnel

Walk through the exact tunnel code path with our remaining changes:

1. `arrayview --serve --port 8123`:
   - Server starts on 8123
   - `_configure_vscode_port_preview(8123)` writes settings (OK, runs once)
   - User sets port to Public
   - ✓ Same as before except port number

2. `arrayview file.npy`:
   - `_server_alive(8123)` → True (server running)
   - POST to `/load` → Session created in thread (Fix 2A, should be fine)
   - `_open_browser(url, blocking=True, force_vscode=(window_mode == "vscode"))`
   - In `_do()`: `is_remote` → True
   - Print message (once only)
   - `_ensure_vscode_extension()` → attempt install
   - Write signal file always (Fix 1B)
   - Schedule retries
   - ✓ Should work

### Step 4: Test

- Run `uv run pytest tests/test_api.py tests/test_large_arrays.py -x`
- Manual test on tunnel if possible

## Non-goals

- Do NOT change the auto-port scanning logic (Fixes 3A/3B) — those are correct
  and help SSH users
- Do NOT revert the Session-in-thread fix (Fix 2A) — that's correct and helps
  tunnel users loading second arrays
- Do NOT revert signal-file decoupling (Fix 1B) — that helps Julia users
- Do NOT revert blocking=force_vscode (Fix 1A) — that only affects Julia path
