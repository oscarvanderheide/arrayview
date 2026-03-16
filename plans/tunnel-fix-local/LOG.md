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

**Result**: (pending — needs user to test on tunnel)
