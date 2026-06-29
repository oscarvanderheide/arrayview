---
name: debug-vscode-extension-python
description: Diagnosing VS Code opener failures in the signal-file and forwarded WebSocket path.
triggers:
  - "ArrayView failed to open"
  - "vscode extension"
  - "remote tunnel"
  - "asExternalUri"
  - "port public"
edges:
  - target: context/conventions.md
    condition: when making code changes after confirming the failure mode
last_updated: 2026-06-26
---

# Debug VS Code Extension Opening

## Context

The VS Code opener extension no longer starts a Python subprocess or custom
editor transport. Python writes a signal file with a localhost viewer URL; the
extension resolves that URL through VS Code, opens a webview panel, and checks
the backend with `/ping`.

## Steps

1. **Check the extension log first**
   ```bash
   tail -n 120 "$HOME/.arrayview/extension.log"
   ```
   Look for `SIGNAL-DATA:`, `OPEN:`, `BACKEND:`, `PORT:`, and `asExternalUri`.

2. **Confirm the signal file contains a URL**
   Signal payloads should have `mode: "url"` and a `http://localhost:<port>/...`
   URL. Payloads without a URL are stale and should not be produced by current
   Python code.

3. **Check the backend directly from the extension host**
   The extension probes `/ping`. From the same remote shell:
   ```bash
   curl -fsS http://localhost:8000/ping
   ```
   Use the actual port from the signal file. A refused connection means the
   Python server did not stay alive long enough or the wrong port was signaled.

4. **Check tunnel forwarding**
   In remote/tunnel sessions, the extension should call port configuration and
   `asExternalUri`. If the forwarded URL opens but WebSocket fails, inspect VS
   Code's Ports view and whether the port was promoted to public.

5. **If editing `extension.js`**
   Bump `vscode-extension/package.json`, bump `_VSCODE_EXT_VERSION` in
   `src/arrayview/_vscode_extension.py`, rebuild
   `src/arrayview/arrayview-opener.vsix`, and verify the embedded version.

## Gotchas

- Always use `localhost`, not `127.0.0.1`; VS Code port forwarding keys off
  localhost URLs.
- The extension is a URL opener only. Do not reintroduce a second backend
  transport inside the extension.
- If `uv run arrayview --serve --port <port>` reports success but the port
  refuses connections, start the empty server directly as described in the repo
  AGENTS instructions.

## Verify

- [ ] Extension log shows the signal was handled once
- [ ] `/ping` works on the signaled `localhost` port
- [ ] Remote/tunnel URL was resolved with `asExternalUri`
- [ ] If `extension.js` changed, `vscode-extension/package.json`, `_VSCODE_EXT_VERSION`, and the VSIX version all match

## Update Scaffold
- [ ] Update `.mex/context/project-state.md` if the shipped opener behavior changed
- [ ] Update any `.mex/context/` files that are now out of date
