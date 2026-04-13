---
name: vscode-display
description: Working on VS Code display routing, IPC, extension install, signal files, or window ID. The most non-obvious integration in the codebase.
triggers:
  - "VS Code"
  - "vscode"
  - "Simple Browser"
  - "signal file"
  - "IPC"
  - "window ID"
  - "extension"
  - "VSIX"
  - "VSCODE_IPC_HOOK_CLI"
  - "ARRAYVIEW_WINDOW_ID"
  - "tunnel"
  - "stdio server"
  - "custom editor"
  - "click to open"
  - "ArrayViewEditorProvider"
edges:
  - target: context/architecture.md
    condition: for display routing table and server mode overview
  - target: context/decisions.md
    condition: for why window ID uses EnvironmentVariableCollection (v0.14.0 decision)
  - target: patterns/debug-render.md
    condition: if the viewer opens but renders incorrectly
last_updated: 2026-04-13
---

# VS Code Display

## Context

VS Code display routing has two paths and many failure modes. Read `context/decisions.md` before touching any of this.

**Path 1 — VS Code local terminal (network mode):**
```
view(arr)
  → _launcher.py detects _in_vscode_terminal()
  → _vscode._ensure_vscode_extension() installs/updates the VSIX
  → _vscode._open_direct_via_signal_file() writes open-request-v0900.json
  → Extension reads signal file → calls vscode.commands.executeCommand("simpleBrowser.show", url)
```

**Path 2 — VS Code tunnel (stdio mode):**
```
view(arr)
  → _launcher.py detects _in_vscode_tunnel() and _is_vscode_remote()
  → spawns _stdio_server.py as subprocess
  → VS Code extension connects to subprocess stdin/stdout
  → Binary RGBA frames flow over stdio (no TCP port)
```

**Key files:**
- `_platform.py` — all environment detection functions
- `_vscode.py` — extension install, signal-file IPC (`_open_direct_via_signal_file`), shm IPC (`_open_direct_via_shm`), browser open
- `_stdio_server.py` — stdio transport server
- `vscode-extension/extension.js` — the VS Code extension itself
- `src/arrayview/arrayview-opener.vsix` — bundled extension, version `_VSCODE_EXT_VERSION`

**Signal file:**
- Location: `~/.arrayview/open-request-v0900.json`
- Compat: extension also watches `open-request-v0800.json` (defined in `_VSCODE_COMPAT_SIGNAL_FILENAMES`)
- Max age: 60 seconds (`_VSCODE_SIGNAL_MAX_AGE_MS = 60_000`)
- Contents: `{"url": "...", "windowId": "...", "timestamp": ...}`

**Window ID (v0.14.0):**
- Extension injects `ARRAYVIEW_WINDOW_ID=<uuid>` per window via `EnvironmentVariableCollection`
- Python reads `os.environ.get("ARRAYVIEW_WINDOW_ID")` — if missing, falls back to stale-env heuristics
- `uv run` strips this env var — Python must check the EnvironmentVariableCollection path or fall back gracefully

## Task: Debug VS Code Display Not Opening

### Steps
1. Check `_in_vscode_terminal()` returns True: print `os.environ.get("TERM_PROGRAM")` and `os.environ.get("VSCODE_IPC_HOOK_CLI")`
2. If both are None: run `_platform._find_vscode_ipc_hook()` — it walks ancestor processes. If it returns None, the terminal is not recognized as VS Code.
3. Check extension is installed: `code --list-extensions | grep arrayview`
4. Check signal file is written: `ls -la ~/.arrayview/open-request-v0900.json`
5. Check signal file age — if older than 60s, the extension ignores it
6. Check extension logs: VS Code → Output panel → ArrayView Opener

### Gotchas
- `uv run` strips `VSCODE_IPC_HOOK_CLI` from the process env — detection must walk ancestors
- tmux detaches the process tree — `_find_vscode_ipc_hook()` has two extra fallback strategies for tmux
- On macOS, PID ancestry via `ps` is unreliable — `ARRAYVIEW_WINDOW_ID` (v0.14.0) is the fix
- The extension's `EnvironmentVariableCollection` update only takes effect in **new** terminals opened after extension activation
- `VSCODE_SIGNAL_MAX_AGE_MS = 60_000` — if Python writes the signal file and the extension takes >60s to read it (extension host reload), the file is rejected

## Task: Update the VS Code Extension

### Steps
1. Edit `vscode-extension/extension.js` or `vscode-extension/package.json`
2. Bump version in `vscode-extension/package.json`
3. Update `_VSCODE_EXT_VERSION` in `_vscode.py` to match
4. Build VSIX: `cd vscode-extension && vsce package` (requires Node.js + `vsce`)
5. Copy output `.vsix` to `src/arrayview/arrayview-opener.vsix`
6. Test: `view(arr)` in a fresh VS Code terminal — check extension auto-installs

### Gotchas
- `_VSCODE_EXT_VERSION` in `_vscode.py` must exactly match the version in `package.json` — mismatch causes auto-reinstall loop
- The VSIX is bundled inside the Python wheel (`hatchling` picks it up from `src/arrayview/`)
- Signal filename version (`open-request-v0900.json`) is separate from extension version — only change it when the signal format is breaking-changed
- Old signal filenames are listed in `_VSCODE_COMPAT_SIGNAL_FILENAMES` — keep them for one release cycle

## Task: Direct Webview / Custom Editor (Click-to-Open)

When an array file is opened via the VS Code custom editor provider (`ArrayViewEditorProvider`), the extension calls `setupArrayViewPanel()` which fetches the rendered viewer HTML from `_handle_get_viewer_html` in `_stdio_server.py` and sets it as `panel.webview.html`.

**Critical difference from FastAPI path:** the webview origin is `vscode-webview://…`, not the FastAPI server. Any resource loaded via a relative URL (e.g. `<script src="/gsap.min.js">`) resolves against the webview origin and 404s silently.

### Gotchas

- **GSAP must be inlined.** `_handle_get_viewer_html` replaces `<script src="/gsap.min.js"></script>` with an inline `<script>` containing the vendored GSAP content. If GSAP is not available, `_playStartupAnimation()` crashes (no guard on `typeof gsap`) and the loading overlay never closes — the spinner stays visible forever even though the viewer is functional (`pointer-events: none` on the overlay).
- **`__BODY_CLASS__` must be substituted.** `_server.py` replaces it with `"av-loading"` (or `""`) but `_stdio_server.py` must do the same. Without it, the body has the literal class `__BODY_CLASS__`, `body.av-loading` CSS rules never apply, and chrome elements are immediately visible — the startup animation has nothing to reveal.
- **Both substitutions live in `_handle_get_viewer_html` in `_stdio_server.py`.** Keep them in sync with `_server.py`'s `serve_viewer_html`.

## Verify

- [ ] `_VSCODE_EXT_VERSION` in `_vscode.py` matches version in `vscode-extension/package.json`
- [ ] Signal filename unchanged unless format changed (breaking change requires new version suffix)
- [ ] Tested in a real VS Code terminal (not just unit test) — the IPC path only works end-to-end
- [ ] Tested with `uv run arrayview ...` (env stripping path)
- [ ] If extension updated: VSIX file at `src/arrayview/arrayview-opener.vsix` is the new build

## Update Scaffold
- [ ] Update `context/decisions.md` if a new IPC mechanism was chosen
- [ ] Update "Current Project State" in `ROUTER.md` (extension version)
- [ ] Update `context/architecture.md` Display Routing table if routing logic changed
