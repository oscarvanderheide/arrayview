---
name: debug-vscode-extension-python
description: Diagnosing VS Code custom-editor or direct-webview failures caused by Python resolution, workspace virtualenv drift, or uv fallback behavior.
triggers:
  - "ArrayView failed to open"
  - "Python process exited with code 1"
  - "vscode extension"
  - "custom editor"
  - "remote tunnel"
  - "uv run --with arrayview"
  - "python 3.12"
edges:
  - target: context/conventions.md
    condition: when making code changes after confirming the failure mode
last_updated: 2026-05-06
---

# Debug VS Code Extension Python

## Context

The VS Code opener extension starts ArrayView in two different ways:
1. Signal-file mode from Python (`view(...)`, CLI, Julia, etc.)
2. Custom-editor mode when the user clicks `.npy` / `.nii` / similar files in VS Code

Custom-editor mode lives in `vscode-extension/extension.js` and spawns:
- explicit `pythonPath` from a signal file when present
- workspace `.venv/bin/python` when present
- `python3`
- `python`
- `uv run --python 3.12 --with arrayview python`

ArrayView requires Python `>=3.12`. A common tunnel failure is a workspace `.venv`
or default shell Python at `3.11`, which makes the final `uv` fallback fail with
an unsatisfiable dependency error.

## Steps

1. **Check the extension log first**
   ```bash
   tail -n 120 ~/.arrayview/extension.log
   ```
   Look for `CUSTOM-EDITOR:`, `PYTHON: spawning`, stderr lines, and the final exit code.

2. **Identify which candidate failed**
   The log shows the exact command order. Typical failure pattern:
   ```
   PYTHON: spawning /path/to/.venv/bin/python -m arrayview --mode stdio file.npy
   PYTHON: ... No module named arrayview
   PYTHON: spawning uv run --python 3.12 --with arrayview python -m arrayview --mode stdio file.npy
   ```

3. **Reproduce from the failing workspace root**
   ```bash
   timeout 10s sh -c 'uv run --python 3.12 --with arrayview python -m arrayview --mode stdio file.npy </dev/null'
   ```
   Success is a `SESSION:{...}` line on stderr. If it exits `1`, keep the stderr.

4. **Check the workspace virtualenv directly**
   ```bash
   .venv/bin/python - <<'PY'
   import arrayview, sys
   print(sys.executable)
   print(arrayview.__file__)
   print(getattr(arrayview, '__version__', 'missing'))
   PY
   ```
   If this raises `ModuleNotFoundError`, the workspace `.venv` exists but does not
   have ArrayView installed.

5. **Interpret the usual root causes**
   - `No module named arrayview` from `.venv/bin/python`: project env exists but lacks ArrayView
   - `current Python version ... does not satisfy Python>=3.12`: the fallback is using an older interpreter
   - `SESSION:{...}` appears: stdio startup is healthy; any remaining issue is on the extension/webview side

6. **If editing `extension.js`**
   Follow the VS Code extension skill checklist: bump `vscode-extension/package.json`, bump `_VSCODE_EXT_VERSION` in `src/arrayview/_vscode_extension.py`, rebuild `src/arrayview/arrayview-opener.vsix`, and verify the embedded version.

## Gotchas

- Running `python -m arrayview --mode stdio ...` directly from a shell script without redirecting stdin can feed shell text into the stdio server. Use `</dev/null` for bounded startup checks.
- In tunnel workspaces, the clicked-file custom editor does **not** use the current repo checkout automatically unless that workspace `.venv` points to it.
- A reboot or deleting `~/.vscode-server` can change the default interpreter resolution without changing the extension code.

## Verify

- [ ] `~/.arrayview/extension.log` shows the expected candidate order
- [ ] The failing workspace reproducer emits `SESSION:{...}` with the intended interpreter
- [ ] If `extension.js` changed, `vscode-extension/package.json`, `_VSCODE_EXT_VERSION`, and the VSIX version all match
- [ ] After reinstalling or reloading, the custom editor opens the target array without `Python process exited with code 1`

## Update Scaffold
- [ ] Update `.mex/context/project-state.md` if the shipped fallback behavior changed
- [ ] Update any `.mex/context/` files that are now out of date
- [ ] Add this pattern to `.mex/patterns/INDEX.md`