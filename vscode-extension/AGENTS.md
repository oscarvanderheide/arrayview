# VS Code Extension

## Key Functions (`_vscode.py`)

- `_ensure_vscode_extension()` — installs VSIX only if version not on disk
- `_open_via_signal_file()` — signal-file IPC, extension polls and opens panel (network mode)
- `_schedule_remote_open_retries()` — retry for tunnel/first-install latency

## Transport

- **Network**: FastAPI + WebSocket, extension opens a VS Code webview panel tab.
- Remote/tunnel sessions use VS Code forwarded ports plus `asExternalUri`; do not add a second backend transport inside the extension.

`_VSCODE_EXT_VERSION` in `src/arrayview/_vscode_extension.py` must match `vscode-extension/package.json`.
Rebuild: `cd vscode-extension && vsce package -o ../src/arrayview/arrayview-opener.vsix`

## High-Risk

- Skip `--force` if correct version on disk; clean stale dirs first
- IPC hook recovery when env vars stripped by `uv run` or subprocess wrappers
- tmux: `VSCODE_IPC_HOOK_CLI` not inherited; must walk client PIDs
- Signal routing: local → per-window targeted file; remote/tunnel → shared fallback
- Remote ports: configure preview, promote tunnel privacy when available, resolve URL via `asExternalUri`
