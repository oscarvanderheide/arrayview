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
- Which backend version a click runs is *not* obvious: `arrayview.packageSpec`
  (machine-scoped, so it leaks across windows) can redirect `--with arrayview`
  to a local checkout, and an already-running daemon on port 8000 overrides
  both. Confirm with `curl -s localhost:8000/ping` → `package_version`.
  See "Testing backend changes without cutting a release" in `CONTRIBUTING.md`.
- Launch candidate order: workspace `.venv` → `uv tool` install (only when
  `packageSpec` is the default, so it cannot shadow a checkout) → `uv run
  --with` → `python3 -m`. The tool entry exists because `uv run --with` rebuilds
  its ephemeral env on every new release; a tool install trades that for an
  explicit `uv tool upgrade arrayview`.
