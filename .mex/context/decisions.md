---
name: decisions
description: Key architectural and technical decisions with reasoning. Load when making design choices or understanding why something is built a certain way.
triggers:
  - "why do we"
  - "why is it"
  - "decision"
  - "alternative"
  - "we chose"
  - "why not"
edges:
  - target: context/architecture.md
    condition: when a decision relates to system structure
  - target: context/stack.md
    condition: when a decision relates to technology choice
  - target: patterns/vscode-display.md
    condition: when the decision involves VS Code display routing or IPC
last_updated: 2026-04-13
---

# Decisions

## Decision Log

### Single self-contained HTML frontend
**Date:** 2024 (pre-git history)
**Status:** Active
**Decision:** The entire frontend lives in one `_viewer.html` file with no build step.
**Reasoning:** Zero build infrastructure. Packaged as data inside the Python wheel. Trivially served by FastAPI. Editable in any text editor with no toolchain setup. Fast iteration — change the file, reload the browser.
**Alternatives considered:** React/Vite SPA (rejected — build step, npm dependency, complicates packaging into a Python wheel); separate CSS/JS files (rejected — adds serving complexity, breaks single-file distribution).
**Consequences:** All CSS and JS must stay in `_viewer.html`. No code splitting. Hot module replacement is not available — browser refresh is the dev loop.

### Lazy imports for all heavy dependencies
**Date:** 2024 (pre-git history)
**Status:** Active
**Decision:** numpy, matplotlib, nibabel, FastAPI, uvicorn are all imported lazily, not at module top-level.
**Reasoning:** CLI fast path — when the server is already running, `arrayview file.npy` just posts to the existing server and exits. Cold-starting numpy + matplotlib adds ~350 ms. Lazy import keeps this near-zero.
**Alternatives considered:** Eager imports with `__init__.py` guard (rejected — still pays cost on every CLI invocation even when server is alive).
**Consequences:** The `_LazyMod` proxy class and `_server_mod()` / `_nib()` accessor pattern must be used consistently. Any new heavy import must be lazy.

### Dedicated render thread (not `concurrent.futures`)
**Date:** 2024
**Status:** Active
**Decision:** The render thread is a raw `threading.Thread` pulling from a `_queue.SimpleQueue`, not a `ThreadPoolExecutor`.
**Reasoning:** Python's `concurrent.futures` executor sets a `_global_shutdown` flag during interpreter exit that prevents submitting new work. Long-running Jupyter sessions hit this during kernel restart. A raw daemon thread is unaffected.
**Alternatives considered:** `ThreadPoolExecutor` (rejected — `_global_shutdown` flag causes silent failures); `asyncio` tasks (rejected — rendering is CPU-bound; mixing into the event loop stalls the WS handler).
**Consequences:** `_RENDER_QUEUE` and `_RENDER_THREAD` must be managed manually in `_session.py`. The `_render()` coroutine bridges between the asyncio event loop and the render thread using `Future.set_result` via `call_soon_threadsafe`.

### Stable VS Code window ID via EnvironmentVariableCollection (v0.14.0)
**Date:** 2026 (recent, see commit ecd036f / 0635d80)
**Status:** Active
**Decision:** The VS Code extension stores a stable `ARRAYVIEW_WINDOW_ID` env var per window using `EnvironmentVariableCollection`. Python reads it to know which VS Code window to target without walking the process tree.
**Reasoning:** PID ancestry matching is unreliable on macOS — `ps` returns a flat list with no tree structure, so walking from Python up to the VS Code window PID fails. `EnvironmentVariableCollection` survives `uv run` env stripping and is per-window, solving the multi-window targeting problem.
**Alternatives considered:** PID ancestry walk (rejected — broken on macOS); process name heuristics (rejected — not unique); PPID-based (rejected — too shallow).
**Consequences:** When `ARRAYVIEW_WINDOW_ID` is missing (first activation or old extension), the code falls back to stale-env heuristics. See `_vscode.py`. Extension version tracking in `_VSCODE_EXT_VERSION = "0.14.0"`.

### Stdio transport for VS Code tunnel
**Date:** 2024
**Status:** Active
**Decision:** VS Code tunnel uses a separate stdio server (`_stdio_server.py`) where the extension spawns a Python subprocess and communicates over stdin/stdout rather than a TCP port.
**Reasoning:** VS Code tunnel port forwarding is unreliable for localhost connections in some network configurations. Stdio transport is always available and requires no port forwarding — the extension is the process parent.
**Alternatives considered:** TCP on forwarded port (rejected — forwarding not guaranteed); WebSocket to extension host (rejected — complex authentication).
**Consequences:** Two entirely separate server code paths must be kept in sync feature-wise. New server features implemented in `_server.py` must also be considered for `_stdio_server.py`.

### pywebview in subprocess (not in-process)
**Date:** 2024
**Status:** Active
**Decision:** `_open_webview()` always launches pywebview in a fresh `subprocess.Popen`, passing the URL as an argument.
**Reasoning:** When called from a Jupyter kernel, multiprocessing bootstrap fails because the kernel's `__main__` is not a standard Python script. A subprocess sidesteps this entirely.
**Alternatives considered:** `multiprocessing.Process` (rejected — Jupyter bootstrap issue); in-process pywebview (rejected — blocks the event loop, cannot coexist with uvicorn).
**Consequences:** The webview subprocess is a dumb browser with no Python API surface. All logic stays in the server.
