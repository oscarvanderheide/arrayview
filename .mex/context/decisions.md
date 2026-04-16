---
name: decisions
description: Key architectural and technical decisions with reasoning. Load when making design choices or understanding why something is built a certain way.
triggers:
  - "why do we"
  - "why is it"
  - "decision"
  - "alternative"
  - "we chose"
edges:
  - target: context/architecture.md
    condition: when a decision relates to system structure
  - target: context/stack.md
    condition: when a decision relates to technology choice
  - target: context/frontend.md
    condition: when a decision relates to the frontend architecture or viewer modes
last_updated: 2026-04-16
---

# Decisions

## Decision Log

### Single self-contained _viewer.html — no build step
**Date:** pre-2024 (initial design)
**Status:** Active
**Decision:** The entire frontend (CSS + JS) lives in one `_viewer.html` file with no bundler or build step.
**Reasoning:** The package is installed as a Python wheel. Shipping a single file that works as a package resource eliminates build infrastructure (npm, webpack, etc.) and makes the project trivially installable. The file is served directly from package resources via `importlib.resources`.
**Alternatives considered:** Separate JS/CSS files with a bundler (rejected — adds npm as a required toolchain and a build step to every contribution); Jinja2 templating (rejected — adds complexity without benefit when one file is sufficient).
**Consequences:** Frontend is ~15 600 lines in one file. All edits happen in `_viewer.html` only — never create companion JS/CSS files.

### Lazy imports everywhere in _launcher.py
**Date:** 2024 (extracted from _app.py)
**Status:** Active
**Decision:** `_launcher.py` does not import numpy, _session, _render, _io, or uvicorn at module level. All are deferred to first use via accessor functions (`_server_mod()`, `_uvicorn()`, etc.) or `_LazyMod`.
**Reasoning:** The CLI fast path — when the server is already running — only needs to send a `/load` HTTP request. Eager imports add ~300–350 ms per invocation, which is unacceptable for a tool used interactively dozens of times per session.
**Alternatives considered:** Importing everything eagerly (rejected — too slow); using `importlib.import_module` everywhere (equivalent, but the `_mod_cache = None` + accessor pattern is more readable and explicit).
**Consequences:** Any new heavy import added to `_launcher.py` must follow the accessor pattern. Violating this breaks the fast path.

### Global state lives exclusively in _session.py
**Date:** 2024 (modular refactor from _app.py)
**Status:** Active
**Decision:** `SESSIONS`, `SERVER_LOOP`, `VIEWER_SOCKETS`, `VIEWER_SIDS`, `SHELL_SOCKETS`, `_RENDER_QUEUE`, `_RENDER_THREAD` are all defined in `_session.py` and imported by name elsewhere. No other module defines or shadows these.
**Reasoning:** Before the refactor, global state was scattered across `_app.py`. Centralizing it in one module makes concurrent access patterns explicit and prevents accidental redefinition.
**Alternatives considered:** Thread-local storage (rejected — server loop and sessions need to be shared across threads); a dedicated `State` singleton class (rejected — no benefit over module-level globals for this use case).
**Consequences:** Any new global mutable state must be added to `_session.py`. Never redefine these names in `_server.py`, `_launcher.py`, or anywhere else.

### Render thread is threading.Thread + SimpleQueue, not concurrent.futures
**Date:** 2024
**Status:** Active
**Decision:** The CPU-bound render work runs in a raw `threading.Thread` driven by `_queue.SimpleQueue`, not a `ThreadPoolExecutor`.
**Reasoning:** During Python interpreter shutdown, `concurrent.futures` sets a `_global_shutdown` flag that prevents submitting new work. Because the server can still receive requests during shutdown, this caused render failures. Raw `threading.Thread` with `daemon=True` is unaffected by the executor lifecycle.
**Alternatives considered:** `ThreadPoolExecutor` (rejected — shutdown race); `asyncio.run_in_executor` with default executor (same issue).
**Consequences:** The prefetch pool *does* use `concurrent.futures.ThreadPoolExecutor` (it submits non-critical background work and gracefully handles `RuntimeError` on shutdown). Only the critical render path must stay on `SimpleQueue`.

### stdio transport for VS Code tunnel (direct webview)
**Date:** 2024
**Status:** Active
**Decision:** When a VS Code tunnel is detected, `_stdio_server.py` replaces FastAPI+WebSocket. Messages are JSON on stdin, length-prefixed binary on stdout. The VS Code extension bridges `postMessage` ↔ subprocess stdio.
**Reasoning:** VS Code tunnel environments block arbitrary TCP ports. The Direct WebView API bypasses the network entirely by running the viewer as a subprocess of the extension. This gives reliable display without requiring port forwarding.
**Alternatives considered:** Port forwarding via VS Code tunnel (rejected — unreliable, depends on tunnel configuration); WebSocket over stdio tunnel (rejected — more complex than length-prefixed binary).
**Consequences:** `_stdio_server.py` must mirror every route and feature of `_server.py`. New server features must be implemented in both. The `_vscode.py` signal-file and shared-memory IPC are part of this same display path.

### ROI in qMRI uses per-pane overlay canvases, not a shared overlay
**Date:** 2026-04-16
**Status:** Active
**Decision:** Each qMRI pane gets its own `.qv-roi-overlay` canvas element. ROI shapes are drawn on all overlays simultaneously during drag, and stats are fetched per parameter map.
**Reasoning:** In qMRI mode, the single main canvas is hidden and replaced by N independent pane canvases, each with its own coordinate system and scale. A shared overlay (like the main `#roi-overlay`) cannot span multiple independently-positioned canvases. Per-pane overlays also allow the ROI mirroring UX where drawing on one pane instantly shows the same shape on all others.
**Alternatives considered:** A single overlay canvas spanning the entire `#qmri-view-wrap` (rejected — would need to track pane positions and clip per-pane, fragile across resize/mosaic layout changes); reusing the main `#roi-overlay` (rejected — it's inside `#canvas-viewport` which is hidden in qMRI mode).
**Consequences:** Overlay canvases must set `background: transparent` to override the global `canvas { background: var(--bg) }` rule. qMRI view objects carry `roiOverlay` and `roiCtx` properties. `_drawAllQvRois()` replaces `_drawAllRois()` in qMRI mode; `_redrawRoiOverlays()` dispatches to the correct function based on mode.

### UI visibility changes go through reconcilers, not ad hoc style/classList calls
**Date:** 2025 (View Component System refactor)
**Status:** Active
**Decision:** All visibility and layout state changes in `_viewer.html` must go through the four reconciler functions (unified UI reconciler, layout container visibility, compare sub-mode state, CB/island visibility).
**Reasoning:** Before reconcilers, mode-switch functions each managed their own ad hoc `style.display` toggles. This caused regressions where fixing one mode's layout broke another. Reconcilers enforce a single source of truth for UI state.
**Alternatives considered:** Each mode function manages its own visibility (previous approach, rejected due to regression rate); CSS class toggling with no reconciler (rejected — same problem in a different form).
**Consequences:** When adding a new UI element, wire it into the appropriate reconciler. Never set `style.display` or toggle classes in mode-entry/exit functions — that work belongs in the reconcilers.
