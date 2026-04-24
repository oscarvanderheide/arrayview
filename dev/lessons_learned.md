# Lessons Learned

Hard-won knowledge from past development sessions. Check this before starting work on related areas.

## VS Code Extension / Simple Browser

**Problem:** Extension install and signal-file IPC breaks frequently when touched.
**What didn't work:** Changing `--force` install logic, modifying IPC hook detection without testing all paths.
**Solution:** Always load the `vscode-simplebrowser` skill before touching this area. Test on both local VS Code and tunnel. Never `--force` reinstall if the correct version is already on disk.

## Colorbar / Histogram Height

**Problem:** Histogram expanded height overflows in multi-view mode.
**Root cause:** `_computeCbExpandedH()` was using `slim-cb-wrap` (normal mode) in all modes. In multi-view, the active colorbar is `mv-cb-wrap`.
**Fix:** Use the correct wrapper ID based on mode. Always call `_computeCbExpandedH()` before expanding the histogram.

## Dynamic Island Positioning

**Problem:** Colorbar, eggs, and info elements use `position: fixed` and must be repositioned when switching modes (normal, multi-view, immersive, compare).
**Key insight:** Each mode has its own positioning logic. When adding features that affect layout, check all modes — not just the one being developed. Use `/ui-consistency-audit` skill.

## Auto-fit and Zoom State

**Problem:** `_fitZoom`, `userZoom`, `_zoomAdjustedByUser`, and `_autoFitPending` interact in subtle ways.
**Key insight:** `_fitZoom` is recomputed whenever `scaleCanvas` runs with `_autoFitPending = true`. When entering/exiting immersive mode, set `_autoFitPending = true` and `_zoomAdjustedByUser = false` before calling `setFullscreenMode()` so the layout recalculation uses the correct viewport size.

## VS Code Multi-Window Signal Targeting

**Problem:** With 2 VS Code windows open, SimpleBrowser opens in the wrong window.
**Root cause:** On macOS, the VS Code extension host process cannot find `VSCODE_IPC_HOOK_CLI` by walking up its process tree (the hook is only inherited by terminal shell processes, not the extension host). So extensions register in "PID mode" (`fallbackId: true`, `hookTag: ""`). Meanwhile, Python CAN find the hook via parent-process walk from the terminal shell. This mismatch meant Python wrote to `open-request-ipc-{hookTag}.json` but extensions only watched `open-request-pid-{EXT_PID}.json`.
**Fix (v0.9.20):**
1. Extension records `ppids` (ancestor PIDs up to depth 8) in `window-{id}.json` registration.
2. Python detects PID-mode extensions (no hookTag in registrations) and falls through to ancestor-PID matching: collects its own ancestor PIDs, finds the window whose extension host shares the closest common ancestor (renderer process is per-window → unique discriminator between windows).
3. Python writes `open-request-pid-{EXT_PID}.json` when extension is in PID mode, matching what the extension actually watches.
**Key insight:** On macOS, the VS Code renderer process is per-window and is a common ancestor of BOTH the extension host and the terminal's PTY host. Use depth-scored ancestor intersection to find the correct window.

## Multi-view vs Normal Mode Colorbar

**Problem:** Two separate colorbar systems: `slim-cb-wrap` (normal/compare) and `mv-cb-wrap` (multi-view).
**Key insight:** Many colorbar functions need to check `multiViewActive || compareMvActive` to pick the right element. When adding colorbar features, always handle both paths. The `ColorBar` class abstracts some of this but the global state (`_cbExpanded`, `_cbAnimT`, etc.) is still shared.

## UI Audit Stability

**Problem:** `tests/ui_audit.py` pixel diffs always failed for zoom scenarios (20-40% diff per run).
**Root cause 1:** Baselines predate major UI changes — stale baselines fail everything. Fix: run `--update-baselines` after intentional UI changes.
**Root cause 2:** Zoom scenarios have non-deterministic canvas content (sub-pixel rendering differences, canvas pan position). Even with seeded test data, the rendered pixels vary slightly.
**Fix:** Skip pixel diff for zoom scenarios (`scenario.zoom > 0`). Layout correctness in zoom is validated by DOM assertions (R2/R3/R14 etc.) which are deterministic. Also reset `canvas-wrap` scroll to (0,0) after zoom_in for consistent layout.
**Key insight:** Pixel diffs are reliable for static modes (fit/compare/qmri). For zoom/pan modes, rely on DOM assertions only.

## cb-tri-zone Yellow Arrows

**Problem:** `.cb-tri-zone` (multiview colorbar) had CSS `position: absolute; bottom: -10px` but height:0. Arrows appeared correct by DOM inspection but were invisible.
**Root cause:** The CSS for `.cb-tri-zone.expanded` adds `height: 10px`, but the `expanded` class was never added in multiview mode because the drawMvColorbar() function didn't sync the class.
**Fix (batch commit):** Added `.cb-tri-zone` class CSS rule (was only `#slim-cb-tri-zone`); `drawMvColorbar()` now syncs `expanded` class.
**Verification:** Check `document.querySelector('.cb-tri-zone').classList.contains('expanded')` 100ms after pressing 'd' — should be `true`.

## UI Reconciler Pattern

**Problem:** UI element visibility scattered across 15+ toggle sites per element. Mode combinatorics make it impossible to keep all sites in sync — same class of bug fixed 4+ times for colorbars alone.

**Solution:** Grouped reconciler functions that derive UI state from mode flags:
- `_reconcileLayout()` — container visibility (canvas-wrap, compareWrap, array-name, mv-orientation, canvas-bordered)
- `_reconcileCompareState()` — compare sub-mode UI (diff/wipe panes, wipe-mode/focus classes, flex-wrap)
- `_reconcileCbVisibility()` — colorbar/island visibility
- `_reconcileUI()` — wrapper that calls all three

**Key insight:** Mode entry/exit functions set flags, then call the reconciler. The reconciler reads flags and computes correct DOM state. No function needs to know what other functions do — the reconciler is the single source of truth.

**When adding new modes or UI elements:** Add the visibility rule to the appropriate reconciler. All existing call sites automatically get the update.

## Server Route Extraction End State

**Problem:** `_server.py` kept shrinking as feature routes moved out, but it was unclear whether the final goal was “zero routes left” or a stable orchestration layer.
**Key insight:** The natural stopping point is not an empty `_server.py`. Keep FastAPI app setup, shared dependency injection (`get_session_or_404`), template/asset loading, and the tiny infrastructure routes (`/`, `/ping`, `/shell`, `/gsap.min.js`, `/colormap/{name}`) inline. Move feature domains with real business logic into `register_*_routes()` modules.
**Practical rule:** Before extracting another tiny route, ask whether it materially reduces coupling or just hides the server assembly story. WebSocket, rendering, state, persistence, query, loading, and analysis clusters are worth extracting; the small infrastructure surface is not.
*** Add File: /Users/oscar/Projects/packages/python/arrayview/.mex/patterns/extract-server-route-module.md
---
name: extract-server-route-module
description: Moving an existing route cluster out of `_server.py` into a dedicated `_routes_*.py` module while preserving compat surfaces and focused validation.
triggers:
	- "extract route"
	- "route refactor"
	- "split _server.py"
	- "move routes"
edges:
	- target: context/architecture.md
		condition: when deciding whether `_server.py` should keep or delegate a route cluster
	- target: context/project-state.md
		condition: when the extraction changes the active architecture workstream
last_updated: 2026-04-25
---

# Extract Server Route Module

## Context

`_server.py` is now the assembly layer for the FastAPI app. Extract clusters that own meaningful feature behavior. Leave tiny infrastructure routes and shared dependency helpers inline unless there is real coupling pressure.

## Steps

1. Start from one concrete inline cluster in `_server.py` and add direct API coverage for its current contract before moving code.
2. Create a focused `_routes_*.py` module with `register_*_routes(app, ...)` and import dependencies directly from their source modules.
3. Preserve any compat symbols still expected at the `_server.py` surface by importing them back into `_server.py` instead of forcing wide call-site churn.
4. Register the new module from `_server.py` and remove only the extracted route block.
5. Trim `_server.py` imports after the move so the file reads like app assembly, not a half-migrated monolith.

## Gotchas

- `_app.py` is a backward-compat shim. If it re-exports a helper that moved, either update `_app.py` to import from the real module or re-export the symbol from `_server.py` intentionally.
- `register_loading_routes()` still depends on `_notify_shells`; extracting WebSocket code without preserving that callable at the server surface breaks launcher paths.
- Do not over-extract the final infrastructure surface. `/`, `/ping`, `/shell`, `/gsap.min.js`, and `/colormap/{name}` are the natural inline end state.

## Verify

- [ ] Direct tests for the extracted cluster exist before the move
- [ ] The same focused tests pass after extraction
- [ ] `uv run python -m py_compile` passes for the touched modules
- [ ] `_server.py` still exposes any compat symbols relied on by `_app.py` or `_launcher.py`
- [ ] `.mex/context/project-state.md` and any stale architecture notes are updated if the extraction changes the intended server shape

## Update Scaffold
- [ ] Update `.mex/ROUTER.md` "Current Project State" if what's working/not built has changed
- [ ] Update any `.mex/context/` files that are now out of date
- [ ] If this is a new task type without a pattern, create one in `.mex/patterns/` and add to `INDEX.md`
