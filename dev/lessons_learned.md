# Lessons Learned

## DICOM Series Geometry and Metadata

**Problem:** Filename or `InstanceNumber` ordering can silently scramble slices,
and a generic header dump leaks identifying fields.
**Solution:** Group by `SeriesInstanceUID`, order by `ImagePositionPatient`
projected onto the orientation normal, validate geometry across the series, and
fall back to unique `InstanceNumber` only with a warning. Keep Shift+I metadata
allowlisted and attach provenance to every derived or inferred value.

Hard-won knowledge from past development sessions. Check this before starting work on related areas.

## Launch Identity and Ownership

**Problem:** A port and PID are not sufficient evidence that a server is stale
or belongs to ArrayView. Concurrent launchers can also both observe a free port
before either child binds it.
**Solution:** Use an instance UUID, process-start identity, protocol/package
version, ownership mode, per-user registry record, and startup lock. Recheck the
server after acquiring the lock. Stop only when `/ping`, the live process birth
identity, and the registry record all agree. Never enumerate and kill arbitrary
port listeners.

## VS Code Viewer Readiness

**Problem:** Writing a signal file proves neither that the intended VS Code
window claimed it nor that the forwarded backend is reachable.
**Solution:** Correlate request, window, and server IDs through atomic ACK states:
`claimed`, `port_resolved`, `panel_opened`, `visibility_verified`, and
`backend_ready`. Validate `/ping` JSON and the expected instance ID through the
resolved URL. A blocking VS Code launch fails closed unless `backend_ready`
arrives; local mocks do not replace a real tunnel handoff.

## mex Scaffold Drift

**Problem:** `mex check` can produce noisy `MISSING_PATH` errors when `.mex` docs put a whole shell command or raw URL in one inline code span.
**Key insight:** Keep `.mex` command guidance parser-friendly: split the executable from the repo path, prefer repo-local wrappers like `.mex/sync.sh`, and install the hook with `.mex/setup.sh` / `mex watch` only after the scaffold is clean enough that post-commit drift output is mostly signal.

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

## Overlay HUD Startup State

**Problem:** The overlay HUD state defaulted to visible, but the DOM could remain
hidden until the shortcut was toggled twice.
**Fix:** Run the shared UI reconciler whenever the startup loading state ends,
including the animation, fallback, and no-animation paths. State and DOM must be
reconciled together after `av-loading` is removed.

For draggable floating chrome, store the user position separately from the
default anchor. The same anchor reconciler should either compute the default or
clamp the stored position after mode and viewport changes; pointer capture on a
dedicated grip avoids interfering with row controls and canvas navigation.

## CLI Overlay Names

Ordinary file overlays and stack-pattern overlays have different parsing
contracts. Keep their normalizers separate: file overlays resolve concrete
paths and filename-derived labels, while stack overlays preserve recursive
patterns and role names. Thread resolved labels through both existing-server
and spawned-daemon paths.

## Stack Case Pairing

For common `patient/modality/file` and `patient/masks/file` layouts, infer one
shared ancestor depth only when it produces unique, identical base-case sets.
Use that same depth for overlays, including sparse roles. Preserve positional
pairing for flat non-sparse patterns and require `--case-regex` when sparse
layout inference is ambiguous rather than guessing.

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

## ROI Shortcut Registry

**Problem:** ROI/segmentation docs and help can advertise an export key before
the key exists in the command registry.
**Fix:** When changing tool UI, verify every advertised key has both a command
and a keybind, then run `tests/test_command_reachability.py`.

## qMRI Range Windows

**Problem:** qMRI display ranges are owned by multiple paths: backend slice
headers, WebSocket frame metadata, frontend pane locks, and per-pane colorbars.
Fixing only one path can leave initial labels or keyboard range changes stale.
**Fix:** Keep qMRI role policy in `_synthetic_mri.py`, and route qMRI `d`
handling through hovered-pane frontend helpers so real and synthetic panes update
their own `lockedVmin/lockedVmax` plus matching `displayState`.

## Overlay HUD Focus

**Problem:** HUD-only hover styling makes the list look focused while masks on
the image remain equally prominent.
**Fix:** Thread transient per-overlay alpha values through every overlay render
path (HTTP and WebSocket; normal, compare, multiview, qMRI) so hover focus dims
the image overlays as well as the HUD rows.

## Overlay Outline Mode

**Problem:** Filling every clinical contour can hide anatomy when many masks are
loaded at once.
**Fix:** Treat outline-only as a backend compositing option and thread it
through HTTP/WebSocket render requests. Keep the toggle in both the floating HUD
and `/ o` overlay drawer so users can switch without opening help.

## Scoped Range Initialization

**Problem:** Skipping range recomputation for a scoped dimension does not stabilize
the display when the initial manual range is still null; the backend remains free
to return slice-local extrema.
**Fix:** Seed the scoped volume range during viewer initialization, and test the
fresh-open path without opening the histogram first.

## Orthoview First Frames

**Problem:** A pane transport that closes before its first frame can leave an empty
orthoview until unrelated navigation triggers another render.
**Fix:** Retry only panes without a first frame, with bounded backoff and mode guards,
and verify recovery without changing the collection index.

## Progressive Compressed NIfTI Loading

**Problem:** Waiting for a complete `.nii.gz` volume makes patient changes feel
stalled, but a naive background decoder can let overlay prefetch occupy the same
workers as the visible base image.
**Fix:** Stream supported axial planes in source order, release the requested
plane when it is ready, then finish the same decode into the volume cache. Keep
base and overlay continuations in separate bounded pools, allocate large output
arrays only when a worker starts, and fall back to the full loader for layouts
that cannot be streamed exactly.

## Loading Animation Frame Capture

**Problem:** Chromium full-page PNG captures can intermittently appear black in
an image inspection tool even when the saved PNG has normal pixel values.
**Fix:** Keep animated overlays as small fixed layers, assert that they do not
change canvas bounds, validate the saved canvas crop numerically, and inspect a
JPEG contact sheet before treating a suspicious PNG preview as a viewer bug.

## WebGL Colormap Picker Coverage

**Problem:** A WebGL renderer can appear compatible with the shared colormap
picker while silently missing preview or commit updates to its LUT texture.
**Fix:** Exercise the real picker in a browser test while the WebGL renderer is
visible, then assert both the selected colormap and renderer state remain active.

## Ortho WebGL Pane Ownership

**Problem:** A shared 3D canvas can escape the ortho grid, cover controls, or
leave a large black surface around a small volume.
**Fix:** Give each ortho pane its own absolutely positioned WebGL canvas inside
the existing canvas box, and verify its bounds against the corresponding 2D
canvas before testing navigation and colormap updates.
