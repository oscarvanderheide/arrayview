# ArrayView Agent Guide

This file defines how coding agents should work in this repository.

## Mission

Build and maintain a smooth `arrayview` experience across:

- Local: CLI, Python scripts, Jupyter, Julia via PythonCall
- Display modes: native `pywebview` and browser
- VS Code terminals: browser opens should prefer VS Code Simple Browser
- VS Code remote/tunnel sessions: viewer should open in the developer's VS Code client, not on the remote host browser

## Product Overview

`arrayview` is an interactive viewer for multi-dimensional arrays and medical/scientific volumes.
It runs a local FastAPI server with an HTML/JS frontend, then displays it either:

- in a native `pywebview` window,
- in a browser (including VS Code Simple Browser in VS Code terminals), or
- inline in Jupyter notebooks.

## Architecture

```
CLI / Python API
   |
   +- view()          Python entry point  (_launcher.py)
   +- arrayview()     CLI entry point (`uvx arrayview file.npy`)  (_launcher.py)
      |
      +- FastAPI server  (_server.py)
         +- /           viewer HTML
         +- /shell      pywebview shell HTML
         +- /ws/{sid}   WebSocket for render updates
         +- /load       register arrays
         +- /ping       health check
```

## Core Files

### Backend (server-side)

| File | Responsibility |
|------|---------------|
| `src/arrayview/_launcher.py` | Entry points (`view()`, `arrayview()` CLI), process management, window opening |
| `src/arrayview/_server.py` | FastAPI app, all REST routes, WebSocket handlers, HTML templates |
| `src/arrayview/_session.py` | Sessions, global state, caches, render thread, constants |
| `src/arrayview/_render.py` | Rendering pipeline: colormaps, LUTs, slice extraction, RGBA/mosaic/RGB |
| `src/arrayview/_vscode.py` | VS Code extension management, signal-file IPC, browser opening |
| `src/arrayview/_platform.py` | Platform/environment detection |
| `src/arrayview/_io.py` | Array I/O (load from file, format detection) |
| `src/arrayview/_app.py` | **Compat shim only** — re-exports from the modules above; do not add logic here |

### Frontend

| File | Responsibility |
|------|---------------|
| `src/arrayview/_viewer.html` | Viewer UI (single-file, all JS/CSS embedded) |
| `src/arrayview/_shell.html` | Shell page for native tab/window management |

### VS Code Extension

| File | Responsibility |
|------|---------------|
| `vscode-extension/extension.js` | VS Code opener behavior |
| `vscode-extension/package.json` | Extension metadata and version |
| `src/arrayview/arrayview-opener.vsix` | Packaged extension installed by Python code |

## Skills — When to Use

**Always invoke the relevant skill before touching the corresponding area.**

| Skill | Trigger |
|-------|---------|
| `viewer-ui-checklist` | ANY UI change: keyboard shortcuts, layout, new panels, canvas behavior. Keeps `visual_smoke.py` in sync. |
| `modes-consistency` | ANY visual feature: zoom, eggs, colorbars, canvas events, new rendering modes. Ensures the feature works across all six viewing modes (normal, multi-view, compare, diff, registration, qMRI). |
| `invocation-consistency` | ANY server, startup, or display-opening change. Ensures the feature works across all six invocation paths: CLI, Python script, Jupyter, Julia, VS Code tunnel, plain SSH. |
| `vscode-simplebrowser` | ANY change to VS Code extension install logic, signal-file IPC, `_ensure_vscode_extension()`, `_VSCODE_EXT_VERSION`, or `vscode-extension/`. Encodes the root-cause history of the extension-host reload race condition. |
| `task-workflow` | Feature or fix tasks — enforces one-commit-per-TODO-item workflow and required collateral updates (README/help/tests/CHANGELOG). |

Skill files live in `.claude/skills/` and are symlinked from `~/.claude/skills/`.

## Non-Negotiables

- Run commands directly when possible; avoid asking the user to run routine commands.
- Before trying a new fix, check whether it was already attempted in prior logs/notes.
- If re-trying a previously failed approach, explicitly note why it may work now.
- Avoid manual cleanup requirements for users. Viewer shutdown should be automatic and reliable.
- Do not regress existing working paths while fixing tunnel/remote behavior.
- Do not add logic to `_app.py` — it is a compat shim only. Add new logic to the appropriate module.

## Testing

```bash
uv sync --group test
uv run playwright install chromium

# Fast: HTTP API only (~2s)
uv run pytest tests/test_api.py -v

# CLI entry-point tests
uv run pytest tests/test_cli.py -v

# Browser/Playwright tests (~100s)
uv run pytest tests/test_browser.py -v

# All tests
uv run pytest tests/

# Visual smoke test — run after any UI change, review screenshots
uv run python tests/visual_smoke.py
# Screenshots saved to tests/smoke_output/
```

Visual regression baselines are in `tests/snapshots/`. Delete a snapshot file to reset its baseline.

## Validation Matrix

After any change, verify the affected paths:

| What changed | Minimum checks |
|---|---|
| Server / API | `pytest tests/test_api.py` |
| CLI / entry points | `pytest tests/test_cli.py` |
| Viewer UI | `pytest tests/test_browser.py` + `python tests/visual_smoke.py` |
| VS Code / platform | Manual: VS Code local terminal, VS Code remote/tunnel |
| Large array handling | `pytest tests/test_large_arrays.py` |

Manual smoke paths (for platform/display changes):
- Local CLI: `arrayview file.npy`
- Python script: `view(arr)`
- Jupyter inline (default)
- VS Code terminal browser path
- VS Code remote/tunnel path

## Platform Behavior (Must Preserve)

- Local Python/CLI: native window by default, browser fallback available
- Jupyter: inline iframe by default; explicit window mode still supported
- VS Code terminal: browser opens should target VS Code Simple Browser
- VS Code tunnel/SSH remote: should open in VS Code Simple Browser, never open UI on remote host browser by mistake
- Use `localhost` in URLs (not `127.0.0.1`) for reliable VS Code port forwarding

## VS Code Integration

Key functions (all in `_vscode.py`):

- `_ensure_vscode_extension()` — installs/updates the VSIX; must handle stale versions robustly
- `_configure_vscode_port_preview()` — sets up port forwarding for the viewer URL
- `_open_via_signal_file()` — IPC mechanism to open URLs in the VS Code client
- `_schedule_remote_open_retries()` — retries for tunnel environments where IPC may not be immediately available

`_VSCODE_EXT_VERSION` is defined in `_vscode.py` and must match `vscode-extension/package.json`.
If extension behavior changes, rebuild the VSIX and keep versioning in sync.

## Rebuild VS Code Extension

```bash
cd vscode-extension
vsce package -o ../src/arrayview/arrayview-opener.vsix
```

Then update `_VSCODE_EXT_VERSION` in `src/arrayview/_vscode.py`.

## High-Risk Areas

- VS Code extension install/update detection and stale extension versions
- Recovering VS Code IPC hook when env vars are stripped (`uv run` and subprocesses)
- Deciding when to use native window vs browser vs VS Code Simple Browser
- Port-forward/autoforward behavior in tunnel environments
- Shutdown lifecycle and orphan process prevention

## Workflow For Complex Debugging

1. Start a logfile `LOG_<FEATURE>.md`.
2. Record each significant attempt: hypothesis → change made → result → decision (keep/revert/follow-up).
3. Prefer incremental, testable changes.
4. Verify behavior in the most failure-prone environments: VS Code local terminal, VS Code remote/tunnel.
5. If behavior is not as expected, re-read the logfile before a new attempt.

## Source Of Truth

- End-user usage and setup: `README.md`

## Changelog

- **multi-overlay**: `view(arr, overlay=[mask1, mask2, ...])` now accepts a list of overlay masks, each auto-assigned a distinct palette color. Binary masks use the palette color; multi-label masks use per-label colors; continuous/float masks render as heatmap. Server compositing uses `_composite_overlays` in `_server.py`.
- **welcome-screen**: Canvas capped to 50% viewport height; yellow hint text (⌘O / Ctrl+O · drop file) shown below canvas instead of centered overlay. `body.welcome-mode` CSS class applied.
- **smart-colormap**: REMOVED. Automatic colormap selection at load time has been removed. `_recommend_colormap()` and `session.recommended_colormap` deleted; `/metadata` no longer returns `recommended_colormap`. `_recommend_colormap_reason()` is retained to power the info overlay (i key) Colormap row.
- **watch-mode**: `arrayview file.npy --watch` polls file mtime every 1 s and POSTs to `/reload/{sid}` on change; server reloads array in-place, clears caches, bumps `data_version`; frontend polls `/data_version/{sid}` every 1.5 s and re-renders on version bump.
- **export-slice**: `N` key exports the current 2-D slice (raw float data) as a `.npy` file download via `/export_slice/{sid}`. Works in all viewing modes; not available in RGB mode.
- **view-handle**: `view()` now returns a `ViewHandle` (str subclass) exposing `.update(arr)`, `.sid`, `.url`, `.port`. `ViewHandle.update()` POSTs new .npy bytes to `/update/{sid}` endpoint, which replaces session data in-place, clears caches, recomputes stats, and bumps `data_version` so the viewer re-renders automatically.
- **histogram-strip**: `W` key toggles a pixel-value histogram strip below the colorbar. `/histogram/{sid}` endpoint returns `{counts, edges, vmin, vmax}` for the current 2-D slice; frontend renders it on a `<canvas>` via `drawHistogram()`. Refreshes automatically on slice/index changes. Disabled in compare and RGB modes.
- **histogram-drag-clim**: Dragging the yellow vertical lines in the histogram (W key) changes `manualVmin`/`manualVmax` live. Hit radius is 7 px. Hovering anywhere on the histogram shows a tooltip with the bin value range and count. Cursor changes to `ew-resize` near a line. Smoke: scenario 55.
- **linked-crosshair**: REMOVED. The linked crosshair feature introduced in commit a778ab7 caused a visual bug (small array rendering in top-left corner of the window with info overlaid). All `.cv-crosshair` overlay canvases, `_drawCrossHair()`, `_clearAllCrossHairs()`, and mousemove/mouseleave listeners on compare canvases have been removed.
- **drag-reorder**: In compare mode, dragging a panel title (`.compare-title`) and dropping it onto another swaps their order instantly without re-fetching frames. Swaps `compareDisplaySids`, `compareFrames`, and per-pane `cmpManualVmin/cmpManualVmax`. Disabled in registration mode. Drag-over pane gets a dashed outline highlight.
- **screenshot-annotation**: `s` key screenshot now burns filename and current slice index into the bottom-left corner of the exported PNG. `_buildAnnotation()` collects name/shape/indices; `_annotateCanvas(src, lines)` renders a semi-transparent black box with white monospace text onto the offscreen canvas before `toDataURL`. Works in normal and multi-view modes.
- **rect-roi**: `A` key toggles rectangle ROI draw mode on the main canvas. Drag to select a rectangular region; on release, fetches `/roi/{sid}` with `x0,y0,x1,y1` pixel coordinates (data-space) and displays `{n, min, max, mean, std}` in the existing `#roi-panel`. Disabled in compare, multi-view, and qMRI modes. Circular ROI draw (drag without A mode) remains unchanged.
- **info-overlay-enhanced**: `i` key info overlay now includes a "Colormap" row showing the auto-selection reason (e.g. `"RdBu_r (signed data — vmin < 0)"`). `/info/{sid}` endpoint returns `recommended_colormap` and `recommended_colormap_reason` fields. Reason derived from `_recommend_colormap_reason()` in `_session.py`.
- **screenshot-save-location**: `s` screenshot status now reads "Screenshot saved to Downloads." and `N` slice export reads "slice saved to Downloads (.npy)". Help overlay keys `s` and `N` updated to show "(PNG → Downloads)" and "(.npy → Downloads)" respectively.
- **vscode-simplebrowser**: `_ensure_vscode_extension()` now skips `--force` reinstall when the correct version is already on disk (avoids extension-host reload that caused signal-file miss). `_VSCODE_SIGNAL_MAX_AGE_MS` raised from 30 s to 60 s. Local VS Code terminal path now schedules 2 retry signal writes at 10 s intervals (matches remote retry pattern) to survive any residual reload gap.
- **shift-o-picker**: `Shift+O` now opens the picker (same as `Cmd/Ctrl+O`). Welcome hint updated to "⌘O / Shift+O · drop array in window"; empty-hint and help overlay updated to match.
- **hover-info-immediate**: Pressing `H` to enable hover info now immediately shows the pixel tooltip at the current mouse position by dispatching a synthetic `mousemove` event on the canvas, instead of waiting for the user to move the mouse.
- **picker-rewrite**: Picker rewritten as a single unified mode (no tabs, no Open/Compare/Overlay switching). Space toggles selection on any listed array (up to 4); Enter with 0–1 selected opens/navigates; Enter with 2–4 selected enters compare mode with exactly those arrays (current session not auto-included). Overlay mode removed from picker. `enterCompareModeByMultipleSids(sids)` added; `_compareExplicitSids` state bypasses the always-prepend-current logic in `getCompareRenderSids()`. `applyCompareLayout()` and `updateCompareTitles()` use `_compareExplicitSids` for correct pane count before first render. Selected items get a green checkmark and highlighted border. Help overlay updated.
- **vscode-tab-title**: VS Code tab now shows "ArrayView: <array name>" instead of the URL. Extension switched from `simpleBrowser.show()` (title always "Simple Browser") to `vscode.window.createWebviewPanel()` with a custom label; title is passed in the signal payload (`data.title`) derived from `session.name`. Panels are reused by URL (revealed on repeat open). Multi-window fix: `_write_vscode_signal` now always writes to the per-window targeted file (`open-request-ipc-<hookTag>.json`) when `VSCODE_IPC_HOOK_CLI` is found — the registration-file (`window-<hookTag>.json`) existence check has been removed, eliminating the race where the shared fallback was written and claimed by the wrong window. Extension-side: shared-fallback signals with a mismatched `hookTag` are forwarded to the correct targeted file instead of being processed. Extension version bumped to 0.9.15. **Fix (v0.9.16)**: `enableScripts` was inadvertently left `false` in `createWebviewPanel`, which disabled all JavaScript inside the iframe and caused a "Connecting..." hang. Fixed by setting `enableScripts: true, enableForms: true`, adding a nonce-based CSP (`script-src 'nonce-...'`), and setting the iframe `src` via an inline script (matching VS Code's own Simple Browser pattern) rather than as an HTML attribute.
- **vscode-tunnel-signal**: Fixed VS Code tunnel: `_write_vscode_signal` now writes to the shared fallback file (`open-request-v0900.json`) when `_is_vscode_remote()` is True, instead of the per-window targeted file. Root cause: on a remote/tunnel, the extension host is spawned by VS Code Server (not the user's terminal) and does NOT inherit `VSCODE_IPC_HOOK_CLI`, so `OWN_HOOK_TAG=""` and `TARGETED_SIGNAL_FILE=null` in the extension — it only polls the shared fallback. On a tunnel there is only one VS Code session per remote, so there is no multi-window race concern. Also: tab title now correctly passed from all `_open_browser` call sites via `title=f"ArrayView: {name}"` — previously `title` was `None` in the CLI path because the session lives in a subprocess and `SESSIONS.get(sid)` returns `None` in the calling process. Retries (2× at 10 s) now also enabled on remote path since `createWebviewPanel` is idempotent.
- **histogram-colormap-bars**: Histogram bars (W key) are now colored using the active colormap. An offscreen 256-wide canvas draws the colormap gradient; each bar samples the gradient color at the fraction corresponding to its bin center within the current vmin/vmax window. Falls back to flat gray when no gradient stops are available. Smoke: scenario 60.
- **histogram-left-side**: Histogram strip (W key) repositioned from below the colorbar to the left side of the canvas, vertically aligned with it. Each bin maps to a y-band (value increases upward); bars grow rightward from the left edge. Clim lines are now horizontal; drag interaction is vertical (`ns-resize`). `bottomReserve` in `scaleCanvas` no longer includes histogram height. `_histVminY`/`_histVmaxY` replace old `_histVminX`/`_histVmaxX` for hit-testing. Smoke: scenarios 54, 55, 60.
- **axes-colormap-color**: Axis arrow indicators (`.axes-indicator`) now use a colormap-specific color instead of sampling canvas brightness. `gray` colormap → `var(--active-dim)` theme yellow; known colormaps (viridis, plasma, magma, inferno, hot, cool, RdBu_r/RdBu, bwr, seismic, coolwarm, jet, turbo, PiYG, PRGn) → a Nord/pastel color not present in that colormap; custom/unknown → white. `refreshAxesColor()` is also called on `c` and `C` (colormap change) key presses. Smoke: scenario 50.
- **welcome-hint-text**: Welcome hint and empty-hint now show `{cmd,ctrl,shift}+o · drop to open array` (plain ASCII, no platform-specific unicode). Help overlay updated to mention `Ctrl+O, Shift+O` alongside `Cmd+O`.
- **vscode-tmux-detection**: `_find_vscode_ipc_hook()` uses two tmux fallback strategies when `TERM_PROGRAM=tmux`. Strategy 1: `tmux show-environment VSCODE_IPC_HOOK_CLI` (works if the variable is in tmux's tracked env). Strategy 2: `tmux display-message -p '#{session_id}'` to get the current session, then `tmux list-clients -t <session_id> -F '#{client_pid}'` to enumerate ALL clients for that session, then reads each client's environment via `ps ewwww`. This correctly handles multiple attached clients (e.g. shared sessions, session created outside VS Code then attached from VS Code terminal). Scoping to the current session prevents false-positive detection from a VSCODE_IPC_HOOK_CLI belonging to a different VS Code window in another tmux session. See `invocation-consistency` skill for full tmux process-tree explanation.
- **picker-fix**: Removed duplicate stale `showUnifiedPicker` definition that was left by the picker-rewrite commit. The old version (second in the file) referenced undefined `_UNI_MODES`/`_UNI_LABELS`/`_UNI_COLORS` constants, causing a `TypeError` crash every time the picker was opened. Also removed the duplicate `getCompareCandidates` and old `enterComparePrompt` that used the superseded return format.
- **compare-pane-ordering**: Fixed wrong visual order and wrong pane names for 3/4-array compare. Root cause: `.compare-secondary` had CSS `order: 2` while `.compare-tertiary`/`.compare-quaternary` had no `order` (defaulting to `0`), so in a 3-pane compare the tertiary pane appeared before secondary. Fix: `applyCompareLayout()` now explicitly sets `pane.style.order = idx` overriding all class-based CSS orders. Also fixed drag-reorder to sync `_compareExplicitSids` when swapping panes.
- **native-window-tabs**: Fixed multiple pywebview windows spawning when calling `view()` more than once from a CLI script or Julia. Root cause (subprocess path): `/load` POST was missing `notify=True`, so `_notify_shells` was never called server-side and a new shell window was always opened. Fix: add `"notify": True` to the `/load` POST; if `result["notified"]` is True the tab was injected and no new window is opened. Also fixed the in-process `view()` path: `run_coroutine_threadsafe(_notify_shells(...))` result was previously ignored — if no shell socket was connected (window closed/crashed) no new window was opened. Fix: wait for the future result (3 s timeout); if `notified=False`, open a fresh native window.
- **histogram-in-colorbar**: Replaced the separate left-side histogram strip (`W` key toggle, `#hist-wrap`/`#hist-canvas`) with an integrated histogram that lives inside the colorbar. The colorbar is normally 8px tall; on mouse hover it smoothly expands to 40px, auto-fetches histogram data for the current slice, and draws colormap-colored bars growing upward from the gradient strip. Vertical vmin/vmax clim lines (yellow) appear when expanded and can be dragged horizontally (`ew-resize`). Hovering over the expanded colorbar shows a single-value tooltip. On mouse leave, the colorbar collapses back to 8px after a 200ms delay. The `W` key shortcut and help overlay entry have been removed. Histogram data is cached per slice and lazily re-fetched when the slice changes while expanded. Disabled in compare and RGB modes. Smoke: scenarios 54, 55.
- **lebesgue-mode**: `w` key toggles Lebesgue integral mode. When active, the colorbar stays expanded; hovering over a histogram bin highlights all canvas pixels whose raw data values fall in that bin. Non-matching pixels are dimmed with a semi-transparent dark overlay on `#lebesgue-canvas`. The hovered bin is outlined in yellow on the histogram. A new `/lebesgue/{sid}` endpoint returns the raw 2-D slice as float32 binary (fetched once per slice, cached client-side) for per-pixel bin lookup without server round-trips. Disabled in compare and RGB modes. Smoke: scenario 61.
- **compare-qmri**: Pressing `q` while in compare mode enters a combined compare-qMRI view: rows = compared arrays (up to 4), columns = parameter maps. Compact mode (T₁, T₂, |PD| only) is triggered by pressing `q` again when n>3; `q` once more exits. State: `compareQmriActive`, `compareQmriViews`, `compareQmriSids`, `compareQmriDim`. Render pipeline mirrors `qvRender` but uses each view's `sid`. Layout: `compareQmriScaleAllCanvases()` fits the grid into the viewport with column headers (`.cq-header-label`) and row labels (`.cq-row-label`). Per-pane slim colorbars via `drawCqSlimCb`. Smoke: scenario 62.
- **compare-multiview**: Pressing `v` (or `V`) while in compare mode enters a combined compare × 3-plane view: rows = compared arrays (up to 4), columns = Axial/Coronal/Sagittal planes (using `defaultMultiViewDims()`). `v` again exits back to compare mode. State: `compareMvActive`, `compareMvViews`, `compareMvSids`, `compareMvDims`. Render pipeline uses per-view `dimX`/`dimY`/`sliceDir` — same message format as `mvRender` but with per-row `sid`. Layout: `compareMvScaleAllCanvases()` mirrors `compareQmriScaleAllCanvases()`. Per-pane slim colorbars via `drawCmvSlimCb` (delegates to `drawQvSlimCb`). Canvas wheel scrubs the slice along `sliceDir`. `exitCompareMode()` calls `exitCompareMv()` if active. Zoom, resize, scrollbar, border key, and egg positioning all handle `compareMvActive`. Smoke: scenario 63.
- **ui-text-sizing**: Increased font sizes across the viewer: `#array-name` 11→13px; `#slim-cb-labels`/`#mv-cb-labels` 11→13px; `.qv-label`/`.cq-header-label` 14→16px; `.qv-cb-labels`/`.compare-pane-cb-labels` 9→11px. Row labels in compare-qMRI and compare-multiview (`.cq-row-label`) rotated 90° CCW (`writing-mode: vertical-rl; transform: rotate(180deg)`) to a narrow 24px-wide vertical strip, centered in the row. `.cq-header-row` `padding-left` removed (was a hack for the old horizontal label width). `.qv-row` `align-items` changed from `flex-start` to `center`.
- **movie-fps**: `[` and `]` keys change playback fps while Space is active (movie mode). Default fps changed from ~60 to 15. `playFps` variable controls `setTimeout` delay in `playNext()`. Status bar shows current fps. Smoke: scenario 64.
- **remove-cb-hover**: Colorbar no longer auto-expands on mouse hover. Histogram is only visible in Lebesgue mode (`w` key). Fixed mouseup cursor reset (default instead of ew-resize) and mouseleave tooltip hide. Smoke: scenarios 54, 55 updated to use `w` key.
- **cb-width-limits**: `drawSlimColorbar()` now clamps colorbar CSS width to `[120px, 600px]` (was unbounded — could be 1px for tiny arrays or fill the screen for huge ones). Smoke: scenario 65.
- **mv-axis-swap**: Fixed multi-view first pane showing wrong orientation. `enterMultiView()` and `enterCompareMv()` now put `{dimX:dims[2], dimY:dims[1], sliceDir:dims[0]}` as pane 0 (matches normal-mode orientation — largest two dims, standard axial-like view). Previous order had pane 0 as `{dimX:dims[1], dimY:dims[0]}` which had x<y causing a server-side transpose. Smoke: scenario 08 updated to check first pane axis labels.
- **axis-modes**: `p` key renamed to `a`; stretch-to-square now works in all modes (normal, compare, multi-view, compare-mv). Multi-view automatically enables squareStretch on entry and restores the previous value on exit. Help overlay updated. Smoke: scenario 66.
- **r-rotate**: `r` rotates view 90° CW when the slice dimension (not x or y) is the active dim. `dim_x` and `dim_y` swap, and `flip_y` is inverted to preserve orientation across 4 presses. Help overlay `r` row updated. Smoke: scenario 67.
