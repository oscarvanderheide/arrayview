---
name: project-state
description: Current working, in-progress, and shelved status. Load only when the task depends on active workstreams or recent shipped behavior.
triggers:
  - "current state"
  - "what's in progress"
  - "recent work"
  - "active feature"
  - "shipped recently"
last_updated: 2026-08-03
---

# Project State

## Working

- No-argument CLI startup opens a generated, action-gated interactive tutorial.
  It presents as a single whisper line — one key named at a time, with no panel,
  counter, or progress bar — and never says what a key does before it is
  pressed; the explanation replaces the invitation only once the viewer state
  changed. Its real scalar, comparison, and overlay sessions cover navigation,
  playback, colormaps, histograms, ortho, split and multi-array comparison,
  overlays, ROI analysis, and the help map. `Esc` ends it.
- The tour is paced to the reader, not to a clock: a new line is only put on
  screen once input has gone quiet and nothing is covering the frame, so
  exploring after a step is never cut short and an echo can never play out
  behind the panel that step just opened.
- The tour is divided into 13 named sections, each announced before it asks for
  anything. A rail along the bottom is the one piece of tutorial furniture and
  the one thing clickable; `Tab`/`Shift+Tab` also switch. Every section declares
  the viewer state it needs and stages it on entry, so sections are real entry
  points and can be jumped to in any order.
- Two sections run on their own sessions because they cannot share the main
  one: a vector field disables statistical projections for whatever session it
  is attached to, and a ragged collection is a single session built from
  differently shaped files. `_configure_tutorial_args` generates them into a
  *separate* temp directory — the main bundle is deleted by the spawned daemon
  before the parent regains control — and `_register_tutorial_extras` registers
  them best-effort and publishes their sids as `tutorial_flow_sid` /
  `tutorial_stack_sid` on the viewer URL. Sections whose sid is absent are
  dropped from the tour rather than offered and stalled on.
- Switching sessions is a navigation (the viewer has no in-place primary-session
  swap), so the tour stores the main session's query under
  `arrayview:tutorial:home` and parks its step index before leaving.
- Tutorial step key labels are literal keymap characters (`v` and `V` are
  different commands). `tests/test_tutorial_browser.py` cross-checks every step
  against `keybinds`; do not hand-edit a label without that check passing.
- Public CLI and Python entry points exist and have focused component coverage. Launch convergence remains active work; do not describe an invocation as stable without current real-host first-frame, repeat-launch, and cleanup evidence.
- Linux CLI file launches classify source mounts from procfs before target
  access. Known-disconnected CIFS sources fail before server startup; healthy
  network files are staged through a bounded helper and served from a local
  snapshot. A per-mount guard prevents concurrent helpers and remains fenced to
  an unreaped helper PID after a timeout, then keeps a short cooldown after that
  exact helper exits, so retries cannot accumulate more ArrayView workers
  against that mount. Direct network collection scans and
  network `--watch` polling are rejected. This prevents ArrayView-owned
  long-lived servers from retaining the source path, but cannot reap a task
  already stuck in kernel `D` state.
- Local server reuse is Unix-user-fenced: `/ping` publishes the server UID,
  callers only reuse a compatible server owned by their effective UID, and a
  different user's listener causes local launch planning to select a new port.
  Runtime registry paths prefer `$XDG_RUNTIME_DIR/arrayview` and otherwise use
  `/tmp/arrayview-<uid>` with private registry and log directories.
- Real-host launch evidence has passed for local native CLI, Python-script native, Julia/PythonCall native, a real ipykernel inline view, and plain SSH with `ssh -L`. VS Code local, Remote SSH, and VS Code tunnel are separate acceptance rows and must be reverified independently after opener or routing changes.
- File formats: `.npy`, `.npz`, `.nii`/`.nii.gz`, `.dcm`/DICOM series directories, `.zarr`, `.h5`/`.hdf5`, `.mat`, `.tif`/`.tiff`, `.pt`/`.pth`
- DICOM series use physical patient-position ordering, canonical RAS geometry,
  modality rescale, explicit multi-series selection, and a privacy-filtered
  MRI-aware Shift+I tab with header/derived provenance.
- Browser drops use staged inspection: file and folder uploads create no session
  until Compare, Open separately, or Overlay is chosen. Compatibility reasons,
  opaque DICOM series selection, cancellation, expiry, and final-release cleanup
  share one server contract across browser-backed display environments.
- Rendering pipeline: colormaps, complex modes, mosaic, RGB/RGBA, projections, overlays
- Backend transport: FastAPI HTTP/WebSocket is the single viewer transport; shared helpers keep route modules small for metadata/analysis, compare/diff, overlay compositing, and vector field layout/arrow sampling.
- NIfTI spatial metadata, RAS resampling
- Directory collections are header-scanned and lazy by default: compatible files form a dense virtual stack, mixed shapes use a ragged collection, and `--load lazy|eager` plus `--stack-policy auto|dense|ragged` make both choices explicit. Supported 3D `.nii.gz` stacks now return the requested axial plane while the same one-pass decode finishes in the byte-bounded LRU cache; unsupported layouts fall back safely. Patient changes show a centered loading card until the matching frame arrives. `view_dir()` exposes the same collection controls.
- The working tree targets VS Code opener v0.15.15. It removes the public-webview
  tunnel path and the zero-viewer singleton gate: tunnel requests use
  request-scoped private integrated-browser direct-loopback delivery, with
  exact request/window/backend identity and first-frame acknowledgement.
  Browser commands are serialized narrowly while readiness waits remain
  concurrent. Each request now issues exactly one integrated-browser open
  command: delayed script startup waits inside the bounded readiness deadline
  instead of issuing retries that VS Code turns into duplicate tabs. If the
  private proxy or target cannot be verified, delivery fails closed instead of
  promoting the port.
- Component coverage guards private selection, concurrent command
  serialization, display intent, stale public-setting cleanup, and readiness
  correlation. The desktop-tunnel CLI path reached its first rendered frame
  with v0.15.12 and now keeps the visible integrated-browser URL short by
  resolving the full launch query behind a request-scoped backend route.
  Browser-hosted tunnel, middle-close isolation, reconnect and cleanup,
  explicit external browser, and proxy-disabled clean failure remain open.
- The first v0.15.7 real-host sequence produced four direct-loopback frames in
  about 0.6–0.9 s and a successful concurrent-webview fallback. One
  `large_array.npy` request failed before browser open because
  `launch-prepared` incorrectly waited for the data Session and exhausted its
  1.5 s bookkeeping budget. Viewer journals now live in SID-keyed
  control-plane state: pending registration is sufficient for preparation,
  later phases still require the loaded Session, and final release owns journal
  cleanup. The focused pending-preparation and cleanup component gates pass;
  the post-fix large-array real-host row remains open.
- Colorbar refactor: `ColorBar` JS class partially migrated (in progress)
- Colormap picker: `c` opens an expanded colorbar-island grid without changing the colormap; subsequent `c` taps cycle, hover/hjkl/arrows live-preview, Enter/click commits, Esc cancels, and auto-dismiss pauses while hovered
- Cold-start loading spinner in VS Code and native shell
- Tool menu (`/` menu) supports multi-select where allowed: spacebar toggles tools, Enter applies selection. Mutual exclusion enforced (ROI ↔ Segmentation, and overlay/vectorfield ↔ everything else). Cursor indicator shows focused tile via yellow background + left accent bar
- Dynamic island renders sections for active plugins such as qMRI and segmentation, replacing the old single-plugin priority chain
- ROI mode is a lightweight measurement tool: `Shift+R` shows/hides session ROIs, defaults to true circles, uses slim colorbar controls for shape/stats/hide/clear, and manages stats/rename/delete/export from the ROI manager modal
- ROI mode works alongside qMRI: drawing on any pane mirrors the ROI to all panes in real-time via per-pane overlay canvases; stats are fetched per visible parameter map and shown in the ROI manager
- qMRI map toggle (`_islandToggleQmriMap`) fade animation now covers dimbar and array-name in addition to panes and colorbars
- Segmentation menu shares the ROI layout (yellow accent, magnifier action icon, common `#export-overlay` modal). Pre-activation shows a pulsing "nnInteractive · connecting" row that morphs into the normal shape toolbar once `/seg/activate` resolves
- Overlays are a plugin tile (`OV`): per-overlay row with colour swatch, editable label, eye toggle, × delete; shared opacity slider; `+ add overlay` opens a filesystem picker rooted at the launched file's directory
- The floating overlay HUD is reconciled as soon as startup loading ends, so it
  is visible on first load and `/ o` toggles it correctly from the first press.
  Its drag grip allows viewport-clamped repositioning that survives mode and
  layout reconciliation for the current viewer session. Hovering a visible row
  focuses that mask and dims the other visible overlays across normal, compare,
  multiview, and qMRI render paths. The HUD and overlay drawer can switch masks
  between filled regions and outline-only contours.
- Vector field is a plugin tile (`VF`): row with visibility toggle + density/length sliders bi-directionally wired to the `[ ] { }` keyboard commands
- CLI `--overlay FILE` can be repeated to load multiple overlays at launch;
  unnamed overlays use the filename stem and `NAME=FILE` supplies an explicit
  HUD label.
- Stack collections support `--overlay-dir PATTERN` to discover per-case mask
  filenames as sparse overlay roles; missing masks render as empty and common
  patient/modality directory layouts pair automatically. `--case-regex` remains
  an override for unusual layouts. CLI header scans show an in-place file counter
  and elapsed time on interactive terminals.
- Reused file and collection sessions are protected by per-tab leases, including
  collection overlay sessions. The VS Code opener verifies the local SID before
  creating a panel and waits for the viewer's first rendered frame before
  acknowledging `backend_ready`.
- Filesystem picker endpoint (`GET /fs/list`) clamped to `$HOME`. Accepts `base_sid` + `mode` (`overlay` | `vectorfield`) to filter entries by a cheap header-shape peek (`.npy`, `.nii`/`.nii.gz`, `.h5`, `.zarr`); overlay mode requires identical shape, vectorfield mode requires base shape plus one axis of size 3
- Island collapse affordance: inline `~` at the island's top-right animates the panel into the bottom-left `~` hint circle; external `~` hint only visible while the island is actually collapsed. New `/` hint circle at bottom-right opens the tool menu
- Compare center tool menu: `/` now re-opens the last-used compare center mode from the tool menu, while compare pane header buttons select diff / overlay / wipe directly. Eligible two-array compare layouts can switch into a big-left arrangement with a wider center pane, and the diff colorbar now matches that center-pane width.
- Compare center + ortho auto-layout: layout choice is now auto-picked once on entry from a shared viewport-profile helper, stays stable across resize, and becomes sticky only after manual `G` / `g` override. Compare big-left also now supports in-pane A/B source badges and a shared source colorbar parked in the gap between the stacked source panes.
- Detached compare-on-X: with a single array and a non-spatial active dimension, `X` now enters compare mode by treating two indices from that dimension as A/B sources. Pane titles switch to index-over-total labels, the dimbar marks the detached dimension with a purple `X`, `[` / `]` scrub the left pane index, `{` / `}` scrub the right pane index, and repeated `X` presses cycle the existing compare-center modes before exiting back to normal view. The split-index diff path works through the FastAPI/WebSocket transport.

## Recently Completed

- Persistent preferences menu: the lower-right gear edits allowlisted
  `~/.arrayview/config.toml` values through a session- and server-fenced API.
  Viewer theme, rounded panes, ortho layout, and dimbar mode apply immediately;
  launch-window defaults are labeled for the next launch and remain scoped to
  the machine running the backend.
- Jupyter inline viewports automatically fit ortho content and restore the
  normal `height` on exit. `mode_heights` remains available for explicit
  overrides. Direct, proxied, multi-array, and IJulia inline paths share the
  same mode-change bridge.
- ROI keybind changed from `r` to `Shift+R` (`R`); rotate/flip is now `r`, transpose is `t`
- Crossfade animation added to rotate/transpose transitions
- `--version` flag and version string in help overlay
- In-viewer array picker for multi-array `.mat` and `.npz` files
- ROI manager modal for qMRI canvases (per-parameter-map stats rows, CSV export, label-mask export)
- Invocation lifecycle contracts and focused component tests cover transient local ownership, URL session release, quick connect/disconnect races, and fail-closed ambiguous window routing. These are safeguards, not proof that every real launch path works.
- Hover info wrong values fixed
- Native launcher startup restored
- qMRI pane sizing stabilized
- VS Code extension Windows support (select/pipes, venv path, ppid)
- Tool launcher motion refined

- Shift+C colormap picker redesign: the old centered shortlist is now a narrow translucent right-edge drawer with a close button, a yellow `Colormaps` title plus a `Favorites` subtitle, and a plain 12-swatch two-column quick set that stays visible above the search field. Search matches render in a separate results area below the input, Enter first exits the search field before a second Enter commits, arrow-key movement follows the visible two-column grid, and repeated `c` presses cycle through the currently visible swatches while the picker is open.
- Detached compare-on-X: single-array non-spatial dimensions now support the same compare-center family as two-array compare. The frontend reuses compare mode with per-pane detached indices, the dimbar shows a purple `X`, the compare titles show index-over-total labels, `[` / `]` control pane A, `{` / `}` control pane B, and repeated `X` exits detached compare after cycling the center modes. Focused coverage now includes a browser regression for detached entry/scrubbing/exit plus API coverage for split `/diff` indices on the same session.
- Normal-mode dimbar readability: inactive non-spatial dims no longer get a blanket reduced parent opacity, so the current index reads bright while `/total` stays subdued via the existing child dim-size styling.
- Multiview colorbar spacing now matches normal mode: entering ortho with `v` and switching to orthogonal `big-left` with `g` no longer increases the pane-to-colorbar gap. Focused browser coverage now compares the normal-view gap against both multiview layouts directly.
- V-mode ortho layout cycling is now a two-state toggle: `g` only switches between `horizontal` and `big-left`. The old `vertical` and `big-top` multiview presets have been removed from the shared ortho preset table, the hold-`g` picker/help text, and the remaining big-pane promotion branches, and the shared multiview colorbar width now stays fixed when `g` toggles between the two surviving layouts.
- The old snapshot gallery sidecar has been removed from the viewer. Saving a screenshot still downloads the PNG, but `Shift+G` no longer toggles any screenshot/gallery UI outside compare mode; `G` is now compare-layout cycling only.
- Loupe activation no longer arms on a plain left-button hold. Normal view and multiview panes now use `Ctrl` hover with no mouse button, so plain drag stays available for pane navigation and other mode-specific gestures. While `Ctrl` is held, wheel input keeps its normal slice-scroll behavior instead of being swallowed or repurposed for zoom, and the loupe now redraws when those wheel-driven slice changes land. Focused browser coverage now checks both the normal-view and multiview `Ctrl` hover paths plus `Ctrl`-wheel scrolling and loupe-refresh behavior. qMRI loupe wiring was updated to match, but direct qMRI browser validation is currently blocked by the qMRI reusable-URL browser test in `tests/test_browser.py`.
- Jupyter inline notebooks now default to `height=600`, start in normal mode instead of auto-immersive, and top-align the viewer cluster instead of vertically centering it inside the output cell. This removes the wasted band above the dimbar / below the colorbar while preserving manual `Shift+F` immersive entry. Browser coverage now checks the non-immersive inline start state and low wrapper top padding.
- Architecture followthrough: `_server.py` route extraction is complete. Feature domains now live in `_routes_analysis.py`, `_routes_loading.py`, `_routes_persistence.py`, `_routes_segmentation.py`, `_routes_state.py`, `_routes_query.py`, `_routes_export.py`, `_routes_preload.py`, `_routes_vectorfield.py`, `_routes_rendering.py`, and `_routes_websocket.py`. `_server.py` is now the intended assembly surface: FastAPI app setup, shared dependency injection, HTML/template helpers, and the small infrastructure routes for health/UI/assets/colormap metadata.
- Focused API coverage now directly guards segmentation activate/scribble/click-accept/export paths, export/preload/vectorfield routes, slice/projection/diff/grid/gif rendering, large-array grid/gif guardrails, and websocket metadata plus shell-close cleanup.

## In Progress

- VS Code tunnel private-only delivery and short integrated-browser URLs are
  implemented. A real v0.15.14 desktop-tunnel launch reached `frame-rendered`,
  but its delayed script startup exposed four tabs from one request. Opener
  v0.15.15 removes those repeated browser-open commands and has component
  coverage; real-host validation still requires installing/reloading the active
  tunnel window. Middle-close isolation, reconnect, clean final release,
  explicit external-browser delivery, proxy-disabled fail-closed behavior, and
  the separate browser-hosted tunnel row also remain open.
- Smooth immersive transition — stale scrub geometry handoff is fixed, immersive overlay fade-in is held until after the class switch, shared slim colorbar returns through `drawSlimColorbar()` on reverse, and active scrub suppresses minimap/overflow/drag side effects. Single-pane scrub now targets the actual centered immersive viewport rect instead of a hardcoded corner box, the dimbar stays above the pane during scrub, the shared colorbar sits behind the growing pane, and the phantom extra `av-view-wrap` footprint in normal mode was removed by rebinding `NormalLayout` to the real `#viewer` canvas. Cross-mode parity and deeper reverse-pinch validation still need manual verification.
- ROI + qMRI integration refinements: floodfill not yet supported on qMRI panes; per-pane stats are re-fetched on each ROI draw but not updated on slice scroll

## Not Yet Built

- Independent split view for mismatched-shape arrays (designed, shelved)
