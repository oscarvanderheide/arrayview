# TODO

## Refactoring Opportunities (assessed 2026-04-03)

Ranked by impact on future maintainability. Legacy colorbar migration is complete (no cleanup needed).

### 1. Session cache reset helper (LOW effort, MEDIUM impact)
`_server.py` repeats `raw_cache.clear(); rgba_cache.clear(); mosaic_cache.clear()` in 5+ endpoints.
Extract `Session.reset_caches()` in `_session.py`. ~15 lines of boilerplate removed, single
source of truth for cache invalidation when adding new cache types.

### 2. FastAPI session dependency injection (MEDIUM effort, HIGH impact)
Every endpoint (30+) starts with `session = SESSIONS.get(sid); if not session: return 404`.
Replace with `Depends(get_session_or_404)` — removes ~30 lines of boilerplate, enables
per-session cross-cutting concerns (logging, rate limiting) in one place.

### 3. Extract compare-mode clip wrapper helper (LOW effort, MEDIUM impact)
`_viewer.html` lines ~2610-2660 duplicate clip-wrapper setup 3 times for diff/wipe/pane canvases.
Extract `_ensureClipWrapper(innerEl, sourceInner)` — ~40 lines saved, any clip fix applies everywhere.

### 4. Compare layout manager class (MEDIUM effort, MEDIUM impact)
`scaleCompareCanvases` is ~170 lines of monolithic layout logic. Extract `CompareLayoutManager`
with methods per concern (syncDiffCanvas, shrinkPanes, applyCenteringPan). Enables future
unit-testable layout logic.

### 5. Split `_server.py` into domain modules (HIGH effort, HIGH impact)
3,258 lines mixing vector fields, segmentation, ROI, analysis, export endpoints.
Split into `_server_core.py`, `_server_vectorfield.py`, `_server_segmentation.py`, etc.
Each module <400 lines. Clearer mental model, easier navigation and code review.
