# TODO

## Refactoring Opportunities (assessed 2026-04-03, swept 2026-04-07)

Items 1, 2, 3 done. Items 4 and 5 deliberately deferred — see notes.

### ~~1. Session cache reset helper~~ ✅ done (edfe6b2)
`Session.reset_caches()` extracted in `_session.py`. Replaced 9 sites across `_server.py`
and `_stdio_server.py`. Also fixed a latent bug where byte counters
(`_raw_bytes`/`_rgba_bytes`/`_mosaic_bytes`) were not being reset alongside the caches.

### ~~2. FastAPI session dependency injection~~ ✅ done (e4c6fd7)
Added `get_session_or_404` FastAPI dependency in `_server.py`. 28 endpoints converted
from manual `SESSIONS.get(sid)` boilerplate to `Depends(get_session_or_404)`. 12 sites
left intentionally because they have non-standard shapes (websocket close, JSON error
returns, compound guards, `PENDING_SESSIONS` poll loop, positive-form guards).

### ~~3. Extract compare-mode clip wrapper helper~~ ✅ done (1d4a26e)
`_ensureClipWrapper(inner, vpW, vpH)` extracted as a local helper inside
`compareScaleCanvases` in `_viewer.html`. 21 lines saved across the 3 duplicated blocks.

### 4. Compare layout manager class — DEFERRED
Adding a `CompareLayoutManager` class "for testability" without any test infrastructure
in place would be speculative abstraction. Revisit only if/when there's a concrete bug
or feature that demands the decomposition.

### 5. Split `_server.py` into domain modules — STILL DEFERRED (re-assessed 2026-04-07)
At 3265 lines (down from 3300 — items 1–3 shrunk it slightly), the original deferral
criteria are unmet: the file has not grown, and no feature work is currently blocked
by its size. With no test suite in place, relocating ~60 endpoints across new modules
while preserving shared `SESSIONS`/`app`/helper state is a high-risk restructure whose
only payoff is navigation ergonomics. Revisit only if the file grows past ~4000 lines
or if module boundaries become a concrete friction point during feature work.

## New candidates (spotted 2026-04-07)

### 6. Decompose `view()` in `_launcher.py`
`arrayview._launcher.view()` is a single **624-line function** (lines 701–1325) and
`_launcher.py` as a whole is **2820 lines** — now a bigger structural concern than
`_server.py`. `view()` has clear internal seams worth extracting as helpers:

  - arg validation + per-array kwarg normalization (name/rgb broadcast)
  - window-mode resolution (auto / native / browser / vscode / inline, env+config)
  - server bring-up and port discovery
  - array relay (local vs SSH vs VS Code tunnel)
  - ViewHandle construction and inline-vs-window dispatch

This is the public entry point touched by all six launch modes (see
`invocation-consistency` skill), so any refactor here needs manual verification across
CLI, Python script, Jupyter, Julia, VS Code tunnel, and native window paths. Not safe
to do unattended — deserves a focused session with screenshot verification.

### 7. `_launcher.py` as a whole — consider splitting
At 2820 lines it has clearer seams than `_server.py` does: webview open/fallback,
daemon server lifecycle, array relay (local/SSH/VS Code), CLI + config command,
subprocess view. A split here would have a much smaller blast radius than splitting
`_server.py`. Lower priority than #6, but worth considering after `view()` is
decomposed — the extracted helpers will make the natural module boundaries obvious.
