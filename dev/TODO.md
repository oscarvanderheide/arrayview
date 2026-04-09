# TODO

- ~~d in mosaic mode causes arrayview to crash.~~ ✅ fixed
- investigate the codebase for legacy/unused code. create a plan to clean it up. check with
  me first before doing the actual cleanup
- we need to brainstorm on zooming with ctrl or shift and trackpad (and whatever it is on
  other OSes). like two weeks ago i dont remember it was so fast that i asked to make it
  less sensitive. not its behaviour feels really sluggish and weird.
- when i press i to get value at current mouse position, it feels a bit laggy and also lags
  behind when i move around with mouse cursos
- when i hover over colorbar with mouse, the value should be shown in similar
  letters/style/colors as the value seen when pressing i for hover mode and hovering over
  the array.

- the help menu (?) is difficult to read for me. im not sure if its the combination of
  background and foreground colors (black and yellow), the font, or the font size, or the
  fact that its just a long list of keymaps. can we brainstorm on how to improve this? keep
  in mind that its mainly for exploration by new users. the viewer is designed to be
  keyboard centric and once users know the keybinds they need to use, they shouldnt really
  have to look at the help menu anymore. it feels a bit like a reference list rather than a
  guide at the moment. maybe i need both

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

## UI Maturity Strategy (2026-04-07)

A multi-phase plan to tame the combinatorial mode/feature/keybind bug class
("changed colorbar — works with 1 array but not 2"; "enter mode, change
something, exit, change is lost"; "feature works in mode A but not mode B"):

- **Strategy:** [`dev/plans/ui-maturity-strategy.md`](plans/ui-maturity-strategy.md)
- **Codebase report:** [`dev/plans/ui-maturity-codebase-report.md`](plans/ui-maturity-codebase-report.md)
- **External research:** [`dev/plans/ui-maturity-external-research.md`](plans/ui-maturity-external-research.md)

Four pillars (do not start a phase before the previous is green):
1. ~~**Phase 0** — round-trip pytest matrix~~ ✅ done (280809d)
2. ~~**Phase 1 (Pillar A)** — finish `collectStateSnapshot`; every mode enter/exit symmetric~~ ✅ done (a2f971c, 138307a)
3. ~~**Phase 2 (Pillar C)** — command registry + `when` clauses; kill the 1309-line keydown switch~~ ✅ done (2026-04-08; ~75 commits, 54 commands migrated, handler 1309→35 lines, help overlay auto-generated, `/` command palette, 90-test reachability matrix)
4. **Phase 3 (Pillar B step 1)** — finish `ColorBar` class migration (multi-view, qMRI) ← **next**
5. **Phase 4 (Pillar B step 2)** — collapse 5 scale functions into 1 + layout strategies
6. **Phase 5 (Pillar D step 3)** — Hypothesis stateful tests; optional Photoshop variant

Explicitly skipped: XState, React/Lit, renderer rewrite, hierarchical statecharts,
big snapshot matrix. See strategy doc for why.

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
