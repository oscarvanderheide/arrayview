---
name: extend-compare-mode
description: Extending compare mode in `_viewer.html` — new compare entry paths, pane labeling, center-mode behavior, or compare-specific per-pane render state.
triggers:
  - "compare mode"
  - "compare center"
  - "detached compare"
  - "pane titles"
  - "diff params"
  - "X key"
edges:
  - target: patterns/frontend-change.md
    condition: when the work is primarily a local `_viewer.html` edit
  - target: context/render-pipeline.md
    condition: when compare changes require backend render or diff parameter changes
last_updated: 2026-05-13
---

# Extend Compare Mode

## Context

Compare behavior is split across a few tight surfaces:
- state near `compareCenterMode` and `compareSid`
- pane titles in `updateCompareTitles()`
- frame fetching in `compareRender()`
- center diff fetching in `fetchAndDrawDiff()` and `_fetchDiffHistogram()`
- command and keybinding routing near `compare.cycleCenterMode`
- compare teardown in `exitCompareMode()`

If the feature changes diff request parameters, keep HTTP and stdio parity by updating `_routes_rendering.py`, `_stdio_server.py`, and `_diff.py` together.

## Steps

1. Add new compare-specific state beside the existing compare globals, not in unrelated mode sections.
2. Reuse compare entry helpers when possible. If the feature behaves like compare but changes where panes read from, prefer a small wrapper entry function over a parallel mode implementation.
3. Update pane labeling in `updateCompareTitles()` so the titles reflect compare state directly.
4. If pane sources can diverge, update both `compareRender()` and the center diff/histogram fetch path to build per-pane request params.
5. Route keyboard behavior through the command registry and keybind list together. If a key changes meaning in compare mode, add a compare-scoped command before the generic fallback binding.
6. Clear all new compare state in `exitCompareMode()`.
7. Update `GUIDE_TABS` whenever the key behavior or compare affordances change.

## Gotchas

- `compareDisplaySids` can lag initial compare entry; if 2-pane support matters immediately, also account for `_compareExplicitSids`.
- A compare feature that changes `/diff` params must update `_fetchDiffHistogram()` too, not just `fetchAndDrawDiff()`.
- Detached or synthetic compare sources should not leak into saved `compare_sid` URL/session state unless restore explicitly supports them.
- `[` and `]` already mean “adjust compare parameter” in several center modes. New compare-local bindings must be ordered ahead of the generic bracket handler.

## Verify

- [ ] Focused browser test for compare entry or the new entry path passes
- [ ] Focused browser test for `X` / compare-center behavior passes
- [ ] If diff params changed, focused API coverage for `/diff` passes
- [ ] If diff histogram depends on the same params, verify histogram path too

## Update Scaffold

- [ ] Update `.mex/context/project-state.md` when compare behavior changes
- [ ] Add this pattern to `.mex/patterns/INDEX.md` if it is new
