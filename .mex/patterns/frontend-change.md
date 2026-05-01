---
name: frontend-change
description: Editing `_viewer.html`. Prefer exact code-local search first; load deeper frontend context only when the change crosses sections or modes.
triggers:
  - "_viewer.html"
  - "keyboard shortcut"
  - "layout"
  - "reconciler"
  - "GUIDE_TABS"
edges:
  - target: context/frontend.md
    condition: when the change spans multiple frontend sections or the target area is not already known
  - target: context/architecture.md
    condition: when understanding how the frontend connects to the server
  - target: context/conventions.md
    condition: for shared conventions and the Verify Checklist
  - target: patterns/debug-render.md
    condition: when the change produces wrong visual output
last_updated: 2026-05-01
---

# Frontend Change

## Fast Path

- If the target function/id/section is already known, do **not** load extra `.mex` files first.
- Run one exact `rg`, read one narrow local slice, then edit.
- Load `context/frontend.md` only if the change touches reconcilers, mode routing, command registry, or another unfamiliar cross-section area.

## Context

The frontend lives in `src/arrayview/_viewer.html`. No build step. No separate files.
Section separators are the navigation primitive:
- CSS: `/* ── Section Name ── */`
- JS: `// ── Section Name ──`

## Steps

1. If the desired behavior is still ambiguous, ask 2-3 plain-English clarification questions before reading git history, large diffs, or broad sections of `_viewer.html`.
2. Scope the affected modes/panes. If the user explicitly asked for a full visual check, or this is release validation, invoke `ui-consistency-audit`.
3. Navigate by exact identifier or section separator, not broad keyword sweeps.
4. Read only the local CSS/JS slice you need.
5. Make the change in place and preserve section separator style.
6. If adding a keyboard shortcut: update both the command/keybind registry and `GUIDE_TABS`.
7. If adding a new mode: register it in the `Mode Registry`.
8. Run narrow verification for the touched behavior.
9. If the change affects mode routing/layout behavior across modes, run the targeted mode test.
10. Only run the broader visual audit path when explicitly requested or doing release validation.

## Gotchas

- Never sweep `_viewer.html` broadly when a known symbol or section marker can get you there.
- Modes are exclusive; mode-entry work often needs explicit exit or reconcile logic.
- The help overlay is static via `GUIDE_TABS`; it is easy to forget.
- Some colorbars use `ColorBar`, some use legacy code. Do not mix styles within one local path.
- Reuse shared layout helpers for layout auto-pickers instead of adding new heuristics.
- Full visual audit is not the default path.

## Verify

- [ ] Section separator style matches: `/* ── Section Name ── */` (CSS) or `// ── Section Name ──` (JS)
- [ ] No new external JS/CSS files created — everything stays in `_viewer.html`
- [ ] Help overlay updated if a keyboard shortcut was added or changed
- [ ] Narrow verification for the touched behavior completed
- [ ] If full visual audit was requested, or this is release validation, `ui-consistency-audit` and `uv run python` on `tests/visual_smoke.py` pass
- [ ] If mode routing/layout behavior changed across modes, `uv run pytest` on `tests/test_mode_consistency.py` passes

## Debug

If `visual_smoke.py` fails: see `patterns/debug-render.md`.
If the mode doesn't activate: check Mode Registry — the `enter()` function must be registered by exact mode name string.
If layout breaks in one environment but not others: check mode-specific CSS rules; immersive/compact mode may override your styles.
