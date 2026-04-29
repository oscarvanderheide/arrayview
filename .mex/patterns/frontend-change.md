---
name: frontend-change
description: Making any change to the viewer frontend (_viewer.html) — CSS, JS, new mode, keyboard shortcut, layout, colorbar. Use when touching the frontend.
triggers:
  - "_viewer.html"
  - "frontend"
  - "canvas"
  - "CSS"
  - "keyboard shortcut"
  - "colorbar"
  - "layout"
  - "mode"
  - "UI"
edges:
  - target: context/frontend.md
    condition: always — mode matrix, reconcilers, command registry, View Component System
  - target: context/architecture.md
    condition: when understanding how the frontend connects to the server
  - target: context/conventions.md
    condition: for shared conventions and the Verify Checklist
  - target: patterns/debug-render.md
    condition: when the change produces wrong visual output
last_updated: 2026-04-29
---

# Frontend Change

## Context

The entire frontend lives in `src/arrayview/_viewer.html` (~24,700 lines). No build step. No separate files.

Structure:
- **CSS** (lines ~7–1500): sections separated by `/* ── Section Name ── */`
- **JS** (lines ~1500–14750): sections separated by `// ── Section Name ──`

Key JS sections to know:
- **Constants and Transport Setup** — WS URL, stdio/postMessage transport abstraction
- **Viewer State Variables** — all mutable state (`currentSliceIndices`, `zoomLevel`, mode flags)
- **Mode Registry** — mode name → `enter()`/`exit()` function mapping; register new modes here
- **PanManager** — canvas panning state machine
- **ColorBar class** — shared JS class for all colorbar rendering; being migrated (partially done)
- **Help overlay** — keyboard shortcut list; must be updated when adding shortcuts

Skills to consider:
- `ui-consistency-audit` — full cross-mode audit when explicitly requested, when debugging a broad visual regression, or during release validation
- `modes-consistency` — use if the change affects mode interactions beyond a narrow UI tweak
- `viewer-ui-checklist` — release/explicit sync for `visual_smoke.py`, help overlay, and docs
- `frontend-designer` — for styling/layout changes

## Steps

1. If the desired behavior is still ambiguous, ask 2-3 plain-English clarification questions before reading git history, large diffs, or broad sections of `_viewer.html`.
2. Scope the affected modes/panes for the specific change. If the user explicitly asked for a full visual check, or this is release validation, invoke `ui-consistency-audit`.
3. Read the relevant CSS/JS section(s) in `_viewer.html` using `offset`/`limit`
4. Identify the exact section separator line(s) your change will touch
5. Make the change — edit in place, preserve section separator style
6. If adding a keyboard shortcut: add it to the **help overlay** section too
7. If adding a new mode: register it in the **Mode Registry**
8. Run narrow verification for the touched behavior (manual check, focused test, or targeted scenario)
9. If the user explicitly asked for a full visual check, or this is release validation, run the broader audit path (`ui-consistency-audit`, then `uv run python` on `tests/visual_smoke.py`, plus screenshots as relevant)
10. If the change affects mode routing/layout behavior across modes, run `uv run pytest` on `tests/test_mode_consistency.py`
11. If the feature is documented in `docs/`: update the relevant page

## Gotchas

- **Search by section separator** — the file is 15k lines. Never read the whole file. Always grep for the section separator first, then read 50–100 lines around it.
- **Modes are exclusive** — entering one mode must exit conflicting modes. Check `Mode Registry` for existing exit hooks.
- **ColorBar class migration** — some colorbars use the new `ColorBar` JS class; some use legacy inline code. Do not mix styles in the same colorbar. Check the `ColorBar class` section in `context/frontend.md` and the nearby `_viewer.html` section you are editing.
- **Layout auto-pickers must share viewport profiling** — if you touch compare-center, multiview/ortho, or any other mode with multiple layout presets, reuse `_layoutViewportProfile()` and `_supportsWidePrimaryLayout()` instead of adding a new per-mode viewport heuristic from scratch.
- **help overlay is not auto-generated** — it's a static list. Forgetting to update it leaves users with invisible shortcuts.
- **No hot reload** — changes require a browser refresh. The server does not push frontend updates.
- **Stale daemon trap** — `uv run arrayview ...` can leave `_serve_daemon` orphaned on port 8000 after the launching terminal exits. If the browser still shows old frontend code after a refresh, check `lsof -nP -iTCP:8000 -sTCP:LISTEN`, kill the old daemon PID, then launch again before trusting browser verification.
- **Full visual audit is not the default path** — use targeted verification during development unless the user explicitly asks for the broader screenshot/audit pass or you're validating for release.
- **`visual_smoke.py` runs Playwright** — requires `uv run playwright install chromium` first. See `context/setup.md`.
- **CSS variable names** — dark theme palette uses `--av-*` custom properties. Do not introduce new one-off colors; use or extend the existing palette.
- **`body.av-loading` hides real content** — `#canvas-wrap` can be layout-visible while still fully hidden. Clear `av-loading` on the first rendered frame; do not add artificial dwell around the overlay.
- **Shared slim colorbar is already fixed-position in normal mode** — for immersive scrub, keep it in place and lower its z-index to 1; simultaneously raise `#canvas-wrap` to `position: relative; z-index: 2` so the growing pane paints above it. Clear both on `_resetImmersiveTransforms`. Do not snapshot/re-pin the cb position or create a fixed clone — causes left drift and black rectangle artifacts.
- **Choose one path first** — if this looks like a frontend-only bug, stay in this pattern. Only jump to `patterns/debug-render.md` after you have ruled out a local `_viewer.html` fix.

## Verify

- [ ] Section separator style matches: `/* ── Section Name ── */` (CSS) or `// ── Section Name ──` (JS)
- [ ] No new external JS/CSS files created — everything stays in `_viewer.html`
- [ ] Help overlay updated if a keyboard shortcut was added or changed
- [ ] Narrow verification for the touched behavior completed
- [ ] If full visual audit was requested, or this is release validation, `ui-consistency-audit` and `uv run python` on `tests/visual_smoke.py` pass
- [ ] If mode routing/layout behavior changed across modes, `uv run pytest` on `tests/test_mode_consistency.py` passes
- [ ] New colors use `--av-*` CSS custom properties from the theme section

## Debug

If `visual_smoke.py` fails: see `patterns/debug-render.md`.
If the mode doesn't activate: check Mode Registry — the `enter()` function must be registered by exact mode name string.
If layout breaks in one environment but not others: check mode-specific CSS rules; immersive/compact mode may override your styles.

## Update Scaffold
- [ ] Update `.mex/ROUTER.md` "Current Project State" if what's working/not built has changed
- [ ] Update any `.mex/context/` files that are now out of date
- [ ] If this is a new task type without a pattern, create one in `.mex/patterns/` and add to `INDEX.md`
