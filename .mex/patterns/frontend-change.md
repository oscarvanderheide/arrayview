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
  - target: context/architecture.md
    condition: when understanding how the frontend connects to the server
  - target: context/conventions.md
    condition: for the Verify Checklist
  - target: patterns/debug-render.md
    condition: when the frontend change causes visual artifacts
last_updated: 2026-04-13
---

# Frontend Change

## Context

The entire frontend lives in `src/arrayview/_viewer.html` (~15,600 lines). No build step. No separate files.

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

Skills to invoke first (before any frontend work):
- `ui-consistency-audit` — identifies all affected mode combinations before coding
- `modes-consistency` — ensures the feature works across all viewing modes
- `viewer-ui-checklist` — keeps `visual_smoke.py`, help overlay, and docs in sync
- `frontend-designer` — for styling/layout changes

## Steps

1. Invoke `ui-consistency-audit` skill to identify all affected mode combinations
2. Read the relevant CSS/JS section(s) in `_viewer.html` using `offset`/`limit`
3. Identify the exact section separator line(s) your change will touch
4. Make the change — edit in place, preserve section separator style
5. If adding a keyboard shortcut: add it to the **help overlay** section too
6. If adding a new mode: register it in the **Mode Registry**
7. Run `uv run pytest tests/visual_smoke.py` — visual smoke test
8. Run `uv run pytest tests/test_mode_consistency.py` — cross-mode check
9. If the feature is documented in `docs/`: update the relevant page

## Gotchas

- **Search by section separator** — the file is 15k lines. Never read the whole file. Always grep for the section separator first, then read 50–100 lines around it.
- **Modes are exclusive** — entering one mode must exit conflicting modes. Check `Mode Registry` for existing exit hooks.
- **ColorBar class migration** — some colorbars use the new `ColorBar` JS class; some use legacy inline code. Do not mix styles in the same colorbar. Check `project_colorbar_refactor.md` memory.
- **help overlay is not auto-generated** — it's a static list. Forgetting to update it leaves users with invisible shortcuts.
- **No hot reload** — changes require a browser refresh. The server does not push frontend updates.
- **`visual_smoke.py` runs Playwright** — requires `uv run playwright install chromium` first. See `context/setup.md`.
- **CSS variable names** — dark theme palette uses `--av-*` custom properties. Do not introduce new one-off colors; use or extend the existing palette.

## Verify

- [ ] Section separator style matches: `/* ── Section Name ── */` (CSS) or `// ── Section Name ──` (JS)
- [ ] No new external JS/CSS files created — everything stays in `_viewer.html`
- [ ] Help overlay updated if a keyboard shortcut was added or changed
- [ ] `uv run pytest tests/visual_smoke.py` passes
- [ ] `uv run pytest tests/test_mode_consistency.py` passes
- [ ] New colors use `--av-*` CSS custom properties from the theme section

## Debug

If `visual_smoke.py` fails: see `patterns/debug-render.md`.
If the mode doesn't activate: check Mode Registry — the `enter()` function must be registered by exact mode name string.
If layout breaks in one environment but not others: check mode-specific CSS rules; immersive/compact mode may override your styles.

## Update Scaffold
- [ ] Update `.mex/ROUTER.md` "Current Project State" if what's working/not built has changed
- [ ] Update any `.mex/context/` files that are now out of date
- [ ] If this is a new task type without a pattern, create one in `.mex/patterns/` and add to `INDEX.md`
