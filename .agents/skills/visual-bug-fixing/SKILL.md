---
name: visual-bug-fixing
description: Use when fixing a real visual bug, layout glitch, rendering artifact, or UI regression in arrayview. Requires visual evidence, but keeps the default path targeted unless the bug has broad cross-mode risk.
---

# Visual Bug Fixing

## Rule

Fix visual bugs with evidence, not guesswork. Capture the broken state, fix the cause, capture the result, then run only as much regression coverage as the risk justifies.

## Use This When

- The user reports a visual bug
- You see a layout/rendering regression while working
- A screenshot-based test or `tests/ui_audit.py` fails

Do not use this for brand-new feature design or docs/test sync work.

## Targeted Path

For a small bug in one known area:

1. Capture one baseline screenshot of the broken state
2. Inspect the owning code path
3. Apply the smallest real fix
4. Capture one post-fix screenshot
5. Check the most likely neighboring mode(s)

Escalate to the broader audit path only when the bug touches shared layout, shared chrome, zoom, or multiple modes.

## Where To Look

| Area | Owner |
|------|-------|
| UI structure, CSS, JS layout | `_viewer.html` |
| Rendering routes | `_server.py` + `_routes_rendering.py` |
| Render/compositing helpers | `_render.py`, `_overlays.py` |
| Colorbar behavior | `ColorBar` class in `_viewer.html` |

Do not treat `_app.py` as the implementation surface; it is a compat shim.

## Diagnose First

- CSS positioning / overflow / z-index problem?
- Layout calculation bug in a mode-specific scale function?
- Missing branch for compare, multiview, or qMRI?
- Zoom-specific issue?
- Animation/timing issue? If yes, use the animation verification pattern instead of static screenshots alone.

## Common Fix Patterns

- Overlap bug: inspect bounds and z-order
- Wrong size: trace the relevant scale/layout function
- Mode-only regression: find the missing mode branch
- Render mismatch: inspect route params and backend render helpers

## Broader Audit Path

Use this when the bug has shared-chrome or cross-mode risk:

```bash
uv run python tests/ui_audit.py --tier 1
```

Go to tier 2 only when the affected feature or regression area actually needs it.

## Red Flags

- The fix works only in normal mode
- The patch hides the symptom instead of fixing placement/ownership
- Verification skipped screenshots entirely
- `_app.py` was edited for new behavior

## Verification

- [ ] Baseline screenshot captured
- [ ] Post-fix screenshot captured
- [ ] Nearby modes checked based on risk
- [ ] `tests/ui_audit.py --tier 1` run if the bug touched shared layout/chrome/cross-mode behavior
