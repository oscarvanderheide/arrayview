# Lessons Learned

Hard-won knowledge from past development sessions. Check this before starting work on related areas.

## VS Code Extension / Simple Browser

**Problem:** Extension install and signal-file IPC breaks frequently when touched.
**What didn't work:** Changing `--force` install logic, modifying IPC hook detection without testing all paths.
**Solution:** Always load the `vscode-simplebrowser` skill before touching this area. Test on both local VS Code and tunnel. Never `--force` reinstall if the correct version is already on disk.

## Colorbar / Histogram Height

**Problem:** Histogram expanded height overflows in multi-view mode.
**Root cause:** `_computeCbExpandedH()` was using `slim-cb-wrap` (normal mode) in all modes. In multi-view, the active colorbar is `mv-cb-wrap`.
**Fix:** Use the correct wrapper ID based on mode. Always call `_computeCbExpandedH()` before expanding the histogram.

## Dynamic Island Positioning

**Problem:** Colorbar, eggs, and info elements use `position: fixed` and must be repositioned when switching modes (normal, multi-view, immersive, compare).
**Key insight:** Each mode has its own positioning logic. When adding features that affect layout, check all modes — not just the one being developed. Use `/ui-consistency-audit` skill.

## Auto-fit and Zoom State

**Problem:** `_fitZoom`, `userZoom`, `_zoomAdjustedByUser`, and `_autoFitPending` interact in subtle ways.
**Key insight:** `_fitZoom` is recomputed whenever `scaleCanvas` runs with `_autoFitPending = true`. When entering/exiting immersive mode, set `_autoFitPending = true` and `_zoomAdjustedByUser = false` before calling `setFullscreenMode()` so the layout recalculation uses the correct viewport size.

## Multi-view vs Normal Mode Colorbar

**Problem:** Two separate colorbar systems: `slim-cb-wrap` (normal/compare) and `mv-cb-wrap` (multi-view).
**Key insight:** Many colorbar functions need to check `multiViewActive || compareMvActive` to pick the right element. When adding colorbar features, always handle both paths. The `ColorBar` class abstracts some of this but the global state (`_cbExpanded`, `_cbAnimT`, etc.) is still shared.
