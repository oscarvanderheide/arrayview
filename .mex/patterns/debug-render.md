---
name: debug-render
description: Diagnosing visual rendering failures — wrong colors, blank canvas, artifacts, mosaic layout errors, colormap issues. Use when visual_smoke.py fails or a user reports visual bugs.
triggers:
  - "blank"
  - "wrong color"
  - "artifact"
  - "visual bug"
  - "render"
  - "visual_smoke"
  - "colormap"
  - "mosaic"
  - "overlay"
  - "colorbar"
  - "nan"
edges:
  - target: context/render-pipeline.md
    condition: always — full pipeline detail, cache structure, LUT initialization
  - target: context/architecture.md
    condition: for the rendering pipeline flow in the broader system
  - target: patterns/frontend-change.md
    condition: only after Step 1 points to `_viewer.html` rather than the Python pipeline
  - target: context/conventions.md
    condition: for the Verify Checklist after fixing
last_updated: 2026-04-15
---

# Debug Render

## Context

Rendering pipeline (Python side):
```
extract_slice(session, dim_x, dim_y, idx_list)   → float32 H×W
  ↓
apply_complex_mode(raw, complex_mode)              → float32 (mag/phase/real/imag)
  ↓
_prepare_display(session, raw, …)                  → float32 + vmin/vmax
  ↓
apply_colormap_rgba(session, raw, colormap, …)     → uint8 H×W×4 RGBA
  ↓ (WebSocket binary frame)
Frontend canvas: putImageData / ImageBitmap         → pixels on screen
```

For mosaic: `render_mosaic()` assembles a grid of slices, then applies colormap once to the whole grid.
For RGB arrays: `render_rgb_rgba()` bypasses colormap entirely.
For overlays: `_extract_overlay_mask()` + `_composite_overlay_mask()` runs after RGBA render.

Key cache layers:
- `session.raw_cache` — raw float32 slices (LRU, `RAW_CACHE_BYTES` budget)
- `session.rgba_cache` — RGBA uint8 frames (LRU, `RGBA_CACHE_BYTES` budget)
- `session.mosaic_cache` — mosaic frames (LRU, `MOSAIC_CACHE_BYTES` budget)

## Steps

**Step 1 — Reproduce in Python (bypass frontend)**
```python
from arrayview._session import Session
from arrayview._render import render_rgba
import numpy as np

s = Session(np.random.rand(64, 64))
rgba = render_rgba(s, 0, 1, (0, 0), "gray", 0)
print(rgba.shape, rgba.dtype, rgba.min(), rgba.max())
# Expected: (64, 64, 4) uint8 0 255
```
If this fails → bug is in `_render.py`. Proceed to Step 2.
If this succeeds → bug is in frontend WebSocket handling or canvas drawing. Skip to Step 5.

Pick one path first. Do not load this pattern and `patterns/frontend-change.md` together at the start unless the task genuinely spans both sides.

**Step 2 — Check for NaN/Inf in the slice**
```python
raw = extract_slice(s, 0, 1, [0, 0])
print(np.isnan(raw).any(), np.isinf(raw).any())
# NaN in source data → nan_to_num in extract_slice should have cleared it
```

**Step 3 — Check colormap LUT initialization**
```python
from arrayview._render import LUTS, _init_luts
_init_luts()
print("gray" in LUTS, LUTS["gray"].shape)
# Expected: True (256, 4)
```
If `LUTS` is empty → `_init_luts()` was not called before first render.

**Step 4 — Check vmin/vmax collapse**
If vmin == vmax, `apply_colormap_rgba` normalizes to zero → all pixels map to LUT[0] (black for gray).
```python
from arrayview._render import _compute_vmin_vmax
vmin, vmax = _compute_vmin_vmax(s, raw)
print(vmin, vmax)  # should not be equal
```

**Step 5 — Check WebSocket binary frame format**
In the browser console: look for `ArrayBuffer` size. Expected: `H * W * 4` bytes (RGBA, row-major).
If size is wrong → server is sending wrong shape. Add a print to the `/ws/{sid}` handler in `_server.py`.

**Step 6 — Check mosaic grid layout**
For mosaic artifacts: `mosaic_shape(n)` returns `(rows, cols)`. Verify `rows * cols >= n`.
Grid uses a 2-pixel gap between tiles filled with `[22, 22, 22, 255]` (dark gray). Unexpected color in gap → `nan_mask` not applied correctly.

**Step 7 — Check overlay compositing**
Run `_composite_overlay_mask(rgba, ov_raw, is_label=True)` on a test RGBA and mask. Check that pixels with `ov_raw == 0` are untouched.

## Gotchas

- **Cache stale after data change** — if `session.data` is modified without calling `session.reset_caches()`, stale slices will be served. Always call `reset_caches()` after modifying session data.
- **Complex dtype from `.mat`** — scipy structured dtypes with `real`/`imag` fields are converted in `extract_slice`, not in `_io.load_data`. If you bypass `extract_slice`, you must call `_fix_mat_complex()` yourself.
- **RGB arrays skip colormap** — `render_rgb_rgba` does NOT call `apply_colormap_rgba`. If your code assumes all renders go through colormap, it breaks for RGB sessions.
- **Mosaic gap pixels** — the grid is filled with `np.nan` first, then NaN pixels are colored `[22, 22, 22, 255]`. A NaN in actual data will look like a grid gap — use `extract_slice` which calls `nan_to_num`.
- **Colormap not in LUTS** — custom colormaps added after `_init_luts()` go through `_ensure_lut()`. Never access `LUTS[name]` without calling `_ensure_lut(name)` first.
- **`visual_smoke.py` uses Playwright** — requires `uv run playwright install chromium`. A missing browser causes a confusing import error, not a clear "browser not found" message.

## Verify

- [ ] `render_rgba(session, 0, 1, (0,)*ndim, "gray", 0).shape == (H, W, 4)` for a test session
- [ ] No `nan` or `inf` values after `extract_slice` (they are replaced by `nan_to_num`)
- [ ] `LUTS` is non-empty after calling `_init_luts()`
- [ ] `uv run pytest tests/visual_smoke.py` passes
- [ ] `uv run pytest tests/test_mode_consistency.py` passes

## Update Scaffold
- [ ] If a new failure mode was discovered, add it to the Gotchas section above
- [ ] Update `.mex/ROUTER.md` "Known issues" if this was a persistent bug that is now fixed
