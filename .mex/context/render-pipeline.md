---
name: render-pipeline
description: Slice extraction, colormap LUTs, caching, render thread, and prefetch. Load when working on rendering, colormaps, or performance.
triggers:
  - "render"
  - "colormap"
  - "LUT"
  - "slice"
  - "cache"
  - "extract_slice"
  - "render_rgba"
  - "render thread"
  - "prefetch"
  - "RGBA"
  - "PNG"
edges:
  - target: context/architecture.md
    condition: when understanding where the render pipeline fits in the overall system
  - target: context/conventions.md
    condition: when writing new render functions or extending the pipeline
  - target: patterns/debug-render.md
    condition: when a render produces wrong output or the wrong cache is hit
last_updated: 2026-04-15
---

# Render Pipeline

## Pipeline Order

Every slice request follows this fixed order. Do not skip steps or reorder.

```
1. extract_slice(session, dim_x, dim_y, idx)
   → returns float32 2D numpy array (raw slice from ND array)
   → result cached in session.raw_cache (LRU, keyed by (dim_x, dim_y, key_idx))

2. apply_complex_mode(arr, mode)
   → applies FFT, magnitude, phase, real, imag transforms to complex arrays
   → skip if data is not complex

3. _prepare_display(arr, vmin, vmax, log_scale)
   → normalizes to [0, 255] uint8

4. render_rgba(arr_uint8, lut_name) / render_rgb_rgba(arr) / render_mosaic(...)
   → applies LUT (colormap) → returns RGBA uint8 2D array
   → result cached in session.rgba_cache or session.mosaic_cache

5. PNG encode (Pillow)
   → sent as binary WebSocket frame
```

## Colormap LUTs

`LUTS` dict in `_render.py` — maps colormap name → 256×4 uint8 numpy array (RGBA).

- Initialized lazily by `_init_luts()` on first use. Not initialized at import time.
- Available colormaps: `gray`, `lipari`, `navia`, `viridis`, `plasma`, `magma`, `inferno`, `cividis`, `rainbow`, `RdBu_r`, `twilight_shifted`, plus `RdBu_r_black` (diff mode).
- `lipari` and `navia` require `qmricolors` to be imported first (done inside `_init_luts()`).
- To add a colormap: add name to `COLORMAPS` list in `_session.py`, then add LUT construction in `_init_luts()` in `_render.py`. Restart required.

## Session Caches

Each `Session` has three LRU caches, all `OrderedDict`:

| Cache | Key | Holds | Budget env var |
|-------|-----|-------|---------------|
| `raw_cache` | `(dim_x, dim_y, key_idx)` | float32 2D slices | `ARRAYVIEW_RAW_CACHE_MB` (5% RAM) |
| `rgba_cache` | derived | RGBA uint8 tiles | `ARRAYVIEW_RGBA_CACHE_MB` (10% RAM) |
| `mosaic_cache` | derived | mosaic RGBA | `ARRAYVIEW_MOSAIC_CACHE_MB` (2.5% RAM) |

Cache budgets are computed once at import time from available RAM (via `psutil`). Override per-session via env vars.

LRU eviction pattern:
```python
session.raw_cache.move_to_end(key)   # on hit
session.raw_cache[key] = result      # on miss, then evict if over budget
while session._raw_bytes > session.RAW_CACHE_BYTES and session.raw_cache:
    _, v = session.raw_cache.popitem(last=False)
    session._raw_bytes -= v.nbytes
```

**`reset_caches()`** clears all three caches and resets byte counters. Call this on data version bump (e.g. after `/reload`).

## Render Thread

`_RENDER_QUEUE: SimpleQueue` in `_session.py`. The render thread (`arrayview-render`, daemon) drains the queue and resolves `asyncio.Future`s on the server event loop.

```python
await _render(loop, lambda: extract_slice(session, dim_x, dim_y, idx))
```

- Never use `asyncio.run_in_executor` for render work — the executor's `_global_shutdown` flag races with daemon thread cleanup.
- The queue accepts `None` as a sentinel to stop the thread.

## Prefetch Pool

Separate 1-thread `ThreadPoolExecutor` (`arrayview-prefetch`). Warms `raw_cache` for the next `PREFETCH_NEIGHBORS` (3) slices in the scroll direction. Budget-gated: skips if `slice_bytes * PREFETCH_NEIGHBORS > PREFETCH_BUDGET_BYTES` (16 MB). Errors are silently swallowed.

## Special Array Types

- **RGB arrays** — detected by `_setup_rgb()`. Skip colormap; go directly to `render_rgb_rgba()`. `session.rgb_axis` holds the axis index; `session.spatial_shape` is shape without that axis.
- **Complex arrays** — `apply_complex_mode()` transforms to real before colormapping. Supported modes: magnitude, phase, real, imag, FFT (various).
- **NIfTI** — canonical-reoriented on load by `_load_nifti_with_meta()`. `session.spatial_meta` holds affine, voxel sizes, axis labels. RAS resample cached in `session.resampled_volume`.
- **Large memmap arrays** — `_default_start_dims_for_data()` may override the default startup axes to avoid high-stride reads.

## High-Risk Areas

- **Cache key collisions** — the `key_idx` tuple replaces slice-dim positions with `None`. A bug here causes stale tiles to display for wrong indices.
- **LUT initialization race** — `_init_luts()` uses a double-checked lock. Do not call `LUTS[name]` directly before calling `_init_luts()`.
- **Dtype handling** — `extract_slice()` must return `float32`. RGB uint8 arrays must NOT go through the float pipeline. Inserting a new format type that returns a different dtype will silently corrupt renders.
