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
last_updated: 2026-07-01
---

# Render Pipeline

## Pipeline Order

Every slice request follows this fixed order. Do not skip steps or reorder.

```
1. extract_slice(session, dim_x, dim_y, idx)
   → returns float32 2D numpy array (raw slice from ND array)
   → result cached in session.raw_cache (LRU, keyed by (dim_x, dim_y, key_idx))

2. _prepare_display(session, raw, complex_mode, dr, log_scale, vmin_override, vmax_override)
   → calls apply_complex_mode(raw, complex_mode) internally (mag/phase/real/imag)
   → returns (float32 data, vmin, vmax) — NOT uint8; normalization to [0,255]
     happens later inside apply_colormap_rgba

3. render_rgba(session, ...) / render_rgb_rgba(session, ...) / render_mosaic(session, ...)
   → each takes a session + dims/idx/colormap params (NOT a pre-extracted array),
     calls extract_slice internally, then applies LUT → RGBA uint8 2D array
   → result cached in session.rgba_cache or session.mosaic_cache

4. WebSocket binary frame
   → raw RGBA bytes prefixed by a binary header (seq, w, h as uint32 +
     vmin, vmax as float32). Pillow PNG encoding is ONLY used by the
     HTTP export endpoints (e.g. /grid/{sid}), not the live WS render path.
```

Note: FFT is NOT part of `apply_complex_mode`. It is a separate data-level
transform applied in the `/fft/{sid}` endpoint (`_routes_state.py`), which
mutates `session.data` and stashes the original in `session.fft_original_data`.

## Colormap LUTs

`LUTS` dict in `_render.py` — maps colormap name → 256×4 uint8 numpy array (RGBA).

- Initialized lazily by `_init_luts()` on first use. Not initialized at import time.
- Available colormaps: `gray`, `lipari`, `navia`, `viridis`, `plasma`, `magma`, `inferno`, `cividis`, `rainbow`, `RdBu_r`, `twilight_shifted`, `turbo`, plus `RdBu_r_black` (diff mode).
- `lipari` and `navia` require `qmricolors` to be imported first (done inside `_init_luts()`).
- To add a colormap: add the name to `COLORMAPS` in `_session.py`. `_init_luts()` iterates `COLORMAPS` and builds each LUT from `matplotlib.colormaps[name]` automatically, so standard matplotlib colormaps need no extra work. Only a custom (non-matplotlib) colormap like `RdBu_r_black` needs manual LUT construction in `_init_luts()`. Env vars are read once at import, so a restart is required.

## Session Caches

Each `Session` has three LRU caches, all `OrderedDict`:

| Cache | Key | Holds | Budget env var |
|-------|-----|-------|---------------|
| `raw_cache` | `(dim_x, dim_y, key_idx)` | float32 2D slices | `ARRAYVIEW_RAW_CACHE_MB` (5% RAM) |
| `rgba_cache` | derived | RGBA uint8 tiles | `ARRAYVIEW_RGBA_CACHE_MB` (10% RAM) |
| `mosaic_cache` | derived | mosaic RGBA | `ARRAYVIEW_MOSAIC_CACHE_MB` (2.5% RAM) |

Cache budgets are computed once at module import time from available RAM (via `psutil`), with a 64 MB floor. The env vars below are read once at import — there is no per-session override; changing a value requires a process restart.

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

`_RENDER_QUEUE: SimpleQueue` in `_session.py`. A pool of `_RENDER_WORKERS = max(2, min(4, os.cpu_count() or 2))` daemon threads (named `arrayview-render-{N}`, tracked in `_RENDER_THREADS`) drains the queue and resolves `asyncio.Future`s on the server event loop. `_ensure_render_thread()` keeps the pool at the configured worker count.

```python
await _render(loop, lambda: extract_slice(session, dim_x, dim_y, idx))
```

- Never use `asyncio.run_in_executor` for render work — the executor's `_global_shutdown` flag races with daemon thread cleanup.
- The queue accepts `None` as a sentinel to stop a worker.

## Prefetch Pool

Separate 1-thread `ThreadPoolExecutor` (`arrayview-prefetch`). Warms `raw_cache` for the next `PREFETCH_NEIGHBORS` (3) slices in the scroll direction. Budget-gated: skips if `slice_bytes * PREFETCH_NEIGHBORS > PREFETCH_BUDGET_BYTES` (16 MB). Errors are silently swallowed.

## Special Array Types

- **RGB arrays** — detected by `_setup_rgb()`. Skip colormap; go directly to `render_rgb_rgba()`. `session.rgb_axis` holds the axis index; `session.spatial_shape` is shape without that axis.
- **Complex arrays** — `apply_complex_mode()` transforms to real before colormapping. Supported modes: magnitude, phase, real, imag. FFT is a separate data-level transform (see note above).
- **NIfTI** — canonical-reoriented on load by `_load_nifti_with_meta()`. `session.spatial_meta` holds affine, voxel sizes, axis labels. RAS resample cached in `session.resampled_volume`.
- **Large memmap arrays** — `_default_start_dims_for_data()` may override the default startup axes to avoid high-stride reads.

## High-Risk Areas

- **Cache key collisions** — the `key_idx` tuple replaces slice-dim positions with `None`. A bug here causes stale tiles to display for wrong indices.
- **LUT initialization race** — `_init_luts()` uses a double-checked lock. Do not call `LUTS[name]` directly before calling `_init_luts()`.
- **Dtype handling** — `extract_slice()` must return `float32`. RGB uint8 arrays must NOT go through the float pipeline. Inserting a new format type that returns a different dtype will silently corrupt renders.
- **Compare center color overrides** — the compare center pane can preview its own LUT override (`_diffCenterColormap`) without changing the source panes. If a render bug only appears in the center pane, check the compare-specific colorbar state before blaming the shared LUT cache.
