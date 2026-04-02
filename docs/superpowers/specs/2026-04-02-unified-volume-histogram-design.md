# Unified Volume-Sampled Histogram System

## Problem

Two separate systems handle display-range adjustment:

1. **DR system (server-side):** `compute_global_stats()` pre-samples the volume at session init, stores percentile pairs in `global_stats`. The server applies `dr_idx` during tile rendering via `_compute_vmin_vmax()`. Frontend cycles `dr_idx` / `hv.drIdx` / `_diffDrIdx`.

2. **Histogram quantile system (client-side):** `/histogram/{sid}` computes a histogram from the **current 2D slice only**. The expanded colorbar displays it visually; pressing `d` cycles quantile presets that set `manualVmin`/`manualVmax`.

Problems:
- Histogram quantiles are slice-dependent — scrolling one slice can wildly change the range.
- Two redundant code paths for conceptually the same operation.
- DR system has 4 presets; histogram system has 5 — inconsistent UX.
- qMRI panes need per-pane volume histograms (T1 vs T2 have different distributions), but the current histogram endpoint doesn't support this.

## Design

### New backend endpoint: `/volume-histogram/{sid}`

Computes a histogram by subsampling slices along the scroll dimension.

**Parameters:**
- `dim_x`, `dim_y` — display axes
- `scroll_dim` — dimension to sample along
- `fixed_indices` — comma-separated `dim:idx` pairs for all other dimensions (e.g., `3:0` to pin the parameter-map dim at index 0 for T1)
- `complex_mode` — 0=mag, 1=phase, 2=real, 3=imag
- `bins` — histogram bin count (default 64)

**Algorithm:**
1. Determine scroll extent: `N = shape[scroll_dim]`
2. Pick ~8-16 evenly spaced indices along scroll_dim (all if N <= 16)
3. For each sampled index, build a full index tuple (fixed_indices + scroll index), extract 2D slice via `extract_slice()`, apply complex mode
4. Concatenate all sampled pixels, compute histogram over finite values
5. Return `{counts, edges, vmin, vmax}` — same response shape as current `/histogram/{sid}`

**Caching:** Cache result on the session keyed by `(dim_x, dim_y, scroll_dim, frozen(fixed_indices), complex_mode)`. Invalidate on `data_version` change (reload).

### Remove server-side DR system

Delete:
- `compute_global_stats()` method and `global_stats` attribute on `Session`
- `DR_PERCENTILES` and `DR_LABELS` constants from `_session.py`
- DR-dependent branches in `_compute_vmin_vmax()` and `_prepare_display()`
- `compute_global_stats()` calls in `_server.py` (reload, FFT, etc.)

**Fallback for `_compute_vmin_vmax()`:** When no `vmin_override`/`vmax_override` is provided and no manual range is set, the server falls back to computing percentiles from the current slice data (existing `np.percentile(data, ...)` path). This is the same behavior as today when `global_stats` misses. The `dr` parameter becomes unused and can be hardcoded to 0 (full range) on the server side, since the client now owns all windowing.

**Impact on `_recommend_colormap_reason()`:** This function reads `global_stats` to check if vmin < 0 (to recommend `RdBu_r` for signed data). Replace with a direct check on the data dtype and a small sample — it only needs to know the sign, not exact percentiles.

### Keep existing `/histogram/{sid}` endpoint

The per-slice histogram endpoint is still needed for the Lebesgue overlay (per-pixel quantile lookup on the current slice). It does NOT drive display range anymore.

### Frontend changes

**Unified quantile presets (4 levels):**
```javascript
const QUANTILE_PRESETS = [
    { label: 'full range', lo: 0, hi: 1 },
    { label: '1–99%',      lo: 0.01, hi: 0.99 },
    { label: '5–95%',      lo: 0.05, hi: 0.95 },
    { label: '10–90%',     lo: 0.10, hi: 0.90 },
];
```

Update `ColorBar.QUANTILE_PRESETS` to match.

**Remove DR state:**
- Delete `dr_idx` variable and all references
- Delete `DR_LABELS` injection (`__DR_LABELS__`)
- Delete `hv.drIdx` from qMRI views
- Delete `_diffDrIdx`
- Stop sending `dr` parameter in tile/render requests (or hardcode to 0)

**Unified `d` key handler:**

All modes follow the same pattern:

1. Determine target colorbar(s) — single primary, or per-pane in qMRI/compare
2. If colorbar not expanded: fetch volume histogram → expand → show histogram → start auto-dismiss
3. If already expanded: cycle to next quantile preset → compute vmin/vmax from histogram bins → set `manualVmin`/`manualVmax` (or per-pane equivalents) → re-render

**Volume histogram fetch (`_fetchVolumeHistogram()`):**
```javascript
async function _fetchVolumeHistogram(opts = {}) {
    const { dimX, dimY, scrollDim, fixedIndices, complexMode } = opts;
    const params = new URLSearchParams({
        dim_x: dimX ?? dim_x,
        dim_y: dimY ?? dim_y,
        scroll_dim: scrollDim ?? current_slice_dim,
        complex_mode: complexMode ?? complexMode,
        bins: 64,
    });
    if (fixedIndices) params.set('fixed_indices', fixedIndices);
    const resp = await proxyFetch(`/volume-histogram/${sid}?${params}`);
    return resp.ok ? resp.json() : null;
}
```

**qMRI panes:** Each pane builds its `fixed_indices` from its parameter-map index and any other pinned dimensions. Each pane's ColorBar fetches and caches its own volume histogram independently.

**Cache key in frontend:** `${dimX}:${dimY}:${scrollDim}:${fixedIndices}:${complexMode}`. Stored on the colorbar instance. Invalidated on scroll-dim change, complex-mode change, or data reload — NOT on scroll position change.

### What stays the same

- `manualVmin`/`manualVmax` and `vmin_override`/`vmax_override` mechanism — unchanged
- Colorbar visual rendering, expansion animation, KDE overlay — unchanged
- Lebesgue mode per-pixel overlay — still uses per-slice `/histogram/{sid}`
- Auto-dismiss timer behavior — unchanged
- Mouse-drag windowing on colorbar — unchanged

## Migration

The `dr` query parameter in tile requests becomes vestigial. The server should accept it (for backwards compat with cached URLs / in-flight requests) but ignore it when `vmin_override`/`vmax_override` are provided — which they always will be once the client drives all windowing. Ultimately `dr` can default to 0 (full range) server-side.

## Scope

This spec covers only the histogram/DR unification. The separate issue (immersive qMRI colorbars showing a shared island instead of per-pane) is out of scope.
