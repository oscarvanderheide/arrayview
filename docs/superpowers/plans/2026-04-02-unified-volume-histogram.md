# Unified Volume Histogram Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the dual DR/histogram systems with a single volume-sampled histogram that drives all display-range adjustment.

**Architecture:** New `/volume-histogram/{sid}` backend endpoint subsamples slices along the scroll dimension. Frontend fetches volume histograms (cached per colorbar), computes quantile-based vmin/vmax client-side, and sends as `vmin_override`/`vmax_override`. Server-side DR logic (`global_stats`, `DR_PERCENTILES` cycling) is removed. Server falls back to 1–99% percentile on current slice when no override is provided.

**Tech Stack:** Python (FastAPI, numpy), JavaScript (inline in `_viewer.html`)

---

### Task 1: Add `/volume-histogram/{sid}` endpoint to `_server.py`

**Files:**
- Modify: `src/arrayview/_server.py:1777-1818` (insert new endpoint near existing `/histogram`)
- Modify: `src/arrayview/_render.py:119-139` (reuse `extract_slice`)

- [ ] **Step 1: Write the test**

Add to `tests/test_api.py` inside or after `TestHistogram`:

```python
class TestVolumeHistogram:
    """Tests for /volume-histogram/{sid} endpoint."""

    def test_volume_histogram_returns_counts_and_edges(self, client, sid_3d):
        """3D array: sample along dim 2 (scroll), display dims 0,1."""
        resp = client.get(f"/volume-histogram/{sid_3d}", params={
            "dim_x": 0, "dim_y": 1, "scroll_dim": 2, "bins": 32,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "counts" in data and "edges" in data
        assert len(data["counts"]) == 32
        assert len(data["edges"]) == 33
        assert "vmin" in data and "vmax" in data
        assert data["vmin"] < data["vmax"]

    def test_volume_histogram_with_fixed_indices(self, client, sid_4d):
        """4D array: fix dim 0 (parameter map), scroll along dim 1, display dims 2,3."""
        resp = client.get(f"/volume-histogram/{sid_4d}", params={
            "dim_x": 2, "dim_y": 3, "scroll_dim": 1,
            "fixed_indices": "0:0", "bins": 32,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["counts"]) == 32

    def test_volume_histogram_different_fixed_indices_differ(self, client, sid_4d):
        """Different fixed indices should give different histograms (different parameter maps)."""
        resp_a = client.get(f"/volume-histogram/{sid_4d}", params={
            "dim_x": 2, "dim_y": 3, "scroll_dim": 1,
            "fixed_indices": "0:0", "bins": 32,
        })
        resp_b = client.get(f"/volume-histogram/{sid_4d}", params={
            "dim_x": 2, "dim_y": 3, "scroll_dim": 1,
            "fixed_indices": "0:2", "bins": 32,
        })
        a = resp_a.json()
        b = resp_b.json()
        # Different parameter-map slices should generally differ
        assert a["counts"] != b["counts"] or a["vmin"] != b["vmin"]

    def test_volume_histogram_unknown_sid_is_404(self, client):
        resp = client.get("/volume-histogram/nonexistent", params={
            "dim_x": 0, "dim_y": 1, "scroll_dim": 2,
        })
        assert resp.status_code == 404

    def test_volume_histogram_caches_result(self, client, sid_3d):
        """Two identical requests should return identical results (cache hit)."""
        params = {"dim_x": 0, "dim_y": 1, "scroll_dim": 2, "bins": 32}
        resp1 = client.get(f"/volume-histogram/{sid_3d}", params=params)
        resp2 = client.get(f"/volume-histogram/{sid_3d}", params=params)
        assert resp1.json() == resp2.json()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/oscar/Projects/packages/python/arrayview && uv run pytest tests/test_api.py::TestVolumeHistogram -v`
Expected: FAIL (404 — endpoint doesn't exist yet)

- [ ] **Step 3: Implement the endpoint**

Add to `_server.py` after the existing `/histogram/{sid}` endpoint (after line 1818):

```python
@app.get("/volume-histogram/{sid}")
def get_volume_histogram(
    sid: str,
    dim_x: int,
    dim_y: int,
    scroll_dim: int,
    fixed_indices: str = "",
    complex_mode: int = 0,
    bins: int = 64,
):
    """Return a histogram sampled across the scroll dimension.

    Subsamples up to 16 evenly-spaced slices along *scroll_dim*, merges
    their pixel data, and returns a single histogram.  The result is
    cached on the session so repeated requests are instant.

    *fixed_indices* is a comma-separated list of ``dim:idx`` pairs that
    pin non-display, non-scroll dimensions (e.g. ``"3:0"`` to select the
    first parameter map in qMRI mode).
    """
    session = SESSIONS.get(sid)
    if not session:
        return Response(status_code=404)

    # Parse fixed indices
    fixed = {}
    if fixed_indices:
        for pair in fixed_indices.split(","):
            if ":" in pair:
                d, v = pair.split(":", 1)
                fixed[int(d)] = int(v)

    # Check cache
    cache_key = (dim_x, dim_y, scroll_dim, tuple(sorted(fixed.items())), complex_mode)
    if not hasattr(session, "_volume_hist_cache"):
        session._volume_hist_cache = {}
    cached = session._volume_hist_cache.get(cache_key)
    if cached is not None and cached.get("_data_version") == session.data_version:
        return cached["result"]

    # Sample slices along scroll_dim
    n = session.shape[scroll_dim]
    max_samples = 16
    if n <= max_samples:
        sample_indices = list(range(n))
    else:
        step = n / max_samples
        sample_indices = [int(i * step) for i in range(max_samples)]

    pixels = []
    for si in sample_indices:
        idx_list = list(indices_mid(session.shape))
        idx_list[scroll_dim] = si
        idx_list[dim_x] = 0  # will be sliced
        idx_list[dim_y] = 0  # will be sliced
        for d, v in fixed.items():
            idx_list[d] = v
        raw = extract_slice(session, dim_x, dim_y, idx_list)
        data = apply_complex_mode(raw, complex_mode)
        flat = data.ravel()
        finite = flat[np.isfinite(flat)]
        if finite.size > 0:
            pixels.append(finite)

    if not pixels:
        result = {"counts": [], "edges": [], "vmin": 0.0, "vmax": 1.0}
        session._volume_hist_cache[cache_key] = {
            "_data_version": session.data_version, "result": result,
        }
        return result

    merged = np.concatenate(pixels)
    vmin = float(merged.min())
    vmax = float(merged.max())
    if vmin == vmax:
        result = {
            "counts": [int(merged.size)],
            "edges": [vmin, vmax + 1e-9],
            "vmin": vmin,
            "vmax": vmax,
        }
    else:
        bins = max(8, min(bins, 512))
        counts, edges = np.histogram(merged, bins=bins)
        result = {
            "counts": counts.tolist(),
            "edges": [float(e) for e in edges],
            "vmin": vmin,
            "vmax": vmax,
        }

    session._volume_hist_cache[cache_key] = {
        "_data_version": session.data_version, "result": result,
    }
    return result
```

We need a helper `indices_mid` to build default indices at the midpoint of each dimension. Add it near the top of the endpoint or as a local helper:

```python
def indices_mid(shape):
    """Return indices at the midpoint of each dimension."""
    return tuple(s // 2 for s in shape)
```

Note: `extract_slice` ignores `idx_list[dim_x]` and `idx_list[dim_y]` (they become `slice(None)`), so the values we set there don't matter. The important indices are `scroll_dim` (which we iterate) and the `fixed` dims.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/oscar/Projects/packages/python/arrayview && uv run pytest tests/test_api.py::TestVolumeHistogram -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/arrayview/_server.py tests/test_api.py
git commit -m "feat: add /volume-histogram endpoint with subsampled scroll-dim data"
```

---

### Task 2: Add volume-histogram handler to `_stdio_server.py`

**Files:**
- Modify: `src/arrayview/_stdio_server.py:336-373` (add handler near existing `_handle_histogram`)
- Modify: `src/arrayview/_stdio_server.py:416-429` (add route in `_handle_fetch_proxy`)

- [ ] **Step 1: Add the handler function**

Add after `_handle_histogram` (after line 372):

```python
def _handle_volume_histogram(sid: str, params: dict[str, str]) -> None:
    """Compute histogram by subsampling slices along the scroll dimension."""
    session = SESSIONS.get(sid)
    if not session:
        _write_error("session not found")
        return
    dim_x = int(params.get("dim_x", "0"))
    dim_y = int(params.get("dim_y", "1"))
    scroll_dim = int(params.get("scroll_dim", "2"))
    complex_mode = int(params.get("complex_mode", "0"))
    bins = max(8, min(int(params.get("bins", "64")), 512))

    fixed = {}
    fixed_str = params.get("fixed_indices", "")
    if fixed_str:
        for pair in fixed_str.split(","):
            if ":" in pair:
                d, v = pair.split(":", 1)
                fixed[int(d)] = int(v)

    # Check cache
    cache_key = (dim_x, dim_y, scroll_dim, tuple(sorted(fixed.items())), complex_mode)
    if not hasattr(session, "_volume_hist_cache"):
        session._volume_hist_cache = {}
    cached = session._volume_hist_cache.get(cache_key)
    if cached is not None and cached.get("_data_version") == session.data_version:
        _write_json(cached["result"])
        return

    n = session.shape[scroll_dim]
    max_samples = 16
    if n <= max_samples:
        sample_indices = list(range(n))
    else:
        step = n / max_samples
        sample_indices = [int(i * step) for i in range(max_samples)]

    pixels = []
    for si in sample_indices:
        idx_list = [s // 2 for s in session.shape]
        idx_list[scroll_dim] = si
        for d, v in fixed.items():
            idx_list[d] = v
        raw = extract_slice(session, dim_x, dim_y, idx_list)
        data = apply_complex_mode(raw, complex_mode)
        flat = data.ravel()
        finite = flat[np.isfinite(flat)]
        if finite.size > 0:
            pixels.append(finite)

    if not pixels:
        result = {"counts": [], "edges": [], "vmin": 0.0, "vmax": 1.0}
    else:
        merged = np.concatenate(pixels)
        vmin = float(merged.min())
        vmax = float(merged.max())
        if vmin == vmax:
            result = {"counts": [int(merged.size)], "edges": [vmin, vmax + 1e-9], "vmin": vmin, "vmax": vmax}
        else:
            counts, edges = np.histogram(merged, bins=bins)
            result = {"counts": counts.tolist(), "edges": [float(e) for e in edges], "vmin": vmin, "vmax": vmax}

    session._volume_hist_cache[cache_key] = {"_data_version": session.data_version, "result": result}
    _write_json(result)
```

- [ ] **Step 2: Add routing**

In `_handle_fetch_proxy`, add after the `histogram` route (after line 423):

```python
    elif route == "volume-histogram" and sid:
        _handle_volume_histogram(sid, params)
```

- [ ] **Step 3: Commit**

```bash
git add src/arrayview/_stdio_server.py
git commit -m "feat: add volume-histogram handler to stdio server"
```

---

### Task 3: Remove `global_stats` and simplify server-side DR

**Files:**
- Modify: `src/arrayview/_session.py:168,201,209-254,257-265,279-282`
- Modify: `src/arrayview/_render.py:108-116,230-254`
- Modify: `src/arrayview/_server.py:601,634,1946,1976,2009,2614-2619,2696-2701`

- [ ] **Step 1: Simplify `_compute_vmin_vmax` in `_render.py`**

Replace lines 108-116:

```python
def _compute_vmin_vmax(session, data, dr=0, complex_mode=0):
    if complex_mode == 1 and np.iscomplexobj(session.data):
        return (-float(np.pi), float(np.pi))
    return float(np.percentile(data, 1)), float(np.percentile(data, 99))
```

The `dr` parameter is kept for signature compatibility but ignored. The server always falls back to 1–99% percentile on the current slice data when no `vmin_override`/`vmax_override` is provided.

- [ ] **Step 2: Simplify `_prepare_display` in `_render.py`**

Replace the log_scale branch (lines 238-251) that checks `global_stats`:

```python
    if log_scale:
        data = np.log1p(np.abs(data)).astype(np.float32)
        vmin = float(np.percentile(data, 1))
        vmax = float(np.percentile(data, 99))
    else:
        vmin, vmax = _compute_vmin_vmax(session, data, dr, complex_mode)
```

- [ ] **Step 3: Remove `compute_global_stats` from `_session.py`**

Delete the `compute_global_stats` method (lines 209-254) and the `self.global_stats = {}` line (line 168) and the `self.compute_global_stats()` call (line 201).

Update `_recommend_colormap_reason` to not use `global_stats`. Replace:

```python
def _recommend_colormap_reason(data, global_stats: dict) -> str:
    """Return a human-readable reason for the recommended colormap choice."""
    dtype = np.dtype(getattr(data, "dtype", np.float32))
    if dtype.kind == "b":
        return "gray (bool dtype — binary data)"
    if np.iscomplexobj(data):
        return "gray (complex dtype — showing magnitude)"
    vmin, _ = global_stats.get(1, global_stats.get(0, (0.0, 1.0)))
    if dtype.kind in ("i", "f") and vmin < 0:
        return "RdBu_r (signed data — vmin < 0)"
    return "gray (default — unsigned/positive data)"
```

With:

```python
def _recommend_colormap_reason(data) -> str:
    """Return a human-readable reason for the recommended colormap choice."""
    dtype = np.dtype(getattr(data, "dtype", np.float32))
    if dtype.kind == "b":
        return "gray (bool dtype — binary data)"
    if np.iscomplexobj(data):
        return "gray (complex dtype — showing magnitude)"
    if dtype.kind == "i":
        return "RdBu_r (signed integer dtype)"
    if dtype.kind == "f":
        # Quick sign check via small sample
        sample = np.array(data).ravel()[:10000]
        sample = np.nan_to_num(sample)
        if sample.size > 0 and float(sample.min()) < 0:
            return "RdBu_r (signed data — vmin < 0)"
    return "gray (default — unsigned/positive data)"
```

- [ ] **Step 4: Remove `DR_PERCENTILES` and `DR_LABELS` from `_session.py`**

Delete:
```python
DR_PERCENTILES = [(0, 100), (1, 99), (5, 95), (10, 90)]
DR_LABELS = ["0-100%", "1-99%", "5-95%", "10-90%"]
```

Remove them from `__all__` as well.

- [ ] **Step 5: Remove `compute_global_stats` calls in `_server.py`**

Search for all `compute_global_stats` calls and remove them. These are at lines ~601, ~634, ~1976, ~2009. Also remove the `session.global_stats` import/usage.

Update the call to `_recommend_colormap_reason` (line ~1946) to drop the `global_stats` argument:

```python
# Before:
_recommend_colormap_reason(session.data, session.global_stats)
# After:
_recommend_colormap_reason(session.data)
```

- [ ] **Step 6: Simplify GIF and grid endpoints**

In both `/gif/{sid}` and `/grid/{sid}` endpoints, replace the `global_stats` lookup blocks.

For `/gif/{sid}` (lines 2696-2701), replace:
```python
    if dr in session.global_stats:
        vmin, vmax = session.global_stats[dr]
    else:
        pct_lo, pct_hi = DR_PERCENTILES[dr % len(DR_PERCENTILES)]
        vmin = float(np.percentile(all_data, pct_lo))
        vmax = float(np.percentile(all_data, pct_hi))
```
With:
```python
    vmin = float(np.percentile(all_data, 1))
    vmax = float(np.percentile(all_data, 99))
```

Same replacement for `/grid/{sid}` (lines 2614-2619).

- [ ] **Step 7: Remove `DR_PERCENTILES` and `DR_LABELS` imports from `_server.py` and `_stdio_server.py`**

In `_server.py`, update the import from `_session` to remove `DR_PERCENTILES` and `DR_LABELS`.

In `_stdio_server.py`, remove the import of `DR_LABELS` (used only for template injection) if present.

- [ ] **Step 8: Run full API test suite**

Run: `cd /Users/oscar/Projects/packages/python/arrayview && uv run pytest tests/test_api.py -v`
Expected: All tests pass. Some existing tests may reference DR-related behavior — fix any failures.

- [ ] **Step 9: Commit**

```bash
git add src/arrayview/_session.py src/arrayview/_render.py src/arrayview/_server.py src/arrayview/_stdio_server.py tests/test_api.py
git commit -m "refactor: remove global_stats and DR_PERCENTILES, simplify to 1-99% default"
```

---

### Task 4: Frontend — add `_fetchVolumeHistogram` and update quantile presets

**Files:**
- Modify: `src/arrayview/_viewer.html:4569-4575` (QUANTILE_PRESETS)
- Modify: `src/arrayview/_viewer.html:4885-4905` (add volume histogram fetch)
- Modify: `src/arrayview/_viewer.html:9180-9186,9254-9260` (inline presets in lebesgue/normal paths)

- [ ] **Step 1: Update `ColorBar.QUANTILE_PRESETS` to 4 levels**

At line 4569, replace:

```javascript
        ColorBar.QUANTILE_PRESETS = [
            { label: 'full range', lo: 0, hi: 1 },
            { label: '0.1\u201399.9%', lo: 0.001, hi: 0.999 },
            { label: '1\u201399%', lo: 0.01, hi: 0.99 },
            { label: '5\u201395%', lo: 0.05, hi: 0.95 },
            { label: '10\u201390%', lo: 0.10, hi: 0.90 },
        ];
```

With:

```javascript
        ColorBar.QUANTILE_PRESETS = [
            { label: 'full range', lo: 0, hi: 1 },
            { label: '1\u201399%', lo: 0.01, hi: 0.99 },
            { label: '5\u201395%', lo: 0.05, hi: 0.95 },
            { label: '10\u201390%', lo: 0.10, hi: 0.90 },
        ];
```

- [ ] **Step 2: Add `_fetchVolumeHistogram` function**

Add after `_fetchHistogram` (after line 4905):

```javascript
        let _volHistCache = {};
        let _volHistFetching = false;

        async function _fetchVolumeHistogram(opts = {}) {
            const fetchDimX = opts.dimX ?? dim_x;
            const fetchDimY = opts.dimY ?? dim_y;
            const scrollDim = opts.scrollDim ?? current_slice_dim;
            const cMode = opts.complexMode ?? complexMode;
            const fixedIndices = opts.fixedIndices ?? '';
            const cacheKey = `${fetchDimX}:${fetchDimY}:${scrollDim}:${fixedIndices}:${cMode}`;

            if (_volHistCache[cacheKey]) return _volHistCache[cacheKey];

            const params = new URLSearchParams({
                dim_x: fetchDimX,
                dim_y: fetchDimY,
                scroll_dim: scrollDim,
                complex_mode: cMode,
                bins: 64,
            });
            if (fixedIndices) params.set('fixed_indices', fixedIndices);
            try {
                const resp = await proxyFetch(`/volume-histogram/${sid}?${params}`);
                if (!resp.ok) return null;
                const data = await resp.json();
                _volHistCache[cacheKey] = data;
                return data;
            } catch (_) { return null; }
        }

        function _invalidateVolHistCache() {
            _volHistCache = {};
        }
```

- [ ] **Step 3: Update the inline `_QUANTILE_PRESETS` in the `d` key handler**

There are two inline copies of `_QUANTILE_PRESETS` in the `d` key handler (lines 9180-9186 and 9254-9260). Replace both with 4 levels:

```javascript
                    const _QUANTILE_PRESETS = [
                        { label: 'full range', lo: 0, hi: 1 },
                        { label: '1\u201399%', lo: 0.01, hi: 0.99 },
                        { label: '5\u201395%', lo: 0.05, hi: 0.95 },
                        { label: '10\u201390%', lo: 0.10, hi: 0.90 },
                    ];
```

- [ ] **Step 4: Invalidate volume histogram cache on relevant changes**

Add `_invalidateVolHistCache()` calls wherever the data context changes:
- After `complexMode` changes (search for `complexMode =` assignments near mode cycling)
- After `/clearcache` calls triggered by data reload
- After `current_slice_dim` changes (dim cycle)

The exact locations need to be found by searching for existing `_histData = null` invalidations and adding `_invalidateVolHistCache()` nearby.

- [ ] **Step 5: Commit**

```bash
git add src/arrayview/_viewer.html
git commit -m "feat: add _fetchVolumeHistogram and update quantile presets to 4 levels"
```

---

### Task 5: Frontend — unify `d` key handler

This is the core frontend change. All `d` key paths converge on the same pattern: fetch volume histogram → expand colorbar → cycle quantiles.

**Files:**
- Modify: `src/arrayview/_viewer.html:9087-9307` (the `d` key handler block)

- [ ] **Step 1: Rewrite qMRI hover path (lines 9118-9138)**

Replace the `drIdx` cycling path:

```javascript
                // In qMRI mode, hovering a specific pane: expand + cycle quantile
                if (qmriActive && _hoveredQmriView) {
                    const hv = _hoveredQmriView;
                    const cb = qmriMosaicActive ? hv._mosaicColorBar : hv._colorBar;
                    if (!cb) return;
                    if (!cb._expanded) {
                        await cb.expand();
                        cb.startAutoDismiss();
                        showStatus('histogram');
                    } else {
                        const q = cb.cycleQuantile();
                        if (q) {
                            hv.lockedVmin = q.vmin; hv.lockedVmax = q.vmax;
                            hv.seq++; hv.rendering = false; hv.pending = false;
                            qvRender(hv);
                            const paneLabel = QMRI_LABELS[hv.colormap] || hv.colormap;
                            showStatus(`${paneLabel}: ${q.label}`);
                        }
                        cb.startAutoDismiss();
                    }
                    return;
                }
```

- [ ] **Step 2: Update the qMRI ColorBar `fetchHistogram` to use volume histogram**

In the qMRI view initialization (line 10757-10763), replace `fetchHistogram`:

```javascript
                        fetchHistogram: async () => {
                            const fixedPairs = [];
                            fixedPairs.push(qmriDim + ':' + view.qmriIdx);
                            const fixedIndices = fixedPairs.join(',');
                            return _fetchVolumeHistogram({
                                dimX: dim_x, dimY: dim_y,
                                scrollDim: current_slice_dim,
                                complexMode: complexMode,
                                fixedIndices,
                            });
                        },
```

Do the same for compare-qMRI ColorBar initialization (search for similar `fetchHistogram` in compare-qMRI setup).

- [ ] **Step 3: Update primary ColorBar `fetchHistogram` to use volume histogram**

At line 4599, replace:

```javascript
            fetchHistogram: () => compareActive ? _fetchHistogramCompare() : _fetchHistogram(),
```

With:

```javascript
            fetchHistogram: () => compareActive ? _fetchHistogramCompare() : _fetchVolumeHistogram(),
```

Note: compare mode still uses `_fetchHistogramCompare()` which merges histograms from multiple sessions. That function should also be updated to use volume histograms, but that's a separate concern — for now, compare mode can continue using per-slice histograms since it involves multiple arrays.

- [ ] **Step 4: Update the normal/lebesgue `d` key paths to use volume histogram data**

In the lebesgue path (lines 9178-9211), change the `_histData` reference to use volume histogram. The histogram data is already fetched by the ColorBar's `fetchHistogram` callback, so `_histData` should be populated from the volume histogram.

Replace the first `_fetchHistogram` call in the expand path (line 9220-9221):

```javascript
                        const fn = compareActive ? _fetchHistogramCompare : _fetchVolumeHistogram;
```

But we also need the per-slice histogram for the Lebesgue overlay. So fetch both:

```javascript
                        // Fetch volume histogram for quantile-based windowing
                        const volFn = compareActive ? _fetchHistogramCompare : _fetchVolumeHistogram;
                        const volData = await volFn();
                        if (volData) _histData = volData;
                        // Also fetch per-slice histogram for Lebesgue pixel overlay
                        _fetchHistogram();
```

Actually, looking more carefully at the code, `_histData` is used for both the visual histogram AND the quantile computation. The Lebesgue overlay uses a separate `/lebesgue` endpoint. So we can simply replace `_histData` with volume histogram data.

Replace line 9220-9221:

```javascript
                        const fn = compareActive ? _fetchHistogramCompare : _fetchVolumeHistogram;
                        await fn().then(d => { if (d) _histData = d; });
```

- [ ] **Step 5: Commit**

```bash
git add src/arrayview/_viewer.html
git commit -m "feat: unify d key handler to use volume histogram for all modes"
```

---

### Task 6: Frontend — remove DR state variables and clean up

**Files:**
- Modify: `src/arrayview/_viewer.html` (multiple locations)
- Modify: `src/arrayview/_server.py:3088` (template injection)
- Modify: `src/arrayview/_stdio_server.py:568` (template injection)

- [ ] **Step 1: Remove `DR_LABELS` template variable**

In `_viewer.html` line 1504, delete:
```javascript
        const DR_LABELS = __DR_LABELS__;
```

In `_server.py`, remove the `.replace("__DR_LABELS__", str(DR_LABELS))` line (~line 3088).

In `_stdio_server.py`, remove the `.replace("__DR_LABELS__", str(DR_LABELS))` line (~line 568).

- [ ] **Step 2: Remove `dr_idx` variable and references**

In `_viewer.html`:

1. Line 1686: Change `let colormap_idx = 0, dr_idx = 1;` to `let colormap_idx = 0;`
2. Line 1849: Delete `let _diffDrIdx = null;`

- [ ] **Step 3: Hardcode `dr: '0'` in all server requests**

Replace all occurrences of `dr: String(dr_idx)`, `dr: dr_idx`, `dr: String(v.drIdx !== null ? v.drIdx : dr_idx)`, `dr: String(_diffDrIdx !== null ? _diffDrIdx : dr_idx)` with `dr: '0'` (for string params) or `dr: 0` (for object params).

Affected locations (approximate line numbers):
- 5790, 5866, 5941, 5998, 7166, 7470
- 11090, 11115, 11462, 11486, 11666, 11694
- 12105, 12150

For the GIF URL (line 7470), replace `dr=${dr_idx}` with `dr=0`.

- [ ] **Step 4: Remove `drIdx` from qMRI view objects**

Line 10745: Remove `drIdx: null,` from the view object initialization.

- [ ] **Step 5: Remove `dr_idx` from state save/restore**

Line 6775: Remove `dr_idx,` from `collectStateSnapshot()`.
Line 6833: Remove `dr_idx = s.dr_idx ?? dr_idx;` from `applyStateSnapshot()`.

- [ ] **Step 6: Remove the diff DR cycling path (lines 9088-9095)**

Delete the block:
```javascript
                if (_mouseOverCenterPane && compareActive && diffMode > 0) {
                    if (_diffDrIdx === null) _diffDrIdx = dr_idx;
                    _diffDrIdx = (_diffDrIdx + 1) % DR_LABELS.length;
                    await fetchAndDrawDiff();
                    showStatus(`Diff dynamic range: ${DR_LABELS[_diffDrIdx]}`);
                    return;
                }
```

The diff mode should now use the same expand → cycle quantile pattern via its ColorBars.

- [ ] **Step 7: Run existing browser tests to check for regressions**

Run: `cd /Users/oscar/Projects/packages/python/arrayview && uv run pytest tests/test_browser.py -v -k "dynamic_range or histogram" --timeout=60`
Expected: Tests that reference DR_LABELS or DR cycling may need updating.

- [ ] **Step 8: Update browser tests**

In `tests/test_browser.py`, find `test_d_cycles_dynamic_range_shows_status` and update it. The test should now verify that pressing `d` expands the histogram and cycles quantile labels (e.g., "full range", "1–99%", etc.) instead of DR labels.

- [ ] **Step 9: Commit**

```bash
git add src/arrayview/_viewer.html src/arrayview/_server.py src/arrayview/_stdio_server.py tests/test_browser.py
git commit -m "refactor: remove DR state variables, unify to volume histogram quantiles"
```

---

### Task 7: Clean up — invalidation, profile endpoints, and edge cases

**Files:**
- Modify: `src/arrayview/_server.py` (profile endpoints)
- Modify: `src/arrayview/_viewer.html` (cache invalidation)

- [ ] **Step 1: Update profile endpoints in `_server.py`**

The `/profile/1d/{sid}` and `/profile/2d/{sid}` endpoints (lines ~2614, ~2696) reference `global_stats`. Replace:

```python
    if dr in session.global_stats:
        vmin, vmax = session.global_stats[dr]
    else:
        pct_lo, pct_hi = DR_PERCENTILES[dr % len(DR_PERCENTILES)]
        ...
```

With:

```python
    vmin = float(np.percentile(finite, 1))
    vmax = float(np.percentile(finite, 99))
```

(Adjust variable names to match what's in scope — `finite` or the appropriate data variable.)

- [ ] **Step 2: Invalidate `_volume_hist_cache` on reload/FFT**

In `_server.py`, wherever `compute_global_stats()` was called after data changes (reload, FFT), add cache invalidation:

```python
    if hasattr(session, "_volume_hist_cache"):
        session._volume_hist_cache.clear()
```

These locations are approximately lines 601, 634, 1976, 2009.

- [ ] **Step 3: Frontend — invalidate on scroll-dim and complex-mode changes**

In `_viewer.html`, add `_invalidateVolHistCache()` calls:
- When `current_slice_dim` changes (near dim cycling logic)
- When `complexMode` changes
- When data is reloaded (near existing cache clear calls)

Search for `_histData = null` to find existing invalidation points and add the volume cache invalidation alongside.

- [ ] **Step 4: Remove `DR_PERCENTILES` import from `_render.py`**

The `_render.py` file imports `DR_PERCENTILES` from `_session`. Since `_compute_vmin_vmax` and `_prepare_display` no longer use it, remove the import.

Also check `render_mosaic` — it has its own DR_PERCENTILES usage in the log_scale path. Simplify to hardcoded 1-99%:

```python
    if log_scale:
        frames = [np.log1p(np.abs(f)).astype(np.float32) for f in frames]
    all_data = np.stack(frames)
    vmin = float(np.percentile(all_data, 1))
    vmax = float(np.percentile(all_data, 99))
    if complex_mode == 1 and np.iscomplexobj(session.data):
        vmin, vmax = -float(np.pi), float(np.pi)
```

- [ ] **Step 5: Run full test suite**

Run: `cd /Users/oscar/Projects/packages/python/arrayview && uv run pytest tests/ -v --timeout=120 -x`
Expected: All tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/arrayview/_server.py src/arrayview/_render.py src/arrayview/_viewer.html
git commit -m "chore: clean up DR remnants from profiles, render, and cache invalidation"
```

---

### Task 8: Final validation and browser smoke test

- [ ] **Step 1: Run API tests**

Run: `cd /Users/oscar/Projects/packages/python/arrayview && uv run pytest tests/test_api.py -v`
Expected: All pass

- [ ] **Step 2: Run browser tests**

Run: `cd /Users/oscar/Projects/packages/python/arrayview && uv run pytest tests/test_browser.py -v --timeout=120`
Expected: All pass (some may have been updated in Task 6)

- [ ] **Step 3: Run interaction tests**

Run: `cd /Users/oscar/Projects/packages/python/arrayview && uv run pytest tests/test_interactions.py -v --timeout=120`
Expected: All pass

- [ ] **Step 4: Run mode consistency tests**

Run: `cd /Users/oscar/Projects/packages/python/arrayview && uv run pytest tests/test_mode_consistency.py -v --timeout=120`
Expected: All pass

- [ ] **Step 5: Manual smoke test**

Open arrayview with a 3D array:
```bash
cd /Users/oscar/Projects/packages/python/arrayview && uv run python -c "
import numpy as np
import arrayview
arr = np.random.randn(50, 128, 128).astype(np.float32)
arr[0, 60:70, 60:70] = 1000  # outlier slice
arrayview.view(arr)
"
```

Verify:
1. Initial view shows reasonable range (1-99% of current slice)
2. Press `d` → histogram expands, shows volume-sampled distribution
3. Press `d` again → cycles: "full range" → "1–99%" → "5–95%" → "10–90%"
4. Scroll to the outlier slice → range stays stable (not jumping)
5. `Escape` to dismiss histogram

- [ ] **Step 6: Manual qMRI smoke test**

If test data with qMRI parameter maps is available, verify:
1. Each pane's `d` key cycles independently
2. T1 and T2 panes show different histogram shapes
3. Quantile ranges are stable across scroll positions
