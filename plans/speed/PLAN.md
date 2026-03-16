# Speed Improvement Plan — arrayview

> Created: 2026-03-15
> Goal: Reduce startup latency and per-frame rendering overhead.
> Reference comparison: [ndv](https://github.com/pyapp-kit/ndv)

---

## Profiled Baseline (macOS, 2026-03-15)

### Import phase (`import arrayview`)

| Component                     | Time (ms) | Notes                              |
|-------------------------------|-----------|--------------------------------------|
| numpy                         | ~85       | unavoidable                         |
| **fastapi**                   | **~175**  | **biggest single import**           |
| uvicorn                       | ~18       | pulled in by server                 |
| asyncio                       | ~23       | stdlib                              |
| _platform (arrayview)         | ~44       | mostly setting up env detection     |
| _session, _render, _io, etc.  | <1 each   | lazy/deferred                       |
| **Total `import arrayview`**  | **~350**  | fastapi + numpy dominate            |

### Environment detection (called inside `view()`)

| Check                   | Time (ms) | Notes                                 |
|-------------------------|-----------|---------------------------------------|
| **`_in_jupyter()`**     | **~183**  | **imports IPython** (one-time cost)   |
| `_in_vscode_terminal()` | ~26       | walks process tree (`ps ewwww`)       |
| `_is_vscode_remote()`  | <1        | cached after first vscode check       |
| `_in_vscode_tunnel()`  | <1        | cached                                |
| `_can_native_window()`  | <1        | quick check                           |

### Session creation (`Session(arr)`)

| Array shape            | Time (ms) | Notes                                 |
|------------------------|-----------|---------------------------------------|
| 256×256                | 9         | `compute_global_stats()` dominates    |
| 512×512                | 1         | already cached/fast after first       |
| 1024×1024              | 1         | sampling is effective                 |
| 64×256×256 (3D)        | 14        | more slices to sample                 |
| 10×64×128×128 (4D)     | 25        | 20 random slices sampled              |

### Server startup

| Step                       | Time (ms) | Notes                              |
|----------------------------|-----------|--------------------------------------|
| socket bind + listen       | <1        | event-based ready signal, fast       |
| server thread → ready      | ~1        | uvicorn startup overlaps with main   |

### First render (deferred costs)

| Step                          | Time (ms) | Notes                              |
|-------------------------------|-----------|--------------------------------------|
| **`_init_luts()` (mpl import)** | **~145**  | **imports matplotlib on first render** |
| extract_slice (1024×1024)    | ~1        | pure numpy slicing                   |
| apply_colormap (1024×1024)    | ~5-20     | normalize + LUT lookup               |
| binary encode (4MB RGBA)      | <1        | struct.pack + tobytes                |

### Total startup budget: `view(arr)` → first pixel visible

| Phase                                   | Time (ms) | Cumulative |
|-----------------------------------------|-----------|------------|
| `import arrayview` (if not cached)      | ~350      | 350        |
| `_in_jupyter()` (IPython import)        | ~183      | 533        |
| `_in_vscode_terminal()` (process walk)  | ~26       | 559        |
| `Session(arr)` creation + stats         | ~10-25    | ~580       |
| server thread start + ready             | ~1        | ~581       |
| SERVER_LOOP polling (`time.sleep(0.01)`) | 10-50     | ~620       |
| `_open_webview` or `_open_browser`      | 200-2000  | ~800-2600  |
| Browser loads HTML + JS                 | 200-500   | ~1000-3100 |
| WebSocket connect + first request       | 10-50     | ~1050-3150 |
| `_init_luts()` (first render, mpl)      | ~145      | ~1200-3300 |
| Slice + colormap + binary encode        | ~20       | ~1220-3320 |
| Browser canvas render                   | ~5        | ~1225-3325 |

**Total: ~1.2–3.3s** depending on environment (native window slower, browser faster).

---

## Plan Items

### S1 — Lazy FastAPI import (save ~175ms on startup) ★★★

**Status**: not started
**Difficulty**: Medium
**Expected gain**: ~175ms off `import arrayview`

FastAPI is the single most expensive import (~175ms). It pulls in pydantic,
starlette, OpenAPI schema generation, etc. But it's only needed when the server
actually starts.

**Approach**: Don't import `_server.py` at module level. The `_app.py` compat
shim eagerly imports everything from `_server`. Instead:
1. Make `_app.py` defer `_server` imports behind a function or lazy module proxy.
2. In `_launcher.py`, import `_server` only when `_serve_background()` is called.
3. The HTML templates and route definitions only need to exist when uvicorn starts.

**Risk**: Tests that do `from arrayview._app import app` would need adjustment.
The compat shim pattern makes this tricky — need to preserve backward compat.

---

### S2 — Cache `_in_jupyter()` result (save ~183ms on repeat) ★★★

**Status**: not started
**Difficulty**: Low
**Expected gain**: ~183ms (first call only; prevent redundant imports)

`_in_jupyter()` imports IPython every time it's called. Within a single
`view()` call it's called at least twice (directly + via `_is_script_mode()`).
Cache the result after the first call.

Also: `_in_vscode_terminal()` (26ms) walks the process tree via `ps ewwww`
subprocess. This should be cached too.

**Approach**: Module-level sentinel + cache pattern (like `_VSCODE_IPC_HOOK_CACHE`).

---

### S3 — Generation counter for stale WebSocket requests ★★★

**Status**: not started
**Difficulty**: Low
**Expected gain**: eliminates wasted renders when scrolling fast

When the user scrubs a slider, each position fires a WebSocket request.
Currently all get rendered serially. A generation counter would let us skip
outdated requests.

**Approach**: (adapted from ndv)
1. Add `_gen_counter` and `_current_gen` to the WebSocket handler.
2. On each new request, increment `_current_gen`.
3. Before sending a render response, check if `gen == _current_gen`. If not, drop it.
4. The render thread can also check before starting expensive work.

---

### S4 — Server-side downsampling for large slices ★★☆

**Status**: not started
**Difficulty**: Low-Medium
**Expected gain**: proportional to oversized slice ratio (e.g., 4096×4096 → 1024×1024 = 16× less data)

If the extracted slice is larger than the display viewport (which the browser
sends via WebSocket), downsample before colormap application.

**Approach**:
1. Browser sends its canvas dimensions in the WebSocket message.
2. Server compares slice dimensions to canvas dimensions.
3. If slice is >2× canvas in either dimension, use `scipy.ndimage.zoom` or
   stride-based decimation before colormap.
4. Need to handle zoom correctly — when zoomed in, send full resolution for
   the visible region only.

---

### S5 — WebGL shader for colormap application ★★★ (biggest potential win)

**Status**: not started
**Difficulty**: High
**Expected gain**: 5-10× faster per-frame rendering + ~50% less WebSocket transfer

The single most impactful change. Instead of sending pre-rendered RGBA bytes,
send raw float32/uint16 data and let the browser's GPU apply the colormap.

**Approach (incremental)**:
1. Phase A: Add a WebGL canvas layer alongside the existing Canvas2D.
2. Phase B: Upload raw float32 data as a texture.
3. Phase C: Fragment shader applies normalization + 1D colormap texture lookup.
4. Phase D: Colormap changes become instant (just update the 1D texture).
5. Phase E: Zoom/pan can use GPU transforms without re-fetching data.

**Transfer format change**:
- Current: `header(20B) + RGBA(H×W×4)` — e.g., 1024×1024 = 4 MB
- New: `header(20B) + float32(H×W×4)` — same size for float32, but:
  - uint16 data: `H×W×2` = 2 MB (50% less)
  - With zlib: ~0.5-1 MB for smooth medical data

**Risk**: WebGL requires careful context management. Some edge cases (complex
modes, overlay compositing, mosaic) would still need CPU fallback initially.

---

### S6 — `ihist` for integer histogram computation ★★☆

**Status**: not started
**Difficulty**: Low
**Expected gain**: 2-5× faster auto-contrast for uint8/uint16 data

ndv uses the `ihist` library for C-optimized histograms of integer data.
arrayview computes percentiles via numpy sampling. For images from cameras
(uint8/uint16), `ihist` is significantly faster.

**Approach**:
1. Add `ihist` as optional dependency.
2. In `compute_global_stats()`, use `ihist.histogram()` for integer dtypes ≤16-bit.
3. Derive percentiles from histogram counts (CDF-based, like ndv does).

---

### S7 — WebSocket compression for raw data ★☆☆

**Status**: not started
**Difficulty**: Low
**Expected gain**: 2-3× smaller transfers for smooth/sparse data

Enable `permessage-deflate` on the WebSocket or manually compress with zlib.
Medical/scientific data often has spatial smoothness that compresses well.

**Approach**:
1. Server: optionally compress the binary payload with `zlib.compress(level=1)`.
2. Prefix message with a compression flag byte.
3. Browser: `pako.inflate()` (fast JS zlib) before rendering.
4. Make it adaptive: only compress if estimated gain > overhead.

**Risk**: For noisy data (e.g., raw MRI), compression ratio may be poor and
add latency. Need benchmarking.

---

### S8 — Reduce SERVER_LOOP polling ★★☆

**Status**: not started
**Difficulty**: Low
**Expected gain**: 10-50ms off startup

After starting the server thread, `view()` polls `SERVER_LOOP` with
`time.sleep(0.01)` in a loop until it's set. This adds 10-50ms of unnecessary
waiting.

**Approach**: Use a `threading.Event` (like `_server_ready_event`) for
`SERVER_LOOP` too. Set it in `_serve_background()` right after
`asyncio.get_event_loop()`.

---

## Priority Order

1. **S2** — Cache env detection (Low effort, ~183ms gain)
2. **S1** — Lazy FastAPI import (Medium effort, ~175ms gain)
3. **S8** — Event-based SERVER_LOOP (Low effort, ~10-50ms gain)
4. **S3** — Generation counter (Low effort, UX improvement)
5. **S6** — ihist for int histograms (Low effort, faster auto-contrast)
6. **S4** — Server-side downsampling (Medium effort, less data transfer)
7. **S7** — WebSocket compression (Low effort, smaller transfers)
8. **S5** — WebGL colormap shader (High effort, transformative)

Items S1–S3 + S8 could yield **~400ms** startup improvement with minimal risk.
Item S5 is a longer-term project that would fundamentally change the rendering
architecture.

---

## Validation

After each change, run:
```bash
uv run pytest tests/test_api.py -v      # Server/API tests
uv run pytest tests/test_cli.py -v       # CLI entry points
uv run python debug/profile_startup.py   # Re-measure timings
```

For WebSocket-related changes (S3, S5, S7):
```bash
uv run pytest tests/test_browser.py -v   # Browser/Playwright tests
```
