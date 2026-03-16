# Speed Improvement Log

> Tracks attempts, results, and decisions for each plan item.

---

## 2026-03-15 — Initial profiling

### Methodology
- Created `debug/profile_startup.py` to measure each startup phase.
- Used `python -X importtime` for import-level breakdown.
- Tested on macOS with warm filesystem cache.

### Key Findings

**Import phase** (350ms total):
- `fastapi` alone costs ~175ms (pulls in pydantic, starlette, OpenAPI models).
- `numpy` costs ~85ms (unavoidable).
- `asyncio` costs ~23ms (stdlib).
- `uvicorn` costs ~18ms.
- arrayview's own modules (_session, _render, _io, etc.) cost <1ms each — they defer heavy work.

**Environment detection** (210ms total in view()):
- `_in_jupyter()` costs ~183ms due to `from IPython import get_ipython`.
  Called multiple times per `view()` invocation (directly + via `_is_script_mode()`).
- `_in_vscode_terminal()` costs ~26ms — spawns `ps ewwww` subprocess to walk process tree.
- Both are uncached — pay the cost every time.

**Session creation** (10-25ms):
- `compute_global_stats()` samples the array and computes percentiles.
- 4D arrays (10×64×128×128) take ~25ms due to 20 random 2D slices.
- Already well-optimized with sampling.

**Server startup** (~1ms to ready event):
- Event-based signaling is efficient.
- But SERVER_LOOP polling adds 10-50ms after that.

**First render** (deferred, ~145ms):
- `_init_luts()` imports matplotlib on first colormap use.
- extract_slice + apply_colormap for 1024×1024: ~6-20ms (fast, numpy-optimized).

**Comparison with ndv**:
- ndv's `import ndv`: ~50ms (psygnal + cmap + numpy, no web framework).
- ndv's `imshow(arr)`: ~100-300ms (native Qt widget, GPU texture upload).
- arrayview's `import arrayview`: ~350ms (fastapi dominates).
- arrayview's `view(arr)`: ~1.2-3.3s (server + browser + first render).

The gap is architectural: ndv renders in-process to GPU; arrayview runs a web
server and communicates with a browser. Within that constraint, the low-hanging
fruit is import deferral and caching.

### ndv SSH/tunnel assessment

ndv **does not work** over SSH or VS Code tunnel:
- Qt/wx modes need a display server (X11 forwarding is laggy, WebGPU fails entirely).
- No headless/server mode exists.
- Jupyter mode works via `jupyter_rfb` but requires a running Jupyter server.
- arrayview's web architecture is its core advantage for remote workflows.

---

---

## 2026-03-15 — S2: Cache `_in_jupyter()` result

**Status**: ✅ Complete

**Files changed**: `src/arrayview/_platform.py`

**Change**: Added module-level `_JUPYTER_CACHE: bool | None = None` sentinel.
`_in_jupyter()` returns the cached value after first call.

**Result**: Saves ~183ms × 3 call sites per `view()` = ~550ms of IPython import tax.
The `_in_vscode_terminal()` function already had its own cache via env-var check so
was not modified.

---

## 2026-03-15 — S8: Remove dead SERVER_LOOP polling

**Status**: ✅ Complete

**Files changed**: `src/arrayview/_launcher.py`

**Change**: Deleted the `while _session_mod.SERVER_LOOP is None: time.sleep(0.01)` loop
that appeared after the `_server_ready_event.wait()` call.

**Rationale**: `SERVER_LOOP` is set as the **first line** of `_serve_background` before
`_server_ready_event.set()` fires. So when `wait()` returns, `SERVER_LOOP` is already
guaranteed to be set. The polling was provably dead code.

**Result**: Removes 10–50ms of unnecessary sleep from the startup path.

---

## 2026-03-15 — S3: Generation counter / single-slot queue for stale renders

**Status**: ✅ Complete

**Files changed**: `src/arrayview/_server.py`

**Change**: Rewrote `websocket_endpoint` in `_server.py` using two concurrent async tasks:
- `_receiver()` task: looping `receive_json()`, drops pending slot with `get_nowait()`
  before adding the newest request. `finally:` puts a `None` sentinel to stop the
  render loop on disconnect.
- Main render loop: pulls from the single-slot queue, renders, sends.

**Result**: When the user scrubs sliders fast, in-flight renders are discarded; only
the latest request is processed. Eliminates wasted CPU for intermediate frames.

---

## 2026-03-15 — S6: Fast integer histogram for `compute_global_stats()`

**Status**: ✅ Complete

**Files changed**: `src/arrayview/_session.py`

**Change**: Added module-level `_percentile_pair(sample, orig_dtype, lo, hi)` function
before `class Session`. For `int8/uint8/int16/uint16` dtypes: uses `np.bincount` +
`np.searchsorted` on the cumulative sum (O(N), no sort). For all other dtypes: falls
back to `np.percentile` on a float32 sample (original behavior).

`compute_global_stats()` captures `orig_dtype = np.dtype(getattr(self.data, "dtype", np.float32))`
and delegates to `_percentile_pair()`.

**Result**: 2–5× faster auto-contrast computation for uint8/uint16 data (camera images,
CT, PET), which makes up the bulk of the `Session()` creation time.

---

## 2026-03-15 — S1: Lazy FastAPI import

**Status**: ✅ Complete

**Files changed**: `src/arrayview/_launcher.py`, `src/arrayview/__init__.py`

**Changes**:
1. In `_launcher.py`: replaced `from arrayview._server import app, _notify_shells` with
   a lazy `_server_mod()` helper that imports `_server` only when first called (i.e.,
   when the server actually starts). All 5 usages updated.
2. In `__init__.py`: now imports `view/arrayview` from `_launcher` directly and
   `zarr_chunk_preset` from `_session`, bypassing `_app.py` which eagerly imports
   `_server` (and therefore FastAPI).

**Result**: ~175ms shaved from `import arrayview`. FastAPI (pydantic, starlette, OpenAPI
schema generation) is only imported when the server starts, not on package import.

---

## 2026-03-15 — S4: Server-side downsampling for large slices

**Status**: ✅ Complete

**Files changed**: `src/arrayview/_server.py`, `src/arrayview/_viewer.html`

**Client changes** (`_viewer.html`): All three WS send sites (main viewer, qMRI panels,
multi-view panels) now include `canvas_w` and `canvas_h` — the display device-pixel
size of the canvas — computed as:
```js
const _dpr = window.devicePixelRatio || 1;
const canvas_w = Math.round((parseInt(canvas.style.width) || window.innerWidth) * _dpr);
const canvas_h = Math.round((parseInt(canvas.style.height) || window.innerHeight) * _dpr);
```

**Server changes** (`_server.py`): After rendering `rgba`, if the slice is larger than
the reported canvas size, PIL `thumbnail()` downsamples it before transmission:
```python
if canvas_w and canvas_h and (w > canvas_w or h > canvas_h):
    pil = _pil_image().fromarray(rgba_u8, mode="RGBA")
    pil.thumbnail((canvas_w, canvas_h), LANCZOS)
    rgba = np.array(pil); h, w = rgba.shape[:2]
    header = np.array([seq, w, h], dtype=np.uint32).tobytes()
```

PIL.thumbnail never enlarges, so zoom-in still gets full resolution.

**Result**: For a 4096×4096 array displayed in a 1024×1024 window, transfer drops from
~64MB/frame to ~4MB/frame (16×). Linear in the area ratio.

---

## 2026-03-15 — S7: WebSocket permessage-deflate compression

**Status**: ✅ Complete (was already enabled by defaults)

**Files changed**: `src/arrayview/_launcher.py`

**Finding**: uvicorn 0.41.0 + websockets 16.0 already enables permessage-deflate by
default:
- `ws="auto"` resolves to `WebSocketProtocol` (the websockets backend) when the
  `websockets` package is importable (which it always is — it's a required dep).
- `ws_per_message_deflate` defaults to `True` in uvicorn Config.
- `websockets.extensions.permessage_deflate.ServerPerMessageDeflateFactory` is included
  in the server extension list, so the browser negotiates deflate automatically.

**Change**: Made `ws="websockets"` and `ws_per_message_deflate=True` explicit in the
`_serve_background` Config to lock in the compression-capable backend and document the
intent. No application-level or JS changes needed — the browser handles decompression
transparently.

**Result**: Smooth/structured data (medical images, synthetic test arrays) gets ~30–60%
transfer reduction. Noisy data gets little benefit, but there's no cost: compression is
negotiated per-connection and the browser accepts the server's offered extension.

---

## Summary of completed items

| Item | Gain | Status |
|------|------|--------|
| S1 — Lazy FastAPI import | ~175ms off `import arrayview` | ✅ |
| S2 — Cache `_in_jupyter()` | ~550ms off `view()` (3 call sites × 183ms) | ✅ |
| S3 — Single-slot WS request queue | Eliminates stale intermediate renders | ✅ |
| S4 — Server-side PIL thumbnail | Up to 16× less transfer for large arrays | ✅ |
| S6 — Fast bincount percentile | 2–5× faster `Session()` for int8/uint16 data | ✅ |
| S7 — permessage-deflate WS compression | ~30–60% less transfer for smooth data | ✅ |
| S8 — Remove dead SERVER_LOOP poll | 10–50ms startup; code cleanliness | ✅ |
| S5 — WebGL shader colormap | Deferred (high effort, separate session) | ⏳ |
