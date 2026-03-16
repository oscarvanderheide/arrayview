# Large-Array Support Log

Tracks every significant attempt at chunked large-array support per CHUNK_PLAN.md.

Format per entry:
- **Hypothesis**: what we expect
- **Change**: what was done
- **Result**: measured outcome
- **Decision**: keep / revert / follow-up

---

## Phase 0 — Baseline and Logging

### 0.1 Environment

- Date: 2026-03-11
- Machine: Apple M-series (arm64), 17.2 GB RAM
- Python: managed by `uv`
- zarr: 3.1.5
- numpy: 2.4.2
- Storage: local NVMe (MacBook internal SSD)

### 0.2 Benchmark Datasets

| ID | Shape | dtype | Disk size | Notes |
|----|-------|-------|-----------|-------|
| bench_3d_small | (256, 256, 100) | float32 | 26 MB | fast baseline |
| bench_3d_large | (512, 512, 300) | float32 | 315 MB | memmap territory |
| bench_4d | (256, 256, 50, 10) | float32 | 131 MB | 4D scroll |

Both `.npy` (memmap) and `.zarr` variants created for each.
Zarr chunk presets used: `(Y, X, 1)` for 3D; `(Y, X, 1, 2)` for 4D.

### 0.3 Benchmark Script

See `benchmarks/bench_baseline.py`.
Methodology: cold-cache sequential scroll (raw_cache cleared before each slice), measures `extract_slice` wall time.

### 0.4 Baseline Metrics (current main, pre-any-change)

| Dataset | First-frame ms | p50 scroll ms | p95 scroll ms | Peak ΔRSS MB | Cache MB |
|---------|---------------|---------------|---------------|-------------|---------|
| bench_3d_small.npy | 1.6 | 0.2 | 0.3 | 24.9 | 0.3 |
| bench_3d_small.zarr | 1.1 | 0.9 | 1.1 | 0.0 | 0.3 |
| bench_3d_large.npy | 15.8 | 1.2 | 1.6 | 307.7 | 1.0 |
| bench_3d_large.zarr | 2.6 | 1.7 | 1.9 | 0.0 | 1.0 |
| bench_4d.npy | 6.0 | 0.3 | 6.0 | 125.3 | 0.3 |
| bench_4d.zarr | 1.4 | 1.1 | 1.4 | 0.0 | 0.3 |

### 0.5 Key Findings from Baseline

**`.npy` memmap**
- First-frame is slow for large arrays (15.8 ms for 512×512×300) because the OS must page-fault the mmap region.
- Steady-state p50 is very fast (0.2–1.2 ms) once pages are warm.
- RSS grows by the full file size on first access (307 MB for the large 3D case) — OS pages the file in.
- p95 spikes can equal first-frame (4D: p95 = 6.0 ms = first-frame) because the last dim only has 10 slices, so the cache never warms up across runs.

**`.zarr` (chunk preset: `(Y,X,1)`)**
- First-frame is consistently fast (1–2.6 ms) — only one chunk is fetched.
- Steady-state p50 is slightly slower than warm `.npy` (0.9–1.7 ms vs 0.2–1.2 ms) because zarr decompresses each chunk.
- RSS delta is effectively zero — only the requested chunk enters RAM.
- Much more predictable (low variance); p95 ≈ p50.

**Verdict**: `.zarr` with `(Y,X,1)` chunking is better for large random-access workloads. `.npy` wins on warm sequential reads but pays a large first-frame and RSS cost.

### 0.6 Known Hazards Confirmed (code audit)

1. **FFT** (`toggle_fft`, line 1460): `np.array(session.data)` — full array materialization. No guard.
2. **GIF** (`get_gif`, ~line 1855): iterates all slices and stacks them. No size check.
3. **Grid** (`get_grid`, ~line 1799): same as GIF. No size check.
4. **Preload** (`_run_preload`, line 876): skipped if estimated RGBA > 500 MB, but estimate ignores chunk layout.
5. **`compute_global_stats`** (line 312): for small arrays calls `np.array(self.data).ravel()` — full eager load (safe for ≤200k elements, uses sampling above that).

---

## Phase 1 — Chunk Presets (not started)

## Phase 2 — Conversion Workflow (not started)

## Phase 3 — Runtime Behavior Improvements (not started)

## Phase 4 — Guardrails (not started)

## Phase 5 — Cache Policy (not started)

## Phase 6 — Tests (not started)

## Phase 7 — Rollout (not started)

---

## Phase 1 — Chunk Presets (complete)

### 1.1 What was done
- Added `zarr_chunk_preset(shape)` function to `_app.py` (after `DR_LABELS`).
- Added `ZARR_LARGE_XY_TILE = 1024` and `ZARR_T_DEPTH = 2` module-level constants.
- Exported `zarr_chunk_preset` from `arrayview.__init__` for user scripts.

### 1.2 Preset table

| Shape | Chunks | Rationale |
|-------|--------|-----------|
| `(Y, X)` | `(Y, X)` | 2D — single chunk |
| `(Y, X, Z)` | `(Y, X, 1)` | One Z-slice per fetch |
| `(Y, X, Z, T)` | `(Y, X, 1, 2)` | One Z, 2 T-frames |
| `(Y, X, Z, T, C)` | `(Y, X, 1, 1, C)` | Full channel axis |
| XY > 1024 px | clamp to 1024 | Keeps chunk ≤ 4 MB |

### 1.3 Decision: keep

---

## Phase 2 — Conversion Workflow (complete)

### 2.1 What was done
- Created `docs/large-arrays.md` with:
  - When to keep `.npy` vs use `.zarr`
  - Conversion recipes (`.npy`, NIfTI, with/without compression)
  - Verification checklist
  - Known tradeoffs table
  - Tuning guide for advanced users

### 2.2 Decision: keep

---

## Phase 3 — Runtime Behavior Improvements (complete)

### 3.1 Per-slice loading indicator with threshold

**Change**: Added `WS_SLOW_THRESHOLD_MS = 80` in frontend. Spinner now only appears
80ms after a WS request is sent and only if no response has arrived yet.
Timer is cancelled immediately on any WS message.

**Result**: On local NVMe (p95 ≈ 2ms), spinner never appears during normal scroll.
On slow storage or large arrays it still appears — just not on fast paths.

**Decision**: keep

### 3.2 Neighbor prefetch

**Change**: 
- Added `PREFETCH_NEIGHBORS = 3` and `PREFETCH_BUDGET_BYTES = 16 MB` constants.
- Added `_schedule_prefetch()` function using a 1-worker `ThreadPoolExecutor`.
- WS handler now reads `direction` and `slice_dim` from the JS message and calls `_schedule_prefetch` after sending the frame.
- Prefetch is skipped when `slice_bytes * PREFETCH_NEIGHBORS > PREFETCH_BUDGET_BYTES` (avoids pressure on large arrays).

**Result**: Neighboring slices are warmed in `raw_cache` in the background.
The 3-neighbor budget at 16 MB max means prefetch is active for planes up to ~1340×1340 px.

**Decision**: keep

---

## Phase 4 — Guardrails (complete)

### 4.1 FFT guardrail

**Change**: `toggle_fft` now calls `_estimate_array_bytes(session)` before materialising
the array. Returns `{"error": "...", "too_large": True}` if > `HEAVY_OP_LIMIT_BYTES` (500 MB).
Frontend already renders `.error` as a status message.

### 4.2 GIF/Grid guardrail

**Change**: `get_gif` and `get_grid` estimate `frame_bytes * n_slices` before stacking.
Return HTTP 400 JSON with `{"error": "...", "too_large": True}` if over limit.
`saveGif()` in the frontend now checks `res.ok` and shows the error message.

### 4.3 Config

- Override threshold: `ARRAYVIEW_HEAVY_OP_LIMIT_MB` env var (default 500).

### 4.4 Decision: keep

---

## Phase 5 — Cache Policy (complete)

### 5.1 What was done
- Added `_cache_budget(env_var, fraction)` helper and `_total_ram_bytes()`.
- Module-level constants `_RAW_CACHE_BYTES`, `_RGBA_CACHE_BYTES`, `_MOSAIC_CACHE_BYTES`
  computed once at import from 5%/10%/2.5% of total RAM.
- Session `__init__` now uses these adaptive values instead of hardcoded constants.
- Override env vars: `ARRAYVIEW_RAW_CACHE_MB`, `ARRAYVIEW_RGBA_CACHE_MB`, `ARRAYVIEW_MOSAIC_CACHE_MB`.
- Added `/cache_info/{sid}` endpoint returning entries, used/budget bytes for all three caches.

### 5.2 Observed budgets on this machine (17.2 GB RAM)
- raw: 859 MB (was 512 MB)
- rgba: 1718 MB (was 1024 MB)
- mosaic: 429 MB (was 256 MB)

### 5.3 Decision: keep

---

## Phase 6 — Tests (complete)

### 6.1 New test file: `tests/test_large_arrays.py`

22 tests covering:
- `zarr_chunk_preset` for all shape archetypes (9 tests)
- `.zarr`/`.npy` parity — pixel values and metadata (2 tests)
- FFT, GIF, grid guardrails + small-array pass-through (4 tests)
- `/cache_info` endpoint (4 tests)
- Adaptive cache env-var override (3 tests)

All 67 tests pass (42 existing + 3 CLI + 22 new).

### 6.2 Decision: keep

---

## Phase 7 — Rollout Strategy (complete — see docs/large-arrays.md)
