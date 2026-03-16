# Chunked Large-Array Support Plan (NumPy + Zarr)

## 1. Considerations First

### 1.1 Product goals
- Keep `arrayview` smooth for arrays from a few GB up to tens of GB.
- Preserve current behavior for existing users and formats.
- Avoid regressions across local CLI, Python API, Jupyter, VS Code local, and VS Code remote/tunnel.

### 1.2 User interaction model (drives chunk strategy)
- Primary interaction is 2D slice viewing with scroll on one axis.
- Secondary interactions are axis swaps, playback along slice dimension, compare mode, and overlay/vectorfield display.
- Latency sensitivity:
1. First frame should appear quickly.
2. Scroll should feel continuous.
3. Playback should degrade gracefully (lower FPS) instead of stalling UI.

### 1.3 Technical constraints in current codebase
- `.npy` already uses `np.load(..., mmap_mode="r")`.
- Slice extraction is already on-demand and cached.
- Some features force broad materialization and can blow memory/latency on huge data:
1. FFT path materializes full array.
2. Grid/GIF operations stack many slices.
3. Full preload is skipped above estimated threshold.
- Current cache limits are fixed, not adaptive.
- Frontend loading overlay is mostly initial-load UX, not per-slice latency UX.

### 1.4 Performance constraints to explicitly target
- Storage variability: NVMe vs network filesystems.
- CPU variability: decompression cost vs I/O savings.
- Memory ceilings: avoid unbounded growth in session/cache paths.

### 1.5 Format strategy constraints
- Must support both classic NumPy pipelines and chunked pipelines.
- Do not force migration to Zarr for all users.
- Large-array users need a documented and predictable “best path.”

### 1.6 Non-negotiables
- No regressions in VS Code Simple Browser and remote/tunnel routing behavior.
- No manual cleanup burden.
- Keep local workflows simple (`arrayview file.npy` continues to work).

## 2. Architecture Direction

### 2.1 Dual-path data access model
- Path A: `.npy` + memmap for straightforward lazy reads with low complexity.
- Path B: `.zarr` for interaction-optimized chunked access and better random slicing behavior.

### 2.2 Recommended workload mapping
- Use `.npy` when:
1. Access is mostly sequential in a predictable dimension.
2. Dataset size is large but still performs acceptably with memmap on local fast disk.
- Use `.zarr` when:
1. Users frequently swap axes and jump around.
2. Datasets are tens of GB and smooth random slice access matters.
3. Storage is remote or high-latency.

### 2.3 Chunking design principle
- Chunk to match interaction, not mathematical symmetry.
- Keep displayed dimensions broad, scrolled dimension shallow.
- Initial target chunk byte size: 1-8 MB uncompressed (allow up to ~16 MB on fast systems).

## 3. Detailed Execution Plan

## Phase 0: Baseline and Logging
- Create `LOG_LARGE_ARRAYS.md`.
- Record each attempt with hypothesis, change, result, decision.
- Build representative benchmark set:
1. 3D scalar volume.
2. 4D volume/time or channel data.
3. Large compare use case (2-3 arrays).
4. One remote filesystem scenario if available.
- Capture baseline metrics for current main branch:
1. First-frame latency.
2. p50/p95 scroll latency.
3. Playback FPS.
4. Peak RSS and cache bytes.
5. CPU utilization during scroll/playback.

## Phase 1: Define and Document Chunk Presets
- Specify chunk presets by shape archetype:
1. 3D volume `(Y, X, Z)`: example `(512, 512, 1)` or `(1024, 1024, 1)` depending on pixel size.
2. 4D `(Y, X, Z, T)`: example `(512, 512, 1, 2-4)`.
3. If T-playback is dominant, increase T depth and reduce XY chunk if needed.
- Define compression defaults for Zarr:
1. Start with lightweight compressor to minimize decode stalls.
2. Keep option for no compression when storage is very fast and CPU is constrained.
- Add clear guidance in docs:
1. “When to keep `.npy`.”
2. “When to convert to `.zarr`.”
3. “How to pick chunk shapes by interaction.”

## Phase 2: User-Facing Conversion Workflow
- Add a documented conversion recipe from `.npy`/NIfTI to `.zarr`.
- Include optional rechunk step and examples per shape archetype.
- Include verification checklist:
1. Shape/dtype parity.
2. Slice value parity spot checks.
3. Compression ratio and resulting storage size.

## Phase 3: Runtime Behavior Improvements for Large Arrays
- Add per-slice “loading” indicator behavior:
1. Only show when render exceeds a small threshold to avoid flicker.
2. Hide immediately on frame receipt.
- Keep latest-request-wins behavior to avoid queue buildup while scrolling.
- Add bounded neighbor prefetch:
1. Direction-aware (`+/-` around current index).
2. Respect strict memory budget.
3. Auto-throttle when latency rises.

## Phase 4: Guardrails for Heavy Operations
- Add explicit large-array guardrails/warnings for:
1. FFT.
2. GIF/grid generation.
3. Full-dimension preloads.
- Offer safe fallback behavior:
1. Cancel heavy action with clear status message.
2. Allow explicit override for advanced users.

## Phase 5: Cache Policy and Memory Budgeting
- Make cache budgets configurable (env var or settings).
- Add optional adaptive defaults based on detected RAM.
- Define hard ceilings per session and total process budget.
- Improve visibility:
1. Debug endpoint or logs for cache hit/miss and byte usage.
2. Include prefetch hit rate and eviction churn metrics.

## Phase 6: Validation and Regression Testing
- Extend automated tests:
1. API tests for `.zarr` and `.npy` parity on slice outputs.
2. Browser tests for loading indicator behavior under artificial delay.
3. Tests for guardrail behavior on large synthetic shapes.
- Run validation matrix:
1. Local CLI.
2. Python `view(arr)` and `view(zarr_array)`.
3. Jupyter inline.
4. VS Code terminal local.
5. VS Code remote/tunnel.
- Add manual smoke checklist for large arrays:
1. Scroll responsiveness.
2. Axis swap responsiveness.
3. Compare mode with large datasets.
4. No orphan server processes after close.

## Phase 7: Rollout Strategy
- Release as staged rollout:
1. Documentation + conversion first.
2. UX/caching/prefetch improvements next.
3. Guardrails + configurability final.
- Add release notes with:
1. Recommended format by dataset profile.
2. Known tradeoffs.
3. Tuning checklist for advanced users.

## 4. Acceptance Criteria
- Large dataset workflows (tens of GB) function without OOM in normal slice navigation.
- p95 scroll latency meets target in benchmark environments.
- First-frame latency remains acceptable on both `.npy` memmap and `.zarr`.
- No regressions in VS Code local/remote opening behavior.
- Heavy operations are clearly bounded or warned when unsafe.

## 5. Risks and Mitigations
- Risk: Bad chunk shapes cause many-chunk reads and stutter.
- Mitigation: Presets + benchmark-driven tuning + documented heuristics.

- Risk: Compression improves I/O but hurts CPU latency.
- Mitigation: Offer lightweight/no-compression profiles and measure p95 latency.

- Risk: Prefetch increases memory pressure.
- Mitigation: Hard prefetch budget and eviction policy.

- Risk: Feature regressions in compare/overlay paths.
- Mitigation: Targeted integration tests for those modes with large arrays.

## 6. Open Decisions (to resolve before implementation)
- Final latency targets per environment (local NVMe vs remote FS).
- Default chunk presets per modality/workflow.
- Default compressor settings.
- Whether large-array guardrails are soft warnings or hard blocks by default.
