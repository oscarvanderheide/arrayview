"""Session management, global state, caches, render thread, and constants."""

import asyncio
import os
import queue as _queue
import threading
import uuid
from collections import OrderedDict

import numpy as np

# ---------------------------------------------------------------------------
# Verbose flag — set to True by --verbose CLI flag; suppresses debug prints.
# ---------------------------------------------------------------------------
_verbose = False


def _vprint(*args, **kwargs) -> None:
    """Print only when verbose mode is active."""
    if _verbose:
        print(*args, **kwargs)


# ---------------------------------------------------------------------------
# Session & Global State Management
# ---------------------------------------------------------------------------
SERVER_LOOP = None
SERVER_PORT: int | None = None  # actual port the uvicorn server is bound to
VIEWER_SOCKETS = 0  # count of active viewer WebSocket connections
VIEWER_SIDS: set = set()  # session IDs with at least one active viewer WS
SHELL_SOCKETS = []  # webview shell WS connections (for tab injection)
_window_process = None
PENDING_SESSIONS: set = set()  # sids whose data is still loading in a background thread

# ---------------------------------------------------------------------------
# Render thread — bypasses concurrent.futures so it is unaffected by
# Python's interpreter-shutdown executor cleanup (_global_shutdown flag).
# ---------------------------------------------------------------------------
_RENDER_QUEUE: "_queue.SimpleQueue[tuple | None]" = _queue.SimpleQueue()
_RENDER_THREAD: threading.Thread | None = None


def _render_worker() -> None:
    while True:
        item = _RENDER_QUEUE.get()
        if item is None:
            return
        func, fut, loop = item
        try:
            result = func()
            loop.call_soon_threadsafe(fut.set_result, result)
        except Exception as exc:
            loop.call_soon_threadsafe(fut.set_exception, exc)


def _ensure_render_thread() -> None:
    global _RENDER_THREAD
    if _RENDER_THREAD is not None and _RENDER_THREAD.is_alive():
        return
    _RENDER_THREAD = threading.Thread(
        target=_render_worker, daemon=True, name="arrayview-render"
    )
    _RENDER_THREAD.start()


# ---------------------------------------------------------------------------
# Neighbor prefetch thread pool (Phase 3)
# ---------------------------------------------------------------------------
_PREFETCH_POOL = None
_PREFETCH_LOCK = threading.Lock()


def _get_prefetch_pool():
    global _PREFETCH_POOL
    with _PREFETCH_LOCK:
        if _PREFETCH_POOL is None or _PREFETCH_POOL._shutdown:
            import concurrent.futures

            _PREFETCH_POOL = concurrent.futures.ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="arrayview-prefetch"
            )
        return _PREFETCH_POOL


def _schedule_prefetch(session, dim_x, dim_y, idx_list, slice_dim, direction):
    """Warm raw_cache for the next PREFETCH_NEIGHBORS slices in *direction*.

    Runs in a low-priority background thread so it never blocks WS responses.
    Skips prefetch when the estimated byte cost exceeds PREFETCH_BUDGET_BYTES.
    """
    # Import here to avoid circular dependency — extract_slice lives in _render
    from arrayview._render import extract_slice

    n = session.shape[slice_dim]
    slice_bytes = session.shape[dim_y] * session.shape[dim_x] * 4  # float32
    if slice_bytes * PREFETCH_NEIGHBORS > PREFETCH_BUDGET_BYTES:
        return  # data too large; skip prefetch to avoid memory pressure

    current = idx_list[slice_dim]
    targets = []
    for step in range(1, PREFETCH_NEIGHBORS + 1):
        nxt = current + direction * step
        if 0 <= nxt < n:
            targets.append(nxt)

    if not targets:
        return

    def _warm():
        for t in targets:
            idx = list(idx_list)
            idx[slice_dim] = t
            key = (dim_x, dim_y, tuple(idx))
            if key not in session.raw_cache:
                try:
                    extract_slice(session, dim_x, dim_y, idx)
                except Exception:
                    pass  # prefetch errors are silent

    try:
        _get_prefetch_pool().submit(_warm)
    except RuntimeError:
        pass  # pool shutting down


async def _render(loop: asyncio.AbstractEventLoop, func) -> object:
    """Await *func()* in the render thread without using concurrent.futures."""
    _ensure_render_thread()
    fut: asyncio.Future = loop.create_future()
    _RENDER_QUEUE.put((func, fut, loop))
    return await fut


def _percentile_pair(
    sample: "np.ndarray", orig_dtype: "np.dtype", lo: float, hi: float
) -> "tuple[float, float]":
    """Return (pct_lo, pct_hi) of *sample*, using a fast bincount path for
    integer source dtypes whose value range fits in <=65536 bins.

    For integer arrays (int8/uint8/int16/uint16) the full-range histogram is
    O(N) with a very small constant, avoiding the O(N log N) sort that
    np.percentile performs.
    """
    if orig_dtype.kind in ("i", "u") and orig_dtype.itemsize <= 2:
        imin = int(np.iinfo(orig_dtype).min)
        imax = int(np.iinfo(orig_dtype).max)
        int_sample = np.clip(sample, imin, imax).astype(np.int32) - imin
        counts = np.bincount(int_sample, minlength=imax - imin + 1)
        total = int(counts.sum())
        cumsum = np.cumsum(counts)
        lo_idx = int(np.searchsorted(cumsum, lo / 100.0 * total))
        hi_idx = int(np.searchsorted(cumsum, hi / 100.0 * total))
        return float(imin + lo_idx), float(imin + hi_idx)
    # General path: use numpy sort-based percentile
    f32 = sample.astype(np.float32)
    return float(np.percentile(f32, lo)), float(np.percentile(f32, hi))


class Session:
    def __init__(self, data, filepath=None, name=None):
        self.sid = uuid.uuid4().hex
        self.data = data
        self.shape = data.shape
        self.filepath = filepath
        self.name = name or (
            os.path.basename(filepath) if filepath else f"Array {data.shape}"
        )
        self.global_stats = {}
        self.fft_original_data = None
        self.fft_axes = None
        self.data_version = 0  # incremented by /reload when file is reloaded

        self.rgb_axis = None  # set by _setup_rgb(); axis index in actual shape
        self.spatial_shape = data.shape  # shape without rgb_axis (set by _setup_rgb)

        self.raw_cache = OrderedDict()
        self.rgba_cache = OrderedDict()
        self.mosaic_cache = OrderedDict()

        # Phase 5: adaptive budgets from module-level computed constants.
        # These default to a fraction of total RAM; override via env vars.
        self.RAW_CACHE_BYTES = _RAW_CACHE_BYTES
        self.RGBA_CACHE_BYTES = _RGBA_CACHE_BYTES
        self.MOSAIC_CACHE_BYTES = _MOSAIC_CACHE_BYTES
        self._raw_bytes = self._rgba_bytes = self._mosaic_bytes = 0
        self._estimated_mem = self._estimate_memory()

        self.preload_gen = 0
        self.preload_done = 0
        self.preload_total = 0
        self.preload_skipped = False
        self.preload_lock = threading.Lock()

        self.alpha_level = 0  # 0=off, 1=transparent below vmin

        self.vfield = None  # Optional deformation vector field
        self.vfield_component_dim = None  # axis holding xyz displacement components
        self.vfield_time_dim = None  # optional time axis in the raw vfield array
        self.vfield_spatial_axes = None  # image spatial dim -> vfield axis mapping

        self.compute_global_stats()

    def _estimate_memory(self):
        """Estimate memory footprint in bytes (array data + cache budgets)."""
        itemsize = np.dtype(getattr(self.data, "dtype", np.float32)).itemsize
        data_bytes = int(np.prod(self.shape)) * itemsize
        return data_bytes

    def compute_global_stats(self):
        try:
            total = int(np.prod(self.shape))
            max_samples = 200_000
            ndim = len(self.shape)
            orig_dtype = np.dtype(getattr(self.data, "dtype", np.float32))
            if total <= max_samples:
                sample = np.array(self.data).ravel()
            elif ndim >= 4:
                # For high-dimensional arrays (e.g. T×Z×Y×X×C), sampling full
                # outer-axis slices loads hundreds of MB each. Instead, sample
                # 20 random 2-D slices (last two dims) by fixing all outer axes
                # to random indices. Each 2-D slice is at most ~1 MB.
                rng = np.random.default_rng(0)
                outer_shape = self.shape[:-2]  # all dims except last two
                n_slices = 20
                chunks = []
                per_chunk = max(1000, max_samples // n_slices)
                for _ in range(n_slices):
                    idx = tuple(int(rng.integers(0, s)) for s in outer_shape)
                    chunk = np.array(self.data[idx]).ravel()
                    if chunk.size > per_chunk:
                        chunk = chunk[:: max(1, chunk.size // per_chunk)]
                    chunks.append(chunk)
                sample = np.concatenate(chunks)
            else:
                n_take = min(10, self.shape[0])
                step = max(1, self.shape[0] // n_take)
                per_chunk = max(1000, max_samples // n_take)
                chunks = []
                for i in range(0, self.shape[0], step):
                    chunk = np.array(self.data[i]).ravel()
                    if chunk.size > per_chunk:
                        chunk = chunk[:: max(1, chunk.size // per_chunk)]
                    chunks.append(chunk)
                sample = np.concatenate(chunks)
            if np.iscomplexobj(sample):
                sample = np.abs(sample)
            sample = np.nan_to_num(sample)

            self.global_stats = {
                i: _percentile_pair(sample, orig_dtype, lo, hi)
                for i, (lo, hi) in enumerate(DR_PERCENTILES)
            }
        except Exception:
            self.global_stats = {}


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


SESSIONS = {}

COLORMAPS = [
    "gray",
    "lipari",
    "navia",
    "viridis",
    "plasma",
    "RdBu_r",
    "twilight_shifted",
]
DR_PERCENTILES = [(0, 100), (1, 99), (5, 95), (10, 90)]
DR_LABELS = ["0-100%", "1-99%", "5-95%", "10-90%"]

# ---------------------------------------------------------------------------
# Zarr chunk presets (Phase 1)
# ---------------------------------------------------------------------------
ZARR_LARGE_XY_TILE = 1024  # pixels — threshold for tiling XY dimension
ZARR_T_DEPTH = 2  # T-frames per chunk for 4D+ arrays

# ---------------------------------------------------------------------------
# Neighbor prefetch (Phase 3)
# ---------------------------------------------------------------------------
PREFETCH_NEIGHBORS = 3  # slices to prefetch in the scroll direction
PREFETCH_BUDGET_BYTES = 16 * 1024 * 1024  # 16 MB max prefetch per request

# ---------------------------------------------------------------------------
# Heavy-operation guardrails (Phase 4)
# ---------------------------------------------------------------------------
_DEFAULT_HEAVY_OP_MB = 5000
HEAVY_OP_LIMIT_BYTES = (
    int(os.environ.get("ARRAYVIEW_HEAVY_OP_LIMIT_MB", _DEFAULT_HEAVY_OP_MB))
    * 1024
    * 1024
)


def _estimate_array_bytes(session) -> int:
    """Best-effort estimate of the total uncompressed byte size of session.data."""
    try:
        itemsize = np.dtype(session.data.dtype).itemsize
    except Exception:
        itemsize = 4  # assume float32
    return int(np.prod(session.shape)) * itemsize


# ---------------------------------------------------------------------------
# Adaptive cache budgets (Phase 5)
# ---------------------------------------------------------------------------


def _total_ram_bytes() -> int:
    """Return total system RAM in bytes, or 8 GB as a conservative fallback."""
    try:
        import psutil

        return psutil.virtual_memory().total
    except Exception:
        return 8 * 1024**3  # 8 GB fallback


def _cache_budget(env_var: str, fraction: float) -> int:
    """Return a cache byte budget."""
    env_val = os.environ.get(env_var)
    if env_val is not None:
        try:
            return max(1, int(env_val)) * 1024 * 1024
        except ValueError:
            pass  # bad value — fall through to adaptive
    ram = _total_ram_bytes()
    return max(64 * 1024 * 1024, int(ram * fraction))


# Compute once at import time so Sessions created in the same process share budgets.
_RAW_CACHE_BYTES = _cache_budget("ARRAYVIEW_RAW_CACHE_MB", 0.05)  # 5% RAM
_RGBA_CACHE_BYTES = _cache_budget("ARRAYVIEW_RGBA_CACHE_MB", 0.10)  # 10% RAM
_MOSAIC_CACHE_BYTES = _cache_budget("ARRAYVIEW_MOSAIC_CACHE_MB", 0.025)  # 2.5% RAM


def zarr_chunk_preset(shape: tuple[int, ...]) -> tuple[int, ...]:
    """Return a recommended Zarr chunk shape for *shape* optimised for
    interactive slice navigation in arrayview."""
    ndim = len(shape)
    if ndim < 2:
        return shape  # zarr default

    # XY tile size
    cy = min(shape[0], ZARR_LARGE_XY_TILE)
    cx = min(shape[1], ZARR_LARGE_XY_TILE)

    if ndim == 2:
        return (cy, cx)
    elif ndim == 3:
        return (cy, cx, 1)
    elif ndim == 4:
        return (cy, cx, 1, min(ZARR_T_DEPTH, shape[3]))
    elif ndim == 5:
        # (Y, X, Z, T, C) — keep full C axis in one chunk
        return (cy, cx, 1, 1, shape[4])
    else:
        # 6-D+: one element per extra axis, full XY
        extra = tuple(1 for _ in range(ndim - 2))
        return (cy, cx) + extra


__all__ = [
    # Verbose
    "_verbose",
    "_vprint",
    # Global state
    "SERVER_LOOP",
    "VIEWER_SOCKETS",
    "VIEWER_SIDS",
    "SHELL_SOCKETS",
    "_window_process",
    "PENDING_SESSIONS",
    # Render thread
    "_RENDER_QUEUE",
    "_RENDER_THREAD",
    "_render_worker",
    "_ensure_render_thread",
    "_render",
    # Prefetch
    "_PREFETCH_POOL",
    "_PREFETCH_LOCK",
    "_get_prefetch_pool",
    "_schedule_prefetch",
    # Session
    "Session",
    "SESSIONS",
    # Constants
    "COLORMAPS",
    "DR_PERCENTILES",
    "DR_LABELS",
    "ZARR_LARGE_XY_TILE",
    "ZARR_T_DEPTH",
    "PREFETCH_NEIGHBORS",
    "PREFETCH_BUDGET_BYTES",
    "HEAVY_OP_LIMIT_BYTES",
    "_DEFAULT_HEAVY_OP_MB",
    "_estimate_array_bytes",
    # Cache
    "_total_ram_bytes",
    "_cache_budget",
    "_RAW_CACHE_BYTES",
    "_RGBA_CACHE_BYTES",
    "_MOSAIC_CACHE_BYTES",
    # Public API
    "zarr_chunk_preset",
]
