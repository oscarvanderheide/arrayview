import argparse
import asyncio
import io
import json
import math
import os
import queue as _queue
import re
import socket
import sys
import time
import threading
import subprocess
import uuid
import urllib.parse
import urllib.request
import zipfile
from collections import OrderedDict
from importlib.resources import files as _pkg_files

import numpy as np
import nibabel as nib
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response, WebSocket
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.websockets import WebSocketDisconnect
from PIL import Image
from matplotlib import colormaps as mpl_colormaps
import qmricolors  # registers lipari, navia colormaps with matplotlib  # noqa: F401


# ---------------------------------------------------------------------------
# Subprocess GUI Launcher
# ---------------------------------------------------------------------------
_ICON_PNG_PATH: str | None = None


def _get_icon_png_path() -> str | None:
    """Generate the ArrayView logo as a 512×512 PNG (with padding) and cache the path."""
    global _ICON_PNG_PATH
    if _ICON_PNG_PATH is not None:
        return _ICON_PNG_PATH
    try:
        import tempfile
        from PIL import ImageDraw

        # 512×512 canvas with ~10% transparent padding so the icon sits at the
        # same visual size as other macOS dock icons.
        canvas = 512
        pad = int(canvas * 0.10)  # ~51px transparent margin each side
        inner = canvas - 2 * pad  # ~410px icon content area
        S = inner / 16  # scale factor for the 16-unit SVG design
        img = Image.new("RGBA", (canvas, canvas), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        # Background with rounded corners
        d.rounded_rectangle(
            [pad, pad, pad + inner - 1, pad + inner - 1],
            radius=int(2.5 * S),
            fill="#0c0c0c",
        )
        # 3×3 grid of coloured squares
        colors = [
            "#3a0ca3",
            "#560bad",
            "#c77dff",
            "#4361ee",
            "#4cc9f0",
            "#f5c842",
            "#4895ef",
            "#80ed99",
            "#f8961e",
        ]
        xs = [int(pad + 2 * S), int(pad + 6.5 * S), int(pad + 11 * S)]
        ys = [int(pad + 2 * S), int(pad + 6.5 * S), int(pad + 11 * S)]
        sq = int(3 * S)
        for row, y in enumerate(ys):
            for col, x in enumerate(xs):
                d.rectangle([x, y, x + sq - 1, y + sq - 1], fill=colors[row * 3 + col])
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        img.save(tmp.name, "PNG")
        tmp.close()
        _ICON_PNG_PATH = tmp.name
    except Exception:
        _ICON_PNG_PATH = ""
    return _ICON_PNG_PATH or None


def _open_webview(
    url: str, win_w: int, win_h: int, capture_stderr: bool = False
) -> subprocess.Popen:
    """Launch pywebview in a fresh subprocess. Uses subprocess.Popen to avoid
    multiprocessing bootstrap errors when called from a Jupyter kernel."""
    icon_path = _get_icon_png_path() or ""
    script = "\n".join(
        [
            "import sys, webview",
            "u, w, h, icon = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]",
            "webview.create_window('ArrayView', u, width=w, height=h, background_color='#111111')",
            "kw = {'gui': 'qt'} if sys.platform.startswith('linux') else {}",
            "if icon:",
            "    if sys.platform == 'darwin':",
            "        def _set_icon():",
            "            try:",
            "                import AppKit",
            "                from PyObjCTools import AppHelper",
            "                img = AppKit.NSImage.alloc().initWithContentsOfFile_(icon)",
            "                AppHelper.callAfter(AppKit.NSApplication.sharedApplication().setApplicationIconImage_, img)",
            "            except Exception: pass",
            "        kw['func'] = _set_icon",
            "    else:",
            "        kw['icon'] = icon",
            "webview.start(**kw)",
        ]
    )
    return subprocess.Popen(
        [sys.executable, "-c", script, url, str(win_w), str(win_h), icon_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE if capture_stderr else subprocess.DEVNULL,
    )


def _open_webview_with_fallback(url: str, win_w: int, win_h: int) -> subprocess.Popen:
    """Launch pywebview, falling back to _open_browser if the subprocess exits immediately
    OR if no viewer WebSocket connects within ~10 s (catches macOS non-framework Python
    zombies that start but show nothing).

    Used from view() (Python API) where the host process stays alive.
    """
    proc = _open_webview(url, win_w, win_h, capture_stderr=True)
    print(f"[ArrayView] Launching native window (pid={proc.pid})...", flush=True)
    sockets_before = VIEWER_SOCKETS  # capture count so we detect a NEW connection

    def _read_stderr():
        try:
            return proc.stderr.read().decode(errors="replace").strip()
        except Exception:
            return ""

    def _watchdog():
        # Phase 1: watch for an immediate crash (2 s)
        for _ in range(20):
            time.sleep(0.1)
            if proc.poll() is not None:
                stderr_out = _read_stderr()
                print(
                    f"[ArrayView] Native window exited immediately (code {proc.returncode}), opening in browser",
                    flush=True,
                )
                if stderr_out:
                    print(f"[ArrayView] webview stderr: {stderr_out}", flush=True)
                _open_browser(url)
                return

        # Phase 2: process is alive — wait up to 8 s for a NEW viewer WebSocket to connect.
        # We compare against sockets_before so an already-open browser tab doesn't
        # falsely confirm that the native window launched successfully.
        for _ in range(80):
            time.sleep(0.1)
            if VIEWER_SOCKETS > sockets_before:
                print("[ArrayView] Native window connected successfully", flush=True)
                if sys.platform == "darwin":
                    subprocess.Popen(
                        [
                            "osascript",
                            "-e",
                            f'tell application "System Events" to set frontmost of'
                            f" (first process whose unix id is {proc.pid}) to true",
                        ],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                return
            if proc.poll() is not None:
                stderr_out = _read_stderr()
                print(
                    f"[ArrayView] Native window exited (code {proc.returncode}), opening in browser",
                    flush=True,
                )
                if stderr_out:
                    print(f"[ArrayView] webview stderr: {stderr_out}", flush=True)
                _open_browser(url)
                return

        # Phase 3: alive but no UI connection after 10 s — zombie (e.g. non-framework Python on macOS)
        print(
            "[ArrayView] Native window did not connect; falling back to browser",
            flush=True,
        )
        try:
            proc.terminate()
        except Exception:
            pass
        _open_browser(url)

    threading.Thread(target=_watchdog, daemon=True).start()
    return proc


def _open_webview_cli(url: str, win_w: int, win_h: int) -> bool:
    """Launch pywebview from the CLI and synchronously wait to detect an immediate crash.

    Returns True if the window appears to have started (still alive after 2 s).
    Returns False if it crashed; in that case the caller should fall back to browser.
    The CLI process must not exit while the daemon-thread watchdog is still pending,
    so the wait is done synchronously here.
    """
    print("[ArrayView] Launching native window (PyWebView)...", flush=True)
    proc = _open_webview(url, win_w, win_h, capture_stderr=True)
    for _ in range(20):
        time.sleep(0.1)
        if proc.poll() is not None:
            stderr_out = ""
            try:
                stderr_out = proc.stderr.read().decode(errors="replace").strip()
            except Exception:
                pass
            print(
                f"[ArrayView] Native window exited immediately (code {proc.returncode})",
                flush=True,
            )
            if stderr_out:
                print(f"[ArrayView] webview stderr: {stderr_out}", flush=True)
            return False
    print("[ArrayView] Native window started successfully", flush=True)
    return True


# ---------------------------------------------------------------------------
# Session & Global State Management
# ---------------------------------------------------------------------------
SERVER_LOOP = None
VIEWER_SOCKETS = 0  # count of active viewer WebSocket connections
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

        self.preload_gen = 0
        self.preload_done = 0
        self.preload_total = 0
        self.preload_skipped = False
        self.preload_lock = threading.Lock()

        self.mask_level = 0  # 0=off, 1=Otsu, 2=2×Otsu
        self.mask_otsu = None  # cached Otsu threshold (float)
        self.mask_threshold = 0.0  # active threshold applied to rendering

        self.vfield = None  # Optional deformation vector field: (*spatial_shape, 3)

        self.compute_global_stats()

    def compute_global_stats(self):
        try:
            total = int(np.prod(self.shape))
            max_samples = 200_000
            if total <= max_samples:
                sample = np.array(self.data).ravel()
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
            sample = np.nan_to_num(sample).astype(np.float32)

            self.global_stats = {
                i: (float(np.percentile(sample, lo)), float(np.percentile(sample, hi)))
                for i, (lo, hi) in enumerate(DR_PERCENTILES)
            }
        except Exception:
            self.global_stats = {}


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
# These presets are designed for interactive slice navigation, where the
# primary interaction is scrolling along one axis while displaying a 2D plane.
#
# Design principle: keep the displayed (XY) dimensions full-size in each chunk
# so a single chunk fetch satisfies one frame.  Keep the scrolled dimension
# shallow (depth=1) so loading one slice does not pull in neighbours.
#
# Presets by array shape archetype:
#   3D (Y, X, Z)        → chunks (Y, X, 1)      — one Z-slice per chunk
#   4D (Y, X, Z, T)     → chunks (Y, X, 1, 2)   — one Z-slice, 2 T-frames
#   5D (Y, X, Z, T, C)  → chunks (Y, X, 1, 1, C) — full channel, one ZT
#
# For very large XY planes (> ~1024 px) you may tile XY as well, e.g.
#   chunks (512, 512, 1) for a (2048, 2048, 300) volume.
# Use ZARR_LARGE_XY_TILE (pixels) as the threshold.
#
# Target uncompressed chunk byte size: 1–8 MB (up to ~16 MB on fast NVMe).
# At float32 that means:
#   256×256  → 0.26 MB per slice — fine, no XY tiling needed
#   512×512  → 1.0 MB per slice — within target
#   1024×1024→ 4.0 MB per slice — within target, upper end
#   2048×2048→ 16 MB per slice  — at limit; consider 1024×1024 XY tiling
#
# Compression default: no compressor ("raw") for local NVMe workloads where
# decompression latency exceeds I/O savings.  Users on remote/compressed FS
# should use blosc with lz4 (fast) or zstd (better ratio).
ZARR_LARGE_XY_TILE = 1024  # pixels — threshold for tiling XY dimension
ZARR_T_DEPTH = 2  # T-frames per chunk for 4D+ arrays

# ---------------------------------------------------------------------------
# Neighbor prefetch (Phase 3)
# ---------------------------------------------------------------------------
# After rendering a frame the backend eagerly warms the raw_cache for the
# next N slices in the scroll direction.  Only runs when the estimated I/O
# cost is within budget.
PREFETCH_NEIGHBORS = 3  # slices to prefetch in the scroll direction
PREFETCH_BUDGET_BYTES = 16 * 1024 * 1024  # 16 MB max prefetch per request

# ---------------------------------------------------------------------------
# Heavy-operation guardrails (Phase 4)
# ---------------------------------------------------------------------------
# Operations that materialize the entire array (FFT) or stack many slices
# (GIF, grid) are blocked above this threshold.  Users may override via the
# ARRAYVIEW_HEAVY_OP_LIMIT_MB environment variable.
_DEFAULT_HEAVY_OP_MB = 500
HEAVY_OP_LIMIT_BYTES = (
    int(os.environ.get("ARRAYVIEW_HEAVY_OP_LIMIT_MB", _DEFAULT_HEAVY_OP_MB))
    * 1024
    * 1024
)


def _estimate_array_bytes(session) -> int:
    """Best-effort estimate of the total uncompressed byte size of session.data.

    Uses session.shape (not data.nbytes) as the authoritative source so that
    guardrails remain correct even when the underlying data object differs from
    session.shape (e.g. in tests that patch session.shape independently).
    """
    try:
        itemsize = np.dtype(session.data.dtype).itemsize
    except Exception:
        itemsize = 4  # assume float32
    return int(np.prod(session.shape)) * itemsize


# ---------------------------------------------------------------------------
# Adaptive cache budgets (Phase 5)
# ---------------------------------------------------------------------------
# Defaults adapt to available system RAM; override via environment variables.
# All values in bytes.


def _total_ram_bytes() -> int:
    """Return total system RAM in bytes, or 8 GB as a conservative fallback."""
    try:
        import psutil

        return psutil.virtual_memory().total
    except Exception:
        return 8 * 1024**3  # 8 GB fallback


def _cache_budget(env_var: str, fraction: float) -> int:
    """Return a cache byte budget.

    Priority:
      1. Environment variable ``env_var`` (in MB, integer).
      2. ``fraction`` * total_ram, rounded down to the nearest MB.
    """
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
    interactive slice navigation in arrayview.

    Rules:
    - 2-D: return shape unchanged (single chunk — data fits in a frame).
    - 3-D (Y, X, Z): one Z-slice per chunk; tile XY if > ZARR_LARGE_XY_TILE.
    - 4-D (Y, X, Z, T): one Z-slice, ZARR_T_DEPTH T-frames; same XY rule.
    - 5-D+: one slice along every non-XY dimension; full-size along C (last).
    - 1-D or unknown rank: return None (let Zarr choose).
    """
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


LUTS = {
    name: np.concatenate(
        [
            (mpl_colormaps[name](np.arange(256) / 255.0) * 255).astype(np.uint8)[:, :3],
            np.full((256, 1), 255, dtype=np.uint8),
        ],
        axis=1,
    )
    for name in COLORMAPS
}


def _lut_to_gradient_stops(lut, n=32):
    indices = np.linspace(0, 255, n, dtype=int)
    return [[int(lut[i, 0]), int(lut[i, 1]), int(lut[i, 2])] for i in indices]


COLORMAP_GRADIENT_STOPS = {
    name: _lut_to_gradient_stops(LUTS[name]) for name in COLORMAPS
}
COMPLEX_MODES = ["mag", "phase", "real", "imag"]
REAL_MODES = ["real", "mag"]
OVERLAY_COLOR = np.array([255, 80, 80], dtype=np.float32)
OVERLAY_ALPHA = np.float32(0.45)

app = FastAPI()


@app.exception_handler(Exception)
async def _generic_exception_handler(request: Request, exc: Exception):
    import traceback

    print(
        f"[ArrayView] Unhandled error on {request.url.path}: {exc}\n"
        + traceback.format_exc(),
        flush=True,
    )
    return JSONResponse(
        status_code=500, content={"error": str(exc), "type": type(exc).__name__}
    )


# ---------------------------------------------------------------------------
# HTML Templates (loaded once at import time from package files)
# ---------------------------------------------------------------------------
_SHELL_HTML: str = (
    _pkg_files(__package__).joinpath("_shell.html").read_text(encoding="utf-8")
)
_VIEWER_HTML_TEMPLATE: str = (
    _pkg_files(__package__).joinpath("_viewer.html").read_text(encoding="utf-8")
)


def load_data(filepath):
    if filepath.endswith(".npy"):
        return np.load(filepath, mmap_mode="r")
    elif filepath.endswith(".npz"):
        npz = np.load(filepath)
        keys = list(npz.keys())
        if len(keys) == 1:
            return npz[keys[0]]
        raise ValueError(
            f".npz contains multiple arrays: {keys}. "
            "Load it manually and pass the array to view()."
        )
    elif filepath.endswith(".nii") or filepath.endswith(".nii.gz"):
        return nib.load(filepath).dataobj
    elif filepath.endswith(".zarr") or filepath.endswith(".zarr.zip"):
        import zarr

        return zarr.open(filepath, mode="r")
    elif filepath.endswith(".pt") or filepath.endswith(".pth"):
        try:
            import torch
        except ImportError:
            raise ImportError("Install torch to load .pt/.pth files.")
        obj = torch.load(filepath, map_location="cpu", weights_only=True)
        return _tensor_to_numpy(obj, filepath)
    elif filepath.endswith(".h5") or filepath.endswith(".hdf5"):
        import h5py

        f = h5py.File(filepath, "r")
        keys = list(f.keys())
        if len(keys) == 1:
            return f[keys[0]][()]
        raise ValueError(
            f".h5 file contains multiple datasets: {keys}. "
            "Load it manually and pass the array to view()."
        )
    elif filepath.endswith(".tif") or filepath.endswith(".tiff"):
        import tifffile

        return tifffile.imread(filepath)
    elif filepath.endswith(".mat"):
        import scipy.io

        mat = scipy.io.loadmat(filepath)
        arrays = {
            k: v
            for k, v in mat.items()
            if not k.startswith("_") and isinstance(v, np.ndarray)
        }
        if len(arrays) == 1:
            return next(iter(arrays.values()))
        raise ValueError(
            f".mat file contains multiple arrays: {list(arrays.keys())}. "
            "Load it manually and pass the array to view()."
        )
    else:
        raise ValueError(
            "Unsupported format. Supported: .npy, .npz, .nii/.nii.gz, .zarr, "
            ".pt/.pth, .h5/.hdf5, .tif/.tiff, .mat"
        )


def _tensor_to_numpy(obj, source="object"):
    """Convert a tensor-like object (PyTorch, etc.) to a numpy array."""
    if isinstance(obj, np.ndarray):
        return obj
    if hasattr(obj, "detach"):
        obj = obj.detach()
    if hasattr(obj, "cpu"):
        obj = obj.cpu()
    if hasattr(obj, "numpy"):
        return obj.numpy()
    if hasattr(obj, "__array__"):
        return np.array(
            obj
        )  # copy — ensures external GC (e.g. Julia) can't free the backing memory
    raise ValueError(f"Cannot convert {type(obj)} from {source} to a numpy array.")


def mosaic_shape(batch):
    mshape = [int(batch**0.5), batch // int(batch**0.5)]
    while mshape[0] * mshape[1] < batch:
        mshape[1] += 1
    if (mshape[0] - 1) * (mshape[1] + 1) == batch:
        mshape[0] -= 1
        mshape[1] += 1
    return tuple(mshape)


def _compute_vmin_vmax(session, data, dr, complex_mode=0):
    if complex_mode == 1 and np.iscomplexobj(session.data):
        return (-float(np.pi), float(np.pi))
    # Global stats are only meaningful for ≤3-D arrays where all data shares
    # the same scale.  For 4-D+ arrays the extra dims typically represent
    # channels with very different value ranges, so global stats (computed
    # across all channels) give a misleading scale for individual channels.
    if complex_mode == 0 and len(session.shape) <= 3 and dr in session.global_stats:
        vmin, vmax = session.global_stats[dr]
        if vmin != vmax:
            return vmin, vmax
    pct_lo, pct_hi = DR_PERCENTILES[dr % len(DR_PERCENTILES)]
    return float(np.percentile(data, pct_lo)), float(np.percentile(data, pct_hi))


def extract_slice(session, dim_x, dim_y, idx_list):
    key = (dim_x, dim_y, tuple(idx_list))
    if key in session.raw_cache:
        session.raw_cache.move_to_end(key)
        return session.raw_cache[key]

    slicer = [
        slice(None) if i in (dim_x, dim_y) else idx_list[i]
        for i in range(len(session.shape))
    ]
    extracted = np.array(session.data[tuple(slicer)])
    if dim_x < dim_y:
        extracted = extracted.T
    if np.iscomplexobj(extracted):
        result = np.nan_to_num(extracted).astype(np.complex64)
    else:
        result = np.nan_to_num(extracted).astype(np.float32)

    session.raw_cache[key] = result
    session._raw_bytes += result.nbytes
    while session._raw_bytes > session.RAW_CACHE_BYTES and session.raw_cache:
        _, v = session.raw_cache.popitem(last=False)
        session._raw_bytes -= v.nbytes
    return result


def apply_complex_mode(raw, complex_mode):
    if np.iscomplexobj(raw):
        if complex_mode == 1:
            result = np.angle(raw)
        elif complex_mode == 2:
            result = raw.real.copy()
        elif complex_mode == 3:
            result = raw.imag.copy()
        else:
            result = np.abs(raw)
    else:
        result = np.abs(raw) if complex_mode == 1 else raw
    return np.nan_to_num(result).astype(np.float32)


def _compute_otsu_threshold(data) -> float:
    """Compute Otsu's threshold on the absolute values of non-zero, finite elements."""
    flat = np.abs(np.asarray(data, dtype=np.float64).ravel())
    flat = flat[np.isfinite(flat) & (flat > 0)]
    if len(flat) < 10:
        return 0.0
    if len(flat) > 1_000_000:
        rng = np.random.default_rng(42)
        flat = rng.choice(flat, 1_000_000, replace=False)
    hist, edges = np.histogram(flat, bins=256)
    centers = (edges[:-1] + edges[1:]) / 2
    total = float(hist.sum())
    if total == 0:
        return 0.0
    w_b = np.cumsum(hist) / total
    w_f = 1.0 - w_b
    mu_cum = np.cumsum(hist * centers)
    mu_b = np.where(w_b > 0, mu_cum / np.maximum(w_b * total, 1e-10), 0.0)
    mu_f = np.where(
        w_f > 0, (mu_cum[-1] / total - mu_b * w_b) / np.maximum(w_f, 1e-10), 0.0
    )
    sigma_b_sq = w_b * w_f * (mu_b - mu_f) ** 2
    return float(centers[int(np.argmax(sigma_b_sq))])


def _prepare_display(
    session, raw, complex_mode, dr, log_scale, vmin_override=None, vmax_override=None
):
    data = apply_complex_mode(raw, complex_mode)
    if vmin_override is not None and vmax_override is not None:
        if log_scale:
            data = np.log1p(np.abs(data)).astype(np.float32)
        return data, vmin_override, vmax_override
    if log_scale:
        data = np.log1p(np.abs(data)).astype(np.float32)
        # Mirror the global-stats path used in the non-log case so that the
        # visual scale is consistent across slices (same as without log_scale).
        if complex_mode == 0 and len(session.shape) <= 3 and dr in session.global_stats:
            raw_vmin, raw_vmax = session.global_stats[dr]
            vmin = float(np.log1p(abs(raw_vmin)))
            vmax = float(np.log1p(abs(raw_vmax)))
            if vmin == vmax:  # degenerate: fall back to per-slice percentile
                pct_lo, pct_hi = DR_PERCENTILES[dr % len(DR_PERCENTILES)]
                vmin = float(np.percentile(data, pct_lo))
                vmax = float(np.percentile(data, pct_hi))
        else:
            pct_lo, pct_hi = DR_PERCENTILES[dr % len(DR_PERCENTILES)]
            vmin = float(np.percentile(data, pct_lo))
            vmax = float(np.percentile(data, pct_hi))
    else:
        vmin, vmax = _compute_vmin_vmax(session, data, dr, complex_mode)
    return data, vmin, vmax


def _ensure_lut(name: str) -> bool:
    """Ensure name is in LUTS. Returns True if valid."""
    if name in LUTS:
        return True
    try:
        cmap = mpl_colormaps[name]
    except KeyError:
        return False
    rgba = (cmap(np.arange(256) / 255.0) * 255).astype(np.uint8)
    lut = np.concatenate([rgba[:, :3], np.full((256, 1), 255, dtype=np.uint8)], axis=1)
    LUTS[name] = lut
    COLORMAP_GRADIENT_STOPS[name] = _lut_to_gradient_stops(lut)
    return True


def apply_colormap_rgba(
    session,
    raw,
    colormap,
    dr,
    complex_mode=0,
    log_scale=False,
    vmin_override=None,
    vmax_override=None,
):
    data, vmin, vmax = _prepare_display(
        session,
        raw,
        complex_mode,
        dr,
        log_scale,
        vmin_override=vmin_override,
        vmax_override=vmax_override,
    )
    if vmax > vmin:
        normalized = np.clip((data - vmin) / (vmax - vmin), 0, 1)
    else:
        normalized = np.zeros_like(data)
    _ensure_lut(colormap)
    lut = LUTS.get(colormap, LUTS["gray"])
    rgba = lut[(normalized * 255).astype(np.uint8)]
    # Values at or below cmin → transparent so the canvas background shows through.
    # If a mask threshold is set, use that instead of the display minimum.
    mask_thr = getattr(session, "mask_threshold", 0.0)
    if mask_thr > 0:
        abs_raw = np.abs(raw)
        transparent = abs_raw < mask_thr
    elif vmin > 0 and vmax > vmin:
        # User has a explicit positive cmin: hide everything below it.
        transparent = data < vmin
    else:
        # vmin == 0 or flat array: keep legacy behaviour — make exact zeros
        # transparent so sparse arrays / masks show through to the background.
        transparent = data == np.float32(0)
    if transparent.any():
        rgba = rgba.copy()
        rgba[transparent, 3] = 0
    return rgba


def _detect_rgb_axis(shape: tuple) -> int:
    """Return the axis index for the RGB/RGBA channel dimension.

    Checks if the first or last dimension has size 3 or 4.
    Prefers the last dimension if both qualify (channels-last is most common).
    Raises ValueError if no suitable dimension is found.
    """
    if len(shape) < 2:
        raise ValueError(
            f"Array must have at least 2 dimensions for RGB mode, got shape {shape}"
        )
    last = shape[-1]
    first = shape[0]
    if last in (3, 4):
        return len(shape) - 1
    if first in (3, 4):
        return 0
    raise ValueError(
        f"RGB mode requires the first or last dimension to have size 3 or 4 "
        f"(for RGB or RGBA). Got shape {shape} — first={first}, last={last}."
    )


def _setup_rgb(session) -> None:
    """Detect and store the RGB channel axis on a session.

    Sets session.rgb_axis (int) and session.spatial_shape (tuple without rgb_axis).
    Raises ValueError if the array shape is not compatible with RGB mode.
    """
    axis = _detect_rgb_axis(session.shape)
    session.rgb_axis = axis
    session.spatial_shape = tuple(s for i, s in enumerate(session.shape) if i != axis)


def render_rgb_rgba(session, dim_x: int, dim_y: int, idx_list: list) -> np.ndarray:
    """Render an RGB/RGBA session slice to a H×W×4 uint8 RGBA array.

    dim_x, dim_y, and idx_list are in *spatial* coordinates — i.e. they index
    into session.spatial_shape, with the rgb_axis dimension excluded.
    Returns an (H, W, 4) uint8 numpy array ready for the wire protocol.
    """
    rgb_axis = session.rgb_axis
    ndim_actual = len(session.shape)

    # Cache lookup — key uses spatial coordinates only (no colormap/dr).
    cache_key = ("rgb", dim_x, dim_y, tuple(idx_list))
    if cache_key in session.rgba_cache:
        session.rgba_cache.move_to_end(cache_key)
        return session.rgba_cache[cache_key]

    # Map spatial dim indices to actual array dim indices.
    if rgb_axis == 0:
        actual_dim_x = dim_x + 1
        actual_dim_y = dim_y + 1
    else:
        # rgb_axis is last; spatial dims are the same as actual dims before it.
        actual_dim_x = dim_x
        actual_dim_y = dim_y

    # Build slicer over the actual array dimensions.
    # For each actual dim: rgb_axis → slice(None), dim_x/dim_y → slice(None),
    # everything else → the corresponding spatial index.
    slicer: list = []
    spatial_i = 0
    for actual_i in range(ndim_actual):
        if actual_i == rgb_axis:
            slicer.append(slice(None))  # keep all channels; NOT a spatial dim
        else:
            if actual_i in (actual_dim_x, actual_dim_y):
                slicer.append(slice(None))  # keep this spatial display dim
            else:
                slicer.append(int(idx_list[spatial_i]))  # fix this spatial dim
            spatial_i += 1

    arr = np.array(session.data[tuple(slicer)])  # (free_dim_0, free_dim_1, free_dim_2)

    # The three free dims in the result appear in the order of their actual indices.
    free_actual = sorted([actual_dim_x, actual_dim_y, rgb_axis])
    pos_dx = free_actual.index(actual_dim_x)
    pos_dy = free_actual.index(actual_dim_y)
    pos_rgb = free_actual.index(rgb_axis)

    # Rearrange to (rows=dy, cols=dx, channels).
    arr = arr.transpose(pos_dy, pos_dx, pos_rgb)  # (H, W, C)
    arr = np.nan_to_num(arr)

    # Convert to uint8.
    if arr.dtype.kind == "f":
        if arr.max() > 1.5:  # assume [0, 255] float range
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        else:  # assume [0, 1] float range
            arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    h, w, c = arr.shape
    rgba = np.empty((h, w, 4), dtype=np.uint8)
    rgba[:, :, :3] = arr[:, :, :3]
    rgba[:, :, 3] = arr[:, :, 3] if c == 4 else np.uint8(255)

    # Cache result.
    session.rgba_cache[cache_key] = rgba
    session._rgba_bytes += rgba.nbytes
    while session._rgba_bytes > session.RGBA_CACHE_BYTES and session.rgba_cache:
        _, v = session.rgba_cache.popitem(last=False)
        session._rgba_bytes -= v.nbytes

    return rgba


def render_rgba(
    session,
    dim_x,
    dim_y,
    idx_tuple,
    colormap,
    dr,
    complex_mode=0,
    log_scale=False,
    vmin_override=None,
    vmax_override=None,
):
    has_override = vmin_override is not None and vmax_override is not None
    if not has_override:
        key = (
            dim_x,
            dim_y,
            idx_tuple,
            colormap,
            dr,
            complex_mode,
            log_scale,
            getattr(session, "mask_threshold", 0.0),
        )
        if key in session.rgba_cache:
            session.rgba_cache.move_to_end(key)
            return session.rgba_cache[key]
    raw = extract_slice(session, dim_x, dim_y, list(idx_tuple))
    rgba = apply_colormap_rgba(
        session,
        raw,
        colormap,
        dr,
        complex_mode,
        log_scale,
        vmin_override=vmin_override,
        vmax_override=vmax_override,
    )
    if not has_override:
        session.rgba_cache[key] = rgba
        session._rgba_bytes += rgba.nbytes
        while session._rgba_bytes > session.RGBA_CACHE_BYTES and session.rgba_cache:
            _, v = session.rgba_cache.popitem(last=False)
            session._rgba_bytes -= v.nbytes
    return rgba


def _extract_overlay_mask(
    overlay_sid: str | None,
    dim_x: int,
    dim_y: int,
    idx_tuple: tuple[int, ...],
    expected_shape: tuple[int, int],
) -> np.ndarray | None:
    """Return a boolean mask slice for overlay compositing, or None when unavailable."""
    if not overlay_sid:
        return None
    ov_session = SESSIONS.get(str(overlay_sid))
    if ov_session is None:
        return None

    ov_ndim = ov_session.data.ndim
    if dim_x >= ov_ndim or dim_y >= ov_ndim:
        return None

    # Prefer leading-dim alignment, but also try trailing alignment to support
    # data/mask pairs that differ by extra leading dimensions in the base data.
    idx_candidates: list[tuple[int, ...]] = [
        tuple(int(idx_tuple[i]) if i < len(idx_tuple) else 0 for i in range(ov_ndim))
    ]
    if len(idx_tuple) >= ov_ndim:
        trailing = tuple(int(v) for v in idx_tuple[-ov_ndim:])
        if trailing != idx_candidates[0]:
            idx_candidates.append(trailing)

    for ov_idx in idx_candidates:
        try:
            ov_raw = extract_slice(ov_session, dim_x, dim_y, list(ov_idx))
        except Exception:
            continue
        if ov_raw.shape != expected_shape:
            continue
        mask = np.isfinite(ov_raw) & (ov_raw > 0.5)
        if mask.any():
            return mask
    return None


def _composite_overlay_mask(
    rgba: np.ndarray, mask: np.ndarray | None, alpha: float = float(OVERLAY_ALPHA)
) -> np.ndarray:
    """Alpha-composite a red segmentation mask on top of an RGBA frame."""
    if mask is None:
        return rgba

    out = rgba.copy()
    base_rgb = out[mask, :3].astype(np.float32) / 255.0
    base_a = out[mask, 3].astype(np.float32) / 255.0

    ov_a = np.float32(alpha)
    out_a = ov_a + base_a * (1.0 - ov_a)
    denom = np.maximum(out_a, 1e-6)

    ov_rgb = OVERLAY_COLOR / 255.0
    out_rgb = (ov_a * ov_rgb + (1.0 - ov_a) * base_a[:, None] * base_rgb) / denom[
        :, None
    ]

    out[mask, :3] = np.clip(out_rgb * 255.0, 0.0, 255.0).astype(np.uint8)
    out[mask, 3] = np.clip(out_a * 255.0, 0.0, 255.0).astype(np.uint8)
    return out


def render_mosaic(
    session,
    dim_x,
    dim_y,
    dim_z,
    idx_tuple,
    colormap,
    dr,
    complex_mode=0,
    log_scale=False,
):
    idx_norm = list(idx_tuple)
    idx_norm[dim_z] = 0
    key = (dim_x, dim_y, dim_z, tuple(idx_norm), colormap, dr, complex_mode, log_scale)
    if key in session.mosaic_cache:
        session.mosaic_cache.move_to_end(key)
        return session.mosaic_cache[key]

    n = session.shape[dim_z]
    frames_raw = [
        extract_slice(
            session,
            dim_x,
            dim_y,
            [i if j == dim_z else idx_tuple[j] for j in range(len(session.shape))],
        )
        for i in range(n)
    ]
    frames = [apply_complex_mode(f, complex_mode) for f in frames_raw]
    if log_scale:
        frames = [np.log1p(np.abs(f)).astype(np.float32) for f in frames]
    all_data = np.stack(frames)

    if log_scale:
        pct_lo, pct_hi = DR_PERCENTILES[dr % len(DR_PERCENTILES)]
        vmin = float(np.percentile(all_data, pct_lo))
        vmax = float(np.percentile(all_data, pct_hi))
    else:
        vmin, vmax = _compute_vmin_vmax(session, all_data, dr, complex_mode)

    rows, cols = mosaic_shape(n)
    H, W = frames[0].shape
    GAP = 2  # separator pixels between tiles
    total_h = rows * H + (rows - 1) * GAP
    total_w = cols * W + (cols - 1) * GAP
    grid = np.full((total_h, total_w), np.nan, dtype=np.float32)
    for k in range(n):
        r, c = divmod(k, cols)
        r0, c0 = r * (H + GAP), c * (W + GAP)
        grid[r0 : r0 + H, c0 : c0 + W] = all_data[k]

    nan_mask = np.isnan(grid)
    filled = np.where(nan_mask, vmin, grid)
    if vmax > vmin:
        normalized = np.clip((filled - vmin) / (vmax - vmin), 0, 1)
    else:
        normalized = np.zeros_like(filled)

    _ensure_lut(colormap)
    lut = LUTS.get(colormap, LUTS["gray"])
    rgba = lut[(normalized * 255).astype(np.uint8)]
    rgba[nan_mask] = [22, 22, 22, 255]  # dark separator
    session.mosaic_cache[key] = rgba
    session._mosaic_bytes += rgba.nbytes
    while session._mosaic_bytes > session.MOSAIC_CACHE_BYTES and session.mosaic_cache:
        _, v = session.mosaic_cache.popitem(last=False)
        session._mosaic_bytes -= v.nbytes
    return rgba


def _run_preload(
    session,
    gen,
    dim_x,
    dim_y,
    idx_list,
    colormap,
    dr,
    slice_dim,
    dim_z=-1,
    complex_mode=0,
    log_scale=False,
):
    # In RGB mode, the frontend sends indices for spatial_shape (rgb axis excluded).
    # Use spatial_shape for dimension size lookups to stay in sync.
    shape = session.spatial_shape if session.rgb_axis is not None else session.shape
    n = shape[slice_dim]
    H = shape[dim_y]
    W = shape[dim_x]
    if dim_z >= 0:
        nz = shape[dim_z]
        mrows, mcols = mosaic_shape(nz)
        size_bytes = n * (mrows * H) * (mcols * W) * 4
    else:
        size_bytes = n * H * W * 4

    with session.preload_lock:
        session.preload_total = n
        session.preload_done = 0
        if size_bytes > 500 * 1024 * 1024:
            session.preload_skipped = True
            return
        session.preload_skipped = False

    for i in range(n):
        if session.preload_gen != gen:
            return
        idx = list(idx_list)
        idx[slice_dim] = i
        if dim_z >= 0:
            render_mosaic(
                session,
                dim_x,
                dim_y,
                dim_z,
                tuple(idx),
                colormap,
                dr,
                complex_mode,
                log_scale,
            )
        else:
            if session.rgb_axis is not None:
                render_rgb_rgba(session, dim_x, dim_y, list(idx))
            else:
                render_rgba(
                    session,
                    dim_x,
                    dim_y,
                    tuple(idx),
                    colormap,
                    dr,
                    complex_mode,
                    log_scale,
                )
        with session.preload_lock:
            session.preload_done = i + 1
        time.sleep(0.005)


# ---------------------------------------------------------------------------
# Shell WebSocket for Webview Tab Management
# ---------------------------------------------------------------------------
async def _notify_shells(sid, name, url=None, wait: bool = True) -> bool:
    """Push a new-tab message to all connected webview shell windows.

    Returns True if at least one shell received the message.
    wait=True: poll up to 2 s for a shell to connect (used when a window was just opened).
    wait=False: send immediately to whatever shells are currently connected.
    """
    if wait:
        for _ in range(200):  # Wait up to 2 s for window to connect
            if SHELL_SOCKETS:
                break
            await asyncio.sleep(0.01)
    msg = {"action": "new_tab", "sid": sid, "name": name}
    if url:
        msg["url"] = url
    notified = False
    for ws in SHELL_SOCKETS.copy():
        try:
            await ws.send_json(msg)
            notified = True
        except Exception:
            pass
    return notified


@app.websocket("/ws/shell")
async def shell_websocket(ws: WebSocket):
    await ws.accept()
    SHELL_SOCKETS.append(ws)
    try:
        while True:
            msg = await ws.receive_json()
            if msg.get("action") == "close":
                sid = msg.get("sid")
                if sid in SESSIONS:
                    SESSIONS[sid].raw_cache.clear()
                    SESSIONS[sid].rgba_cache.clear()
                    SESSIONS[sid].mosaic_cache.clear()
                    SESSIONS[sid]._raw_bytes = SESSIONS[sid]._rgba_bytes = SESSIONS[
                        sid
                    ]._mosaic_bytes = 0
                    SESSIONS[sid].data = None
                    del SESSIONS[sid]
    except Exception:
        pass
    finally:
        if ws in SHELL_SOCKETS:
            SHELL_SOCKETS.remove(ws)


@app.websocket("/ws/{sid}")
async def websocket_endpoint(ws: WebSocket, sid: str):
    session = SESSIONS.get(sid)
    if not session:
        await ws.close()
        return

    await ws.accept()
    global VIEWER_SOCKETS
    VIEWER_SOCKETS += 1
    loop = asyncio.get_running_loop()

    try:
        while True:
            msg = await ws.receive_json()
            seq = int(msg.get("seq", 0))

            dim_x = int(msg["dim_x"])
            dim_y = int(msg["dim_y"])
            idx_tuple = tuple(int(x) for x in msg["indices"])
            colormap = str(msg.get("colormap", "gray"))
            dr = int(msg.get("dr", 1))
            dim_z = int(msg.get("dim_z", -1))
            complex_mode = int(msg.get("complex_mode", 0))
            log_scale = bool(msg.get("log_scale", False))
            _vmin_ov = msg.get("vmin_override")
            _vmax_ov = msg.get("vmax_override")
            vmin_override = float(_vmin_ov) if _vmin_ov is not None else None
            vmax_override = float(_vmax_ov) if _vmax_ov is not None else None
            direction = int(msg.get("direction", 1))  # scroll direction (+1/-1)
            slice_dim = int(msg.get("slice_dim", -1))

            if session.rgb_axis is not None:
                # RGB/RGBA mode — render directly from channel data; skip colormap.
                rgba = await _render(
                    loop,
                    lambda: render_rgb_rgba(session, dim_x, dim_y, list(idx_tuple)),
                )
                h, w = rgba.shape[:2]
                vmin, vmax = 0.0, 255.0
            elif dim_z >= 0:
                rgba = await _render(
                    loop,
                    lambda: render_mosaic(
                        session,
                        dim_x,
                        dim_y,
                        dim_z,
                        idx_tuple,
                        colormap,
                        dr,
                        complex_mode,
                        log_scale,
                    ),
                )
                h, w = rgba.shape[:2]
                raw = extract_slice(session, dim_x, dim_y, list(idx_tuple))
                _, vmin, vmax = _prepare_display(
                    session,
                    raw,
                    complex_mode,
                    dr,
                    log_scale,
                    vmin_override=vmin_override,
                    vmax_override=vmax_override,
                )
            else:
                rgba = await _render(
                    loop,
                    lambda: render_rgba(
                        session,
                        dim_x,
                        dim_y,
                        idx_tuple,
                        colormap,
                        dr,
                        complex_mode,
                        log_scale,
                        vmin_override,
                        vmax_override,
                    ),
                )
                h, w = rgba.shape[:2]
                raw = extract_slice(session, dim_x, dim_y, list(idx_tuple))
                _, vmin, vmax = _prepare_display(
                    session,
                    raw,
                    complex_mode,
                    dr,
                    log_scale,
                    vmin_override=vmin_override,
                    vmax_override=vmax_override,
                )

                # Overlay compositing (segmentation masks)
                overlay_sid = msg.get("overlay_sid")
                mask = _extract_overlay_mask(
                    overlay_sid, dim_x, dim_y, idx_tuple, expected_shape=(h, w)
                )
                rgba = _composite_overlay_mask(rgba, mask)

            header = np.array([seq, w, h], dtype=np.uint32).tobytes()
            vminmax = np.array([vmin, vmax], dtype=np.float32).tobytes()
            await ws.send_bytes(header + vminmax + rgba.tobytes())

            # Warm neighbor slices in the background (Phase 3 prefetch)
            if slice_dim >= 0 and not (dim_z >= 0):
                _schedule_prefetch(
                    session, dim_x, dim_y, list(idx_tuple), slice_dim, direction
                )
    except WebSocketDisconnect:
        pass  # normal: browser closed the tab/window
    except Exception as _ws_exc:
        import traceback

        print(f"[ArrayView] WS/{sid[:8]}: {_ws_exc}", flush=True)
        traceback.print_exc()
    finally:
        VIEWER_SOCKETS = max(0, VIEWER_SOCKETS - 1)


@app.get("/clearcache/{sid}")
def clear_cache(sid: str):
    session = SESSIONS.get(sid)
    if session:
        session.raw_cache.clear()
        session.rgba_cache.clear()
        session.mosaic_cache.clear()
        session._raw_bytes = session._rgba_bytes = session._mosaic_bytes = 0
    return {"status": "ok"}


@app.get("/cache_info/{sid}")
def cache_info(sid: str):
    """Phase 5: debug endpoint — returns per-session cache usage and budgets."""
    session = SESSIONS.get(sid)
    if not session:
        return Response(status_code=404)
    return {
        "raw_cache": {
            "entries": len(session.raw_cache),
            "used_bytes": session._raw_bytes,
            "budget_bytes": session.RAW_CACHE_BYTES,
            "used_mb": round(session._raw_bytes / 1e6, 2),
            "budget_mb": round(session.RAW_CACHE_BYTES / 1e6, 2),
        },
        "rgba_cache": {
            "entries": len(session.rgba_cache),
            "used_bytes": session._rgba_bytes,
            "budget_bytes": session.RGBA_CACHE_BYTES,
            "used_mb": round(session._rgba_bytes / 1e6, 2),
            "budget_mb": round(session.RGBA_CACHE_BYTES / 1e6, 2),
        },
        "mosaic_cache": {
            "entries": len(session.mosaic_cache),
            "used_bytes": session._mosaic_bytes,
            "budget_bytes": session.MOSAIC_CACHE_BYTES,
            "used_mb": round(session._mosaic_bytes / 1e6, 2),
            "budget_mb": round(session.MOSAIC_CACHE_BYTES / 1e6, 2),
        },
        "heavy_op_limit_mb": round(HEAVY_OP_LIMIT_BYTES / 1e6, 1),
    }


@app.get("/colormap/{name}")
def get_colormap(name: str):
    """Validate a matplotlib colormap name and return its gradient stops."""
    if not _ensure_lut(name):
        return Response(status_code=404)
    return {"ok": True, "gradient_stops": COLORMAP_GRADIENT_STOPS[name]}


# Multipliers applied to the Otsu threshold for each mask level (level 0 = off)
MASK_MULTIPLIERS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0]


@app.post("/mask/{sid}")
async def set_mask(sid: str, request: Request):
    """Cycle mask level (0=off, 1–7 = increasing Otsu multiplier).

    Client sends ``free_dims`` (list of dim indices to keep free) and
    ``indices`` (current full index tuple). Otsu is computed on the
    sub-volume formed by those free dims.
    """
    session = SESSIONS.get(sid)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    body = await request.json()
    level = int(body.get("level", 0))
    level = max(0, min(level, len(MASK_MULTIPLIERS) - 1))
    if level == 0:
        session.mask_level = 0
        session.mask_threshold = 0.0
        session.rgba_cache.clear()
        session._rgba_bytes = 0
        return {"level": 0, "threshold": 0.0, "multiplier": 0.0}
    try:
        free_dims = set(int(d) for d in body.get("free_dims", [0, 1]))
        indices = [int(v) for v in str(body.get("indices", "0")).split(",")]
        loop = asyncio.get_running_loop()

        def _extract_subvol():
            slicer = tuple(
                slice(None) if i in free_dims else indices[i]
                for i in range(len(session.shape))
            )
            return session.data[slicer]

        subvol = await loop.run_in_executor(None, _extract_subvol)
        otsu = await loop.run_in_executor(None, lambda: _compute_otsu_threshold(subvol))
    except Exception as e:
        return {"error": str(e)}
    otsu = float(otsu)
    multiplier = MASK_MULTIPLIERS[level]
    threshold = otsu * multiplier
    session.mask_level = level
    session.mask_threshold = threshold
    session.mask_otsu = otsu
    session.rgba_cache.clear()
    session._rgba_bytes = 0
    return {
        "level": level,
        "threshold": threshold,
        "otsu": otsu,
        "multiplier": multiplier,
    }


@app.post("/preload/{sid}")
async def start_preload(sid: str, request: Request):
    session = SESSIONS.get(sid)
    if not session:
        return {"error": "Invalid session"}

    body = await request.json()
    dim_x = int(body["dim_x"])
    dim_y = int(body["dim_y"])
    idx_list = [int(x) for x in body["indices"]]
    colormap = str(body.get("colormap", "gray"))
    dr = int(body.get("dr", 1))
    slice_dim = int(body["slice_dim"])
    dim_z = int(body.get("dim_z", -1))
    complex_mode = int(body.get("complex_mode", 0))
    log_scale = bool(body.get("log_scale", False))

    session.preload_gen += 1
    gen = session.preload_gen
    threading.Thread(
        target=_run_preload,
        args=(
            session,
            gen,
            dim_x,
            dim_y,
            idx_list,
            colormap,
            dr,
            slice_dim,
            dim_z,
            complex_mode,
            log_scale,
        ),
        daemon=True,
    ).start()
    return {"status": "started"}


@app.get("/preload_status/{sid}")
def get_preload_status(sid: str):
    session = SESSIONS.get(sid)
    if not session:
        return {"error": "Invalid session"}
    with session.preload_lock:
        return {
            "done": session.preload_done,
            "total": session.preload_total,
            "skipped": session.preload_skipped,
        }


def _vfield_n_times(session) -> int:
    """Return number of time frames in the vector field (0 = no vfield, 1 = no time dim)."""
    if session.vfield is None:
        return 0
    vf_ndim = np.asarray(session.vfield).ndim
    img_ndim = np.asarray(session.data).ndim
    return int(np.asarray(session.vfield).shape[0]) if vf_ndim == img_ndim + 2 else 1


@app.get("/metadata/{sid}")
async def get_metadata(sid: str):
    session = SESSIONS.get(sid)
    if not session and sid in PENDING_SESSIONS:
        # Session is still loading in a background thread — poll until ready (max 120 s).
        for _ in range(1200):
            await asyncio.sleep(0.1)
            session = SESSIONS.get(sid)
            if session:
                break
    if not session:
        return Response(status_code=404)
    try:
        return {
            "shape": [
                int(s)
                for s in (
                    session.spatial_shape
                    if session.rgb_axis is not None
                    else session.shape
                )
            ],
            "is_complex": bool(np.iscomplexobj(session.data)),
            "name": session.name,
            "has_vectorfield": session.vfield is not None,
            "vfield_n_times": _vfield_n_times(session),
            "is_rgb": session.rgb_axis is not None,
        }
    except Exception as e:
        import traceback

        traceback.print_exc()
        return Response(
            status_code=500, content=str(e).encode(), media_type="text/plain"
        )


@app.get("/vectorfield/{sid}")
def get_vectorfield(sid: str, dim_x: int, dim_y: int, indices: str, t_index: int = 0):
    """Return downsampled deformation vector field arrows for the current 2-D view."""
    session = SESSIONS.get(sid)
    if not session or session.vfield is None:
        return Response(status_code=404)
    try:
        vf = np.array(session.vfield, dtype=np.float32)
        idx_tuple = tuple(int(x) for x in indices.split(","))

        # Strip time dimension if present (shape T, *spatial, 3)
        n_times = _vfield_n_times(session)
        if n_times > 1:
            t = max(0, min(n_times - 1, t_index))
            vf = vf[t]

        ndim_spatial = vf.ndim - 1  # last dim is vector components (size 3)

        # Build index: fix non-display dims, leave dim_x / dim_y free
        slices = []
        for d in range(ndim_spatial):
            slices.append(slice(None) if d in (dim_x, dim_y) else int(idx_tuple[d]))
        slices.append(slice(None))  # vector components
        vf_slice = vf[
            tuple(slices)
        ]  # shape ≈ (A, B, 3) where A,B are free spatial dims

        # Ensure axis order is (dim_y rows, dim_x cols, 3).
        # The free axes appear in ascending original-dim order; transpose if dim_x < dim_y.
        if dim_x < dim_y:
            vf_slice = vf_slice.transpose(1, 0, 2)

        H, W = vf_slice.shape[:2]

        # Component mapping: component index == original spatial dim index
        vy_comp = vf_slice[:, :, dim_y]  # displacement along dim_y → vertical arrows
        vx_comp = vf_slice[:, :, dim_x]  # displacement along dim_x → horizontal arrows

        # Uniform random sampling with a fixed seed derived from (H, W) so that
        # arrow positions are stable across slices (scrolling doesn't rearrange arrows).
        stride = max(1, max(H, W) // 32)
        n_arrows = max(1, (H // stride) * (W // stride))
        rng = np.random.default_rng(int(H) * 10007 + int(W))
        gy = rng.integers(0, H, n_arrows).astype(int)
        gx = rng.integers(0, W, n_arrows).astype(int)

        vx_s = vx_comp[gy, gx]
        vy_s = vy_comp[gy, gx]

        # Scale: stride * 0.75 / p95_magnitude  (image pixels per voxel unit)
        mags = np.sqrt(vx_s**2 + vy_s**2)
        nonzero = mags[mags > 0]
        p95 = float(np.percentile(nonzero, 95)) if nonzero.size else 1.0
        scale = float(stride * 0.75 / max(p95, 1e-9))

        arrows = [
            [int(gx[i]), int(gy[i]), float(vx_s[i]), float(vy_s[i])]
            for i in range(len(gx))
        ]
        return {"arrows": arrows, "scale": scale, "stride": int(stride)}
    except Exception as e:
        import traceback

        traceback.print_exc()
        return Response(
            status_code=500, content=str(e).encode(), media_type="text/plain"
        )


@app.post("/attach_vectorfield")
async def attach_vectorfield(request: Request):
    """Attach a vector field to an existing session (for the existing-server code path)."""
    body = await request.json()
    sid = str(body["sid"])
    filepath = str(body["filepath"])
    session = SESSIONS.get(sid)
    if not session:
        return {"error": f"session {sid} not found"}
    try:
        vf_data = load_data(filepath)
        session.vfield = vf_data
        return {"ok": True}
    except Exception as e:
        return {"error": str(e)}


def _safe_float(v) -> float | None:
    """Convert to float; return None for NaN/Inf (JSON-safe)."""
    f = float(v)
    return f if math.isfinite(f) else None


@app.get("/pixel/{sid}")
def get_pixel(
    sid: str,
    dim_x: int,
    dim_y: int,
    indices: str,
    px: int,
    py: int,
    complex_mode: int = 0,
):
    session = SESSIONS.get(sid)
    if not session:
        return Response(status_code=404)
    if session.rgb_axis is not None:
        return {"value": None}

    idx_tuple = tuple(int(x) for x in indices.split(","))
    raw = extract_slice(session, dim_x, dim_y, list(idx_tuple))
    data = apply_complex_mode(raw, complex_mode)
    h, w = data.shape
    val = _safe_float(data[py, px]) if (0 <= py < h and 0 <= px < w) else None
    return {"value": val}


@app.get("/roi_circle/{sid}")
def get_roi_circle(
    sid: str,
    dim_x: int,
    dim_y: int,
    indices: str,
    cx: float,
    cy: float,
    r: float,
    complex_mode: int = 0,
):
    session = SESSIONS.get(sid)
    if not session:
        return Response(status_code=404)
    if session.rgb_axis is not None:
        return {"error": "not supported for RGB sessions"}
    idx_tuple = tuple(int(v) for v in indices.split(","))
    raw = extract_slice(session, dim_x, dim_y, list(idx_tuple))
    data = apply_complex_mode(raw, complex_mode)
    h, w = data.shape
    ys, xs = np.ogrid[:h, :w]
    mask = (xs - cx) ** 2 + (ys - cy) ** 2 <= r**2
    roi = data[mask]
    if roi.size == 0:
        return {"error": "empty selection"}
    finite = roi[np.isfinite(roi)]
    return {
        "min": _safe_float(finite.min()) if finite.size else None,
        "max": _safe_float(finite.max()) if finite.size else None,
        "mean": _safe_float(finite.mean()) if finite.size else None,
        "std": _safe_float(finite.std()) if finite.size else None,
        "n": int(finite.size),
    }


@app.get("/roi/{sid}")
def get_roi(
    sid: str,
    dim_x: int,
    dim_y: int,
    indices: str,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    complex_mode: int = 0,
):
    session = SESSIONS.get(sid)
    if not session:
        return Response(status_code=404)
    if session.rgb_axis is not None:
        return {"error": "not supported for RGB sessions"}
    idx_tuple = tuple(int(v) for v in indices.split(","))
    raw = extract_slice(session, dim_x, dim_y, list(idx_tuple))
    data = apply_complex_mode(raw, complex_mode)
    h, w = data.shape
    xa = max(0, min(x0, x1, w - 1))
    xb = min(w, max(x0, x1) + 1)
    ya = max(0, min(y0, y1, h - 1))
    yb = min(h, max(y0, y1) + 1)
    roi = data[ya:yb, xa:xb]
    if roi.size == 0:
        return {"error": "empty selection"}
    finite = roi[np.isfinite(roi)]
    return {
        "min": _safe_float(finite.min()) if finite.size else None,
        "max": _safe_float(finite.max()) if finite.size else None,
        "mean": _safe_float(finite.mean()) if finite.size else None,
        "std": _safe_float(finite.std()) if finite.size else None,
        "n": int(finite.size),
    }


@app.get("/info/{sid}")
def get_info(sid: str):
    session = SESSIONS.get(sid)
    if not session:
        return Response(status_code=404)

    try:
        dtype_str = str(session.data.dtype)
    except AttributeError:
        dtype_str = "unknown"
    info: dict = {
        "shape": list(session.shape),
        "dtype": dtype_str,
        "ndim": len(session.shape),
        "total_elements": int(np.prod(session.shape)),
        "is_complex": bool(np.iscomplexobj(session.data)),
        "filepath": session.filepath,
    }
    try:
        info["size_mb"] = round(session.data.nbytes / 1024**2, 2)
    except AttributeError:
        info["size_mb"] = None
    if session.fft_axes is not None:
        info["fft_axes"] = list(session.fft_axes)
    return info


@app.post("/fft/{sid}")
async def toggle_fft(sid: str, request: Request):
    session = SESSIONS.get(sid)
    if not session:
        return {"error": "Invalid session"}

    body = await request.json()
    axes_str = str(body.get("axes", "")).strip()

    if session.fft_original_data is not None:
        session.data = session.fft_original_data
        session.shape = session.data.shape
        session.fft_original_data = None
        session.fft_axes = None
        session.raw_cache.clear()
        session.rgba_cache.clear()
        session.mosaic_cache.clear()
        session._raw_bytes = session._rgba_bytes = session._mosaic_bytes = 0
        session.compute_global_stats()
        return {"status": "restored", "is_complex": bool(np.iscomplexobj(session.data))}

    try:
        axes = tuple(int(a.strip()) for a in axes_str.split(",") if a.strip())
        if not axes:
            raise ValueError("No axes specified")
    except Exception as e:
        return {"error": str(e)}

    # Phase 4 guardrail: FFT materialises the full array — block for large data
    est = _estimate_array_bytes(session)
    if est > HEAVY_OP_LIMIT_BYTES:
        limit_mb = HEAVY_OP_LIMIT_BYTES // (1024 * 1024)
        est_mb = est // (1024 * 1024)
        return {
            "error": (
                f"FFT blocked: array is ~{est_mb} MB (limit {limit_mb} MB). "
                "Convert to a smaller sub-volume or increase "
                "ARRAYVIEW_HEAVY_OP_LIMIT_MB."
            ),
            "too_large": True,
        }

    session.fft_original_data = session.data
    full = np.array(session.data)
    session.data = np.fft.fftshift(np.fft.fftn(full, axes=axes), axes=axes)
    session.shape = session.data.shape
    session.fft_axes = axes
    session.raw_cache.clear()
    session.rgba_cache.clear()
    session.mosaic_cache.clear()
    session._raw_bytes = session._rgba_bytes = session._mosaic_bytes = 0
    session.compute_global_stats()
    return {
        "status": "fft_applied",
        "axes": list(axes),
        "is_complex": bool(np.iscomplexobj(session.data)),
    }


@app.post("/set_rgb/{sid}")
async def set_rgb_endpoint(sid: str, request: Request):
    """Toggle RGB rendering for a session.

    Body: {"axis": int | null}
    - axis=null  → disable RGB mode (restore full shape)
    - axis=int   → treat that dimension as the RGB/RGBA channel axis
                   (must have size 3 or 4)
    """
    session = SESSIONS.get(sid)
    if not session:
        return {"error": "session not found"}
    body = await request.json()
    axis = body.get("axis")
    if axis is None:
        session.rgb_axis = None
        session.spatial_shape = session.data.shape
    else:
        axis = int(axis)
        if not (0 <= axis < len(session.data.shape)):
            return {
                "error": f"axis {axis} out of range for shape {list(session.data.shape)}"
            }
        if session.data.shape[axis] not in (3, 4):
            return {
                "error": f"dim {axis} has size {session.data.shape[axis]}, need 3 or 4 for RGB/RGBA"
            }
        session.rgb_axis = axis
        session.spatial_shape = tuple(
            s for i, s in enumerate(session.data.shape) if i != axis
        )
    session.rgba_cache.clear()
    session._rgba_bytes = 0
    return {
        "ok": True,
        "is_rgb": session.rgb_axis is not None,
        "spatial_shape": list(session.spatial_shape),
    }


@app.get("/slice/{sid}")
def get_slice(
    sid: str,
    dim_x: int,
    dim_y: int,
    indices: str,
    colormap: str = "gray",
    dr: int = 1,
    slice_dim: int = -1,
    dim_z: int = -1,
    complex_mode: int = 0,
    log_scale: bool = False,
    vmin_override: float | None = None,
    vmax_override: float | None = None,
    overlay_sid: str | None = None,
):
    session = SESSIONS.get(sid)
    if not session:
        return Response(status_code=404)
    idx_tuple = tuple(int(x) for x in indices.split(","))
    if dim_z >= 0:
        rgba = render_mosaic(
            session,
            dim_x,
            dim_y,
            dim_z,
            idx_tuple,
            colormap,
            dr,
            complex_mode,
            log_scale,
        )
        idx_norm = list(idx_tuple)
        idx_norm[dim_z] = 0
        frames_raw = [
            extract_slice(
                session,
                dim_x,
                dim_y,
                [i if j == dim_z else idx_tuple[j] for j in range(len(session.shape))],
            )
            for i in range(session.shape[dim_z])
        ]
        frames = [apply_complex_mode(frame, complex_mode) for frame in frames_raw]
        if log_scale:
            frames = [np.log1p(np.abs(frame)).astype(np.float32) for frame in frames]
            pct_lo, pct_hi = DR_PERCENTILES[dr % len(DR_PERCENTILES)]
            all_data = np.stack(frames)
            vmin = float(np.percentile(all_data, pct_lo))
            vmax = float(np.percentile(all_data, pct_hi))
        else:
            vmin, vmax = _compute_vmin_vmax(session, np.stack(frames), dr, complex_mode)
    else:
        if session.rgb_axis is not None:
            rgba = render_rgb_rgba(session, dim_x, dim_y, list(idx_tuple))
            vmin, vmax = 0.0, 255.0
        else:
            rgba = render_rgba(
                session,
                dim_x,
                dim_y,
                idx_tuple,
                colormap,
                dr,
                complex_mode,
                log_scale,
                vmin_override,
                vmax_override,
            )
            mask = _extract_overlay_mask(
                overlay_sid, dim_x, dim_y, idx_tuple, expected_shape=rgba.shape[:2]
            )
            rgba = _composite_overlay_mask(rgba, mask)
            raw = extract_slice(session, dim_x, dim_y, list(idx_tuple))
            _, vmin, vmax = _prepare_display(
                session,
                raw,
                complex_mode,
                dr,
                log_scale,
                vmin_override=vmin_override,
                vmax_override=vmax_override,
            )
    img = Image.fromarray(rgba[:, :, :3], mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return Response(
        content=buf.getvalue(),
        media_type="image/jpeg",
        headers={
            "Cache-Control": "max-age=300",
            "X-ArrayView-Vmin": str(vmin),
            "X-ArrayView-Vmax": str(vmax),
        },
    )


def _render_normalized(session, dim_x, dim_y, idx_tuple, dr, complex_mode, log_scale):
    """Extract a slice and normalize to [0, 1] float32 using per-slice display range."""
    raw = extract_slice(session, dim_x, dim_y, list(idx_tuple))
    data, vmin, vmax = _prepare_display(session, raw, complex_mode, dr, log_scale)
    if vmax > vmin:
        normalized = np.clip((data - vmin) / (vmax - vmin), 0, 1)
    else:
        normalized = np.zeros_like(data)
    return normalized.astype(np.float32)


def _render_normalized_mosaic(
    session, dim_x, dim_y, dim_z, idx_tuple, dr, complex_mode, log_scale
):
    """Return (float32 normalized mosaic grid [0,1], nan_mask) for all z-slices."""
    n = session.shape[dim_z]
    idx_list = list(idx_tuple)
    frames_raw = [
        extract_slice(
            session,
            dim_x,
            dim_y,
            [i if j == dim_z else idx_list[j] for j in range(len(session.shape))],
        )
        for i in range(n)
    ]
    frames = [apply_complex_mode(f, complex_mode) for f in frames_raw]
    if log_scale:
        frames = [np.log1p(np.abs(f)).astype(np.float32) for f in frames]
    all_data = np.stack(frames)
    if log_scale:
        pct_lo, pct_hi = DR_PERCENTILES[dr % len(DR_PERCENTILES)]
        vmin = float(np.percentile(all_data, pct_lo))
        vmax = float(np.percentile(all_data, pct_hi))
    else:
        vmin, vmax = _compute_vmin_vmax(session, all_data, dr, complex_mode)
    rows, cols = mosaic_shape(n)
    H, W = frames[0].shape
    GAP = 2
    total_h = rows * H + (rows - 1) * GAP
    total_w = cols * W + (cols - 1) * GAP
    grid = np.full((total_h, total_w), np.nan, dtype=np.float32)
    for k in range(n):
        r, c = divmod(k, cols)
        r0, c0 = r * (H + GAP), c * (W + GAP)
        grid[r0 : r0 + H, c0 : c0 + W] = all_data[k]
    nan_mask = np.isnan(grid)
    if vmax > vmin:
        normalized = np.clip(
            np.where(nan_mask, 0.0, (grid - vmin) / (vmax - vmin)), 0, 1
        )
    else:
        normalized = np.zeros_like(grid)
    return normalized.astype(np.float32), nan_mask


@app.get("/diff/{sid_a}/{sid_b}")
def get_diff(
    sid_a: str,
    sid_b: str,
    dim_x: int,
    dim_y: int,
    indices: str,
    dim_z: int = -1,
    dr: int = 1,
    complex_mode: int = 0,
    log_scale: bool = False,
    diff_mode: int = 1,
):
    session_a = SESSIONS.get(sid_a)
    session_b = SESSIONS.get(sid_b)
    if not session_a or not session_b:
        return Response(status_code=404)
    idx_tuple = tuple(int(x) for x in indices.split(","))
    # Use the shorter index tuple to handle mismatched dimensionalities
    ndim_a = len(session_a.shape)
    ndim_b = len(session_b.shape)
    idx_a = idx_tuple[:ndim_a]
    idx_b = idx_tuple[:ndim_b]
    nan_mask = None
    try:
        if dim_z >= 0:
            a, nan_mask_a = _render_normalized_mosaic(
                session_a, dim_x, dim_y, dim_z, idx_a, dr, complex_mode, log_scale
            )
            b, nan_mask_b = _render_normalized_mosaic(
                session_b, dim_x, dim_y, dim_z, idx_b, dr, complex_mode, log_scale
            )
            nan_mask = nan_mask_a | nan_mask_b
        else:
            a = _render_normalized(
                session_a, dim_x, dim_y, idx_a, dr, complex_mode, log_scale
            )
            b = _render_normalized(
                session_b, dim_x, dim_y, idx_b, dr, complex_mode, log_scale
            )
    except Exception:
        return Response(status_code=422)
    # Resize b to match a if shapes differ
    if a.shape != b.shape:
        try:
            from PIL import Image as _Image

            b_img = _Image.fromarray((b * 255).astype(np.uint8), mode="L")
            b_img = b_img.resize((a.shape[1], a.shape[0]), _Image.BILINEAR)
            b = np.array(b_img, dtype=np.float32) / 255.0
        except Exception:
            return Response(status_code=422)
    if diff_mode == 1:
        raw = a - b
        vmin, vmax = -1.0, 1.0
        colormap = "RdBu_r"
    elif diff_mode == 2:
        raw = np.abs(a - b)
        vmax = float(raw.max()) or 1.0
        vmin = 0.0
        colormap = "viridis"
    else:  # diff_mode == 3
        raw = np.abs(a - b) / np.maximum(np.abs(a), 1e-6)
        raw = np.clip(raw, 0.0, 2.0).astype(np.float32)
        vmax = float(raw.max()) or 1.0
        vmin = 0.0
        colormap = "viridis"
    if vmax > vmin:
        normalized = np.clip((raw - vmin) / (vmax - vmin), 0, 1)
    else:
        normalized = np.zeros_like(raw)
    _ensure_lut(colormap)
    lut = LUTS.get(colormap, LUTS["gray"])
    rgba = lut[(normalized * 255).astype(np.uint8)]
    if nan_mask is not None and nan_mask.shape == rgba.shape[:2]:
        rgba[nan_mask] = [22, 22, 22, 255]  # dark separator (matches mosaic_render)
    img = Image.fromarray(rgba[:, :, :3], mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return Response(
        content=buf.getvalue(),
        media_type="image/jpeg",
        headers={
            "Cache-Control": "no-cache",
            "X-ArrayView-Vmin": str(vmin),
            "X-ArrayView-Vmax": str(vmax),
        },
    )


@app.get("/oblique/{sid}")
def get_oblique(
    sid: str,
    center: str,  # comma-sep floats — full N-dim index (e.g. "32.0,30.0,35.0")
    basis_h: str,  # 3 floats in mv_dims order, unit vector → horizontal
    basis_v: str,  # 3 floats in mv_dims order, unit vector → vertical
    mv_dims: str,  # 3 ints: which array dims are the 3 spatial dims
    size_w: int,
    size_h: int,
    colormap: str = "gray",
    dr: int = 1,
    complex_mode: int = 0,
    log_scale: bool = False,
    vmin_override: float | None = None,
    vmax_override: float | None = None,
):
    """Render an oblique (arbitrarily-oriented) slice through a 3-D volume."""
    session = SESSIONS.get(sid)
    if not session:
        return Response(status_code=404)

    from scipy.ndimage import map_coordinates

    ctr = [float(x) for x in center.split(",")]
    bh = [float(x) for x in basis_h.split(",")]
    bv = [float(x) for x in basis_v.split(",")]
    dims = [int(x) for x in mv_dims.split(",")]

    ndim = len(session.shape)
    hw, hh = size_w / 2.0, size_h / 2.0
    s_arr = np.arange(size_w, dtype=np.float64) - hw
    t_arr = np.arange(size_h, dtype=np.float64) - hh
    ss, tt = np.meshgrid(s_arr, t_arr)  # (size_h, size_w)

    # Build full N-dim coordinate grids; non-spatial dims use fixed center value
    coords = np.empty((ndim, size_h, size_w), dtype=np.float64)
    for ai in range(ndim):
        if ai in dims:
            ji = dims.index(ai)
            coords[ai] = ctr[ai] + ss * bh[ji] + tt * bv[ji]
        else:
            coords[ai] = ctr[ai]

    data = session.data
    if np.iscomplexobj(data):
        if complex_mode == 1:
            data_f = np.angle(data).astype(np.float32)
        elif complex_mode == 2:
            data_f = data.real.astype(np.float32)
        elif complex_mode == 3:
            data_f = data.imag.astype(np.float32)
        else:
            data_f = np.abs(data).astype(np.float32)
    else:
        data_f = np.nan_to_num(np.asarray(data, dtype=np.float32))

    sampled = map_coordinates(
        data_f, coords, order=1, mode="constant", cval=0.0
    ).astype(np.float32)

    if log_scale:
        sampled = np.log1p(np.abs(sampled)).astype(np.float32)

    if vmin_override is not None and vmax_override is not None:
        vmin, vmax = float(vmin_override), float(vmax_override)
    else:
        vmin, vmax = _compute_vmin_vmax(session, sampled, dr, complex_mode)

    _ensure_lut(colormap)
    lut = LUTS.get(colormap, LUTS["gray"])
    if vmax > vmin:
        normalized = np.clip((sampled - vmin) / (vmax - vmin), 0, 1)
    else:
        normalized = np.zeros_like(sampled)
    rgba = lut[(normalized * 255).astype(np.uint8)]

    img = Image.fromarray(rgba[:, :, :3], mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return Response(
        content=buf.getvalue(),
        media_type="image/jpeg",
        headers={
            "Cache-Control": "no-store",
            "X-ArrayView-Vmin": str(vmin),
            "X-ArrayView-Vmax": str(vmax),
        },
    )


@app.get("/grid/{sid}")
def get_grid(
    sid: str,
    dim_x: int,
    dim_y: int,
    indices: str,
    slice_dim: int,
    colormap: str = "gray",
    dr: int = 1,
):
    session = SESSIONS.get(sid)
    if not session:
        return Response(status_code=404)
    if session.rgb_axis is not None:
        return JSONResponse(
            status_code=400, content={"error": "not supported for RGB sessions"}
        )
    idx_list = [int(x) for x in indices.split(",")]
    n = session.shape[slice_dim]

    # Phase 4 guardrail: grid stacks all n slices — block for large data
    try:
        itemsize = np.dtype(session.data.dtype).itemsize
    except Exception:
        itemsize = 4
    frame_bytes = session.shape[dim_y] * session.shape[dim_x] * itemsize
    if frame_bytes * n > HEAVY_OP_LIMIT_BYTES:
        limit_mb = HEAVY_OP_LIMIT_BYTES // (1024 * 1024)
        est_mb = frame_bytes * n // (1024 * 1024)
        return JSONResponse(
            status_code=400,
            content={
                "error": (
                    f"Grid blocked: would stack ~{est_mb} MB (limit {limit_mb} MB). "
                    "Increase ARRAYVIEW_HEAVY_OP_LIMIT_MB to override."
                ),
                "too_large": True,
            },
        )

    frames = []
    for i in range(n):
        idx_list[slice_dim] = i
        frames.append(extract_slice(session, dim_x, dim_y, idx_list))

    all_data = np.stack(frames)
    if dr in session.global_stats:
        vmin, vmax = session.global_stats[dr]
    else:
        pct_lo, pct_hi = DR_PERCENTILES[dr % len(DR_PERCENTILES)]
        vmin = float(np.percentile(all_data, pct_lo))
        vmax = float(np.percentile(all_data, pct_hi))

    rows, cols = mosaic_shape(n)
    H, W = frames[0].shape
    GAP = 2
    total_h = rows * H + (rows - 1) * GAP
    total_w = cols * W + (cols - 1) * GAP
    grid = np.full((total_h, total_w), np.nan, dtype=np.float32)
    for k in range(n):
        r, c = divmod(k, cols)
        grid[r * (H + GAP) : r * (H + GAP) + H, c * (W + GAP) : c * (W + GAP) + W] = (
            all_data[k]
        )

    nan_mask = np.isnan(grid)
    filled = np.where(nan_mask, vmin, grid)
    if vmax > vmin:
        normalized = np.clip((filled - vmin) / (vmax - vmin), 0, 1)
    else:
        normalized = np.zeros_like(filled)

    lut = LUTS.get(colormap if colormap in LUTS else "gray", LUTS["gray"])
    rgba = lut[(normalized * 255).astype(np.uint8)]
    rgba[nan_mask] = [22, 22, 22, 255]
    img = Image.fromarray(rgba[:, :, :3], mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")


@app.get("/gif/{sid}")
def get_gif(
    sid: str,
    dim_x: int,
    dim_y: int,
    indices: str,
    slice_dim: int,
    colormap: str = "gray",
    dr: int = 1,
):
    session = SESSIONS.get(sid)
    if not session:
        return Response(status_code=404)
    if session.rgb_axis is not None:
        return JSONResponse(
            status_code=400, content={"error": "not supported for RGB sessions"}
        )
    idx_list = [int(x) for x in indices.split(",")]
    n = session.shape[slice_dim]

    # Phase 4 guardrail: GIF stacks all n slices — block for large data
    try:
        itemsize = np.dtype(session.data.dtype).itemsize
    except Exception:
        itemsize = 4
    frame_bytes = session.shape[dim_y] * session.shape[dim_x] * itemsize
    if frame_bytes * n > HEAVY_OP_LIMIT_BYTES:
        limit_mb = HEAVY_OP_LIMIT_BYTES // (1024 * 1024)
        est_mb = frame_bytes * n // (1024 * 1024)
        return JSONResponse(
            status_code=400,
            content={
                "error": (
                    f"GIF blocked: would stack ~{est_mb} MB (limit {limit_mb} MB). "
                    "Increase ARRAYVIEW_HEAVY_OP_LIMIT_MB to override."
                ),
                "too_large": True,
            },
        )

    frames = []
    for i in range(n):
        idx_list[slice_dim] = i
        frames.append(extract_slice(session, dim_x, dim_y, idx_list))

    all_data = np.stack(frames)
    if dr in session.global_stats:
        vmin, vmax = session.global_stats[dr]
    else:
        pct_lo, pct_hi = DR_PERCENTILES[dr % len(DR_PERCENTILES)]
        vmin = float(np.percentile(all_data, pct_lo))
        vmax = float(np.percentile(all_data, pct_hi))

    lut = LUTS.get(colormap if colormap in LUTS else "gray", LUTS["gray"])
    gif_frames = []
    for frame in frames:
        if vmax > vmin:
            normalized = np.clip((frame - vmin) / (vmax - vmin), 0, 1)
        else:
            normalized = np.zeros_like(frame)
        rgba = lut[(normalized * 255).astype(np.uint8)]
        gif_frames.append(Image.fromarray(rgba[:, :, :3], mode="RGB"))

    buf = io.BytesIO()
    gif_frames[0].save(
        buf,
        format="GIF",
        save_all=True,
        append_images=gif_frames[1:],
        loop=0,
        duration=100,
    )
    return Response(content=buf.getvalue(), media_type="image/gif")


@app.get("/shell")
def get_shell():
    """Tabbed shell UI for native webview windows."""
    return HTMLResponse(content=_SHELL_HTML)


@app.get("/ping")
def ping():
    """Health marker so clients can verify this is an ArrayView server."""
    return {
        "ok": True,
        "service": "arrayview",
        "pid": os.getpid(),
        "viewer_sockets": VIEWER_SOCKETS,
    }


_SUPPORTED_EXTS = frozenset(
    [
        ".npy",
        ".npz",
        ".nii",
        ".nii.gz",
        ".zarr",
        ".zarr.zip",
        ".pt",
        ".pth",
        ".h5",
        ".hdf5",
        ".tif",
        ".tiff",
        ".mat",
    ]
)


def _peek_file_shape(fpath: str, ext: str):
    """Try to return shape quickly without loading the full array. Returns None on failure."""
    try:
        if ext == ".npy":
            arr = np.load(fpath, mmap_mode="r", allow_pickle=False)
            return list(arr.shape)
        if ext in (".nii", ".nii.gz"):
            return list(nib.load(fpath).shape)
    except Exception:
        pass
    return None


@app.get("/listfiles")
def list_files(directory: str = ""):
    """List supported array files recursively (depth ≤ 4, max 300 files).

    Returns entries sorted by relative path; files directly in the target
    directory use just the filename as ``name``, nested files use a relative
    path (e.g. ``subdir/scan.npy``).  Hidden directories (name starts with
    ``.``) are skipped.
    """
    target = os.path.abspath(directory) if directory else os.getcwd()
    MAX_FILES = 300
    MAX_DEPTH = 4
    results = []
    try:
        for root, dirs, files in os.walk(target):
            rel_root = os.path.relpath(root, target)
            depth = 0 if rel_root == "." else rel_root.count(os.sep) + 1
            # Prune hidden dirs and stop recursing beyond MAX_DEPTH
            dirs[:] = sorted(d for d in dirs if not d.startswith("."))
            if depth >= MAX_DEPTH:
                dirs.clear()
            for fname in sorted(files):
                name_lower = fname.lower()
                ext = (
                    ".nii.gz"
                    if name_lower.endswith(".nii.gz")
                    else os.path.splitext(name_lower)[1]
                )
                if ext not in _SUPPORTED_EXTS:
                    continue
                fpath = os.path.join(root, fname)
                rel = os.path.relpath(fpath, target)
                name = fname if root == target else rel
                try:
                    file_size = os.path.getsize(fpath)
                except OSError:
                    continue
                shape = _peek_file_shape(fpath, ext)
                results.append(
                    {"name": name, "path": fpath, "size": file_size, "shape": shape}
                )
                if len(results) >= MAX_FILES:
                    break
            if len(results) >= MAX_FILES:
                break
    except Exception as e:
        return {"error": str(e)}
    return results


@app.get("/sessions")
def get_sessions():
    """Returns list of active sessions (used by shell to populate tabs on load)."""
    return [
        {
            "sid": s.sid,
            "name": s.name,
            "shape": [int(x) for x in s.shape],
            "filepath": s.filepath,
        }
        for s in SESSIONS.values()
    ]


@app.post("/load")
async def load_file(request: Request):
    """Load a file into a new session. Optionally notify webview shells."""
    body = await request.json()
    filepath = str(body["filepath"])
    name = str(body.get("name") or os.path.basename(filepath))
    notify = bool(body.get("notify", False))
    try:
        data = await asyncio.to_thread(load_data, filepath)
    except Exception as e:
        return {"error": str(e)}
    session = Session(data, filepath=filepath, name=name)
    if body.get("rgb"):
        try:
            _setup_rgb(session)
        except ValueError as e:
            return {"error": str(e)}
    SESSIONS[session.sid] = session
    notified = False
    if notify:
        tab_url = None
        if body.get("compare_sids"):
            tab_url = (
                f"/?sid={session.sid}"
                f"&compare_sid={body['compare_sid']}"
                f"&compare_sids={body['compare_sids']}"
            )
        # wait=False: the shell window should already be connected for inject-into-existing.
        # If no shells are connected the native window is gone and the caller must open a new one.
        notified = await _notify_shells(session.sid, name, url=tab_url, wait=False)
    return {"sid": session.sid, "name": name, "notified": notified}


@app.get("/")
def get_ui(sid: str = None):
    """Viewer page."""
    # VS Code Simple Browser internally calls asExternalUri() which strips query
    # parameters, so ?sid= is often lost before the page loads.  Embed the SID
    # directly in the HTML so the viewer JS can find it regardless of the URL.
    if not sid:
        # No sid in URL — VS Code Simple Browser strips the query string before
        # loading the page, so ?sid= is lost.  Inject the latest valid session
        # server-side so the viewer JS can find it regardless of the URL.
        if SESSIONS:
            latest_sid = list(SESSIONS.keys())[-1]
            query_val = json.dumps(f"?sid={latest_sid}&transport=http")
        else:
            query_val = "null"  # viewer will show "Session not found or expired"
    else:
        # sid is present in the URL (valid or not) — let the JS fetch /metadata/{sid}
        # and handle errors itself (shows "Session not found or expired" on 404).
        query_val = "null"
    html = (
        _VIEWER_HTML_TEMPLATE.replace("__COLORMAPS__", str(COLORMAPS))
        .replace("__DR_LABELS__", str(DR_LABELS))
        .replace("__COLORMAP_GRADIENT_STOPS__", json.dumps(COLORMAP_GRADIENT_STOPS))
        .replace("__COMPLEX_MODES__", str(COMPLEX_MODES))
        .replace("__REAL_MODES__", str(REAL_MODES))
        .replace("__ARRAYVIEW_QUERY__", query_val)
    )
    headers = {"Cache-Control": "no-store"}
    return HTMLResponse(content=html, headers=headers)


# ---------------------------------------------------------------------------
# Jupyter / in-process API
# ---------------------------------------------------------------------------
_jupyter_server_port: int | None = None


def _in_jupyter() -> bool:
    try:
        from IPython import get_ipython

        shell = get_ipython()
        if shell is None:
            return False
        return "ipykernel" in type(shell).__module__
    except ImportError:
        return False


def _in_vscode_terminal() -> bool:
    """True when running inside any VS Code integrated terminal (local or remote)."""
    if os.environ.get("TERM_PROGRAM") == "vscode":
        return True
    if os.environ.get("VSCODE_IPC_HOOK_CLI"):
        return True
    # uv run and similar launchers strip env vars; walk ancestor processes.
    return _find_vscode_ipc_hook() is not None


def _is_vscode_remote() -> bool:
    """True when running inside a VS Code remote/tunnel session.

    This covers:
    - Linux SSH remote (VSCODE_IPC_HOOK_CLI set, non-macOS/Windows)
    - macOS/Windows SSH remote (SSH_CONNECTION set)
    - macOS/Windows tunnel remote: detected by finding a remote-cli binary in
      ~/.vscode/cli/servers/ or ~/.vscode-server/ (placed there by the tunnel
      server), which is only present when this machine IS the tunnel remote.
    """
    # Try env var first, then walk process tree (handles uv run env stripping).
    ipc = os.environ.get("VSCODE_IPC_HOOK_CLI") or _find_vscode_ipc_hook()
    if ipc:
        if sys.platform not in ("darwin", "win32"):
            return True
        # SSH remote on macOS/Windows
        if os.environ.get("SSH_CONNECTION") or os.environ.get("SSH_CLIENT"):
            return True
        # Tunnel remote on macOS/Windows: the remote-cli binary is present on
        # this machine only when it is acting as the tunnel server.
        code = _find_code_cli()
        if code and "remote-cli" in code:
            return True
    # TERM_PROGRAM=vscode + SSH_CONNECTION = VS Code SSH remote (belt-and-suspenders)
    if os.environ.get("TERM_PROGRAM") == "vscode" and (
        os.environ.get("SSH_CONNECTION") or os.environ.get("SSH_CLIENT")
    ):
        return True
    return False


def _in_vscode_tunnel() -> bool:
    """True when running inside a VS Code tunnel or SSH remote session."""
    if os.environ.get("SSH_CLIENT") or os.environ.get("SSH_CONNECTION"):
        return True
    if os.environ.get("VSCODE_INJECTION") or os.environ.get("VSCODE_AGENT_FOLDER"):
        return True
    return False


def _can_native_window() -> bool:
    """True if a pywebview native window can be opened.

    Returns False whenever a VS Code IPC hook is detectable, meaning we're
    running inside a VS Code terminal (local or remote/tunnel).  In that case
    we always prefer the Simple Browser route over a native window, because on
    a tunnel-server machine the user isn't looking at that screen.
    """
    # If a VS Code IPC hook is findable we are inside a VS Code terminal.
    # Prefer the browser route regardless of platform.
    if _find_vscode_ipc_hook():
        return False
    if _is_vscode_remote():
        return False
    # Plain SSH (no VS Code): the display is on the client machine, not here.
    if os.environ.get("SSH_CLIENT") or os.environ.get("SSH_CONNECTION"):
        return False
    if sys.platform in ("darwin", "win32"):
        return True
    # Linux/BSD: need a display server AND pywebview's GUI bindings
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        return False
    import importlib.util

    return (
        importlib.util.find_spec("qtpy") is not None
        or importlib.util.find_spec("gi") is not None
    )


def _find_code_cli() -> str | None:
    """Return path to the VS Code CLI ('code'), or None if not found.

    In a VS Code tunnel/remote, the tunnel server provides its own ``code``
    helper at ``~/.vscode-server/.../remote-cli/code``.  We prefer that over
    a desktop ``code`` when ``VSCODE_IPC_HOOK_CLI`` is set, because the
    desktop binary would open a *new* VS Code window instead of routing
    through the tunnel.
    """
    import glob
    import shutil

    # In a VS Code remote/tunnel, prefer the server's remote-cli helper.
    # uv run and similar launchers may strip VSCODE_IPC_HOOK_CLI from the
    # current process, so also consult the recovered ancestor-process value.
    if os.environ.get("VSCODE_IPC_HOOK_CLI") or _find_vscode_ipc_hook():
        # The tunnel helper is typically at one of:
        #   ~/.vscode-server/bin/<commit>/bin/remote-cli/code
        #   ~/.vscode-server/cli/servers/.../server/bin/remote-cli/code
        #   ~/.vscode/cli/servers/.../server/bin/remote-cli/code  (newer tunnels)
        for pattern in [
            os.path.expanduser("~/.vscode-server/bin/*/bin/remote-cli/code"),
            os.path.expanduser(
                "~/.vscode-server/cli/servers/*/server/bin/remote-cli/code"
            ),
            os.path.expanduser("~/.vscode/cli/servers/*/server/bin/remote-cli/code"),
        ]:
            matches = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
            for m in matches:
                if os.access(m, os.X_OK):
                    return m

    found = shutil.which("code")
    if found:
        return found
    candidates: list[str] = []
    if sys.platform == "darwin":
        candidates = [
            "/opt/homebrew/bin/code",
            "/usr/local/bin/code",
            os.path.expanduser(
                "~/Applications/Visual Studio Code.app/Contents/Resources/app/bin/code"
            ),
            "/Applications/Visual Studio Code.app/Contents/Resources/app/bin/code",
        ]
    elif sys.platform.startswith("linux"):
        candidates = [
            "/usr/bin/code",
            "/usr/local/bin/code",
            "/snap/bin/code",
            os.path.expanduser("~/.local/bin/code"),
        ]
    for path in candidates:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    return None


_VSCODE_IPC_HOOK_CACHE: str | None = "__unset__"  # sentinel; None means not found


def _find_vscode_ipc_hook() -> str | None:
    """Return the value of VSCODE_IPC_HOOK_CLI, searching ancestor processes.

    uv run (and similar launchers) strip environment variables before executing
    Python.  Walking up the process tree lets us recover VSCODE_IPC_HOOK_CLI
    from the shell that originally invoked the command.  Result is cached after
    the first call so repeated calls don't re-walk the process tree.
    """
    global _VSCODE_IPC_HOOK_CACHE
    if _VSCODE_IPC_HOOK_CACHE != "__unset__":
        return _VSCODE_IPC_HOOK_CACHE

    def _ppid(pid: int) -> int:
        try:
            with open(f"/proc/{pid}/status") as fh:
                for line in fh:
                    if line.startswith("PPid:"):
                        return int(line.split()[1])
        except Exception:
            pass
        try:
            r = subprocess.run(
                ["ps", "-p", str(pid), "-o", "ppid="],
                capture_output=True,
                text=True,
                timeout=2,
            )
            return int(r.stdout.strip())
        except Exception:
            pass
        return -1

    def _ipc_from_pid(pid: int) -> str:
        # Linux: /proc/<pid>/environ (null-separated KEY=VALUE pairs)
        try:
            with open(f"/proc/{pid}/environ", "rb") as fh:
                for entry in fh.read().split(b"\0"):
                    if entry.startswith(b"VSCODE_IPC_HOOK_CLI="):
                        return entry[len(b"VSCODE_IPC_HOOK_CLI=") :].decode()
        except Exception:
            pass
        # macOS: `ps ewwww` appends the environment after the argument list.
        # Each env var is a whitespace-separated TOKEN of the form KEY=VALUE.
        try:
            r = subprocess.run(
                ["ps", "ewwww", "-p", str(pid)],
                capture_output=True,
                text=True,
                timeout=3,
            )
            for token in r.stdout.split():
                if token.startswith("VSCODE_IPC_HOOK_CLI="):
                    return token[len("VSCODE_IPC_HOOK_CLI=") :]
        except Exception:
            pass
        return ""

    # Own environment first
    val = os.environ.get("VSCODE_IPC_HOOK_CLI", "")
    if val and os.path.exists(val):
        _VSCODE_IPC_HOOK_CACHE = val
        return val

    # Walk up to 12 ancestor processes
    pid = os.getpid()
    for _ in range(12):
        pid = _ppid(pid)
        if pid <= 1:
            break
        val = _ipc_from_pid(pid)
        if val and os.path.exists(val):
            _VSCODE_IPC_HOOK_CACHE = val
            return val

    _VSCODE_IPC_HOOK_CACHE = None
    return None


def _vscode_app_bundle() -> str | None:
    """Return the path to the VS Code .app bundle on macOS, derived from the code CLI."""
    code = _find_code_cli()
    if not code:
        return None
    try:
        real = os.path.realpath(code)
        idx = real.find(".app")
        if idx != -1:
            return real[: idx + 4]
    except Exception:
        pass
    for candidate in [
        "/Applications/Visual Studio Code.app",
        os.path.expanduser("~/Applications/Visual Studio Code.app"),
    ]:
        if os.path.isdir(candidate):
            return candidate
    return None


_VSCODE_EXT_INSTALLED = False  # cached so we only check once per process
_VSCODE_EXT_FRESH_INSTALL = False  # True if we just installed it this session
_VSCODE_EXT_VERSION = "0.9.7"  # must match vscode-extension/package.json
_VSCODE_SIGNAL_FILENAME = "open-request-v0900.json"
_VSCODE_COMPAT_SIGNAL_FILENAMES: tuple[str, ...] = ("open-request-v0800.json",)
_VSCODE_PORT_SETTINGS_SETTLE_SECONDS = 2.0
_VSCODE_SIGNAL_MAX_AGE_MS = 15_000


def _bundled_vscode_vsix_version(vsix_path: str) -> str | None:
    """Return the bundled opener extension version recorded inside the VSIX."""
    try:
        with zipfile.ZipFile(vsix_path) as zf:
            with zf.open("extension/package.json") as f:
                data = json.load(f)
        version = data.get("version")
        return version if isinstance(version, str) else None
    except Exception as exc:
        print(
            f"[ArrayView] could not inspect VSIX version at {vsix_path}: {exc}",
            flush=True,
        )
        return None


def _patch_vscode_extension_metadata(version: str) -> None:
    """Remove broken targetPlatform metadata written by VS Code for local VSIX installs."""
    for base_dir in (
        os.path.expanduser("~/.vscode-server/extensions"),
        os.path.expanduser("~/.vscode/extensions"),
    ):
        package_json = os.path.join(
            base_dir, f"arrayview.arrayview-opener-{version}", "package.json"
        )
        if not os.path.isfile(package_json):
            continue
        try:
            with open(package_json) as f:
                data = json.load(f)
            metadata = data.get("__metadata")
            if (
                isinstance(metadata, dict)
                and metadata.get("targetPlatform") == "undefined"
            ):
                del metadata["targetPlatform"]
                with open(package_json, "w") as f:
                    json.dump(data, f, indent=8)
                    f.write("\n")
        except Exception as exc:
            print(
                f"[ArrayView] could not patch extension metadata at {package_json}: {exc}",
                flush=True,
            )


def _ensure_vscode_extension() -> bool:
    """Install the bundled arrayview-opener VS Code extension for local VS Code use.

    The extension bridges local VS Code terminals to ``simpleBrowser.show(...)``
    and, in remote/tunnel sessions, can actively invoke VS Code's forwarded-port
    commands to promote the port to public preview.

    Force-installs the current VSIX into the running UI extension host.
    Hot-installing immediately activates the new version alongside any older
    version that may still be running. The extension and Python code use a
    versioned signal filename so stale instances won't consume new requests.

    We do NOT uninstall first: an explicit uninstall causes VS Code to mark the
    extension and skip hot-activation on reinstall (see log.txt attempt 11).
    """
    global _VSCODE_EXT_INSTALLED, _VSCODE_EXT_FRESH_INSTALL
    if _VSCODE_EXT_INSTALLED:
        return True

    code = _find_code_cli()
    if not code:
        return False

    env = dict(os.environ)
    ipc = _find_vscode_ipc_hook()
    if ipc and "VSCODE_IPC_HOOK_CLI" not in env:
        env["VSCODE_IPC_HOOK_CLI"] = ipc

    vsix_path = str(_pkg_files(__package__).joinpath("arrayview-opener.vsix"))
    if not os.path.isfile(vsix_path):
        return False
    bundled_version = _bundled_vscode_vsix_version(vsix_path)
    if bundled_version != _VSCODE_EXT_VERSION:
        print(
            f"[ArrayView] extension version mismatch: bundled={bundled_version!r} "
            f"expected={_VSCODE_EXT_VERSION!r} — rebuild arrayview-opener.vsix",
            flush=True,
        )
        return False
    try:
        r = subprocess.run(
            [code, "--install-extension", vsix_path, "--force"],
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
        )
        combined = (r.stdout or "") + (r.stderr or "")
        install_failed = (
            "Cannot install" in combined
            or "Failed Installing Extensions" in combined
            or "extension/package.json not found inside zip" in combined
            or "Error:" in combined
        )
        if r.returncode == 0 and not install_failed:
            _patch_vscode_extension_metadata(_VSCODE_EXT_VERSION)
            _VSCODE_EXT_INSTALLED = True
            _VSCODE_EXT_FRESH_INSTALL = True
            return True
        print(f"[ArrayView] extension install failed: {combined.strip()!r}", flush=True)
    except Exception as exc:
        print(f"[ArrayView] extension install error: {exc}", flush=True)
    return False


def _configure_vscode_port_preview(port: int) -> bool:
    """Write VS Code port settings for the arrayview server.

    In VS Code remote/tunnel sessions this writes both Machine and User
    settings files to maximize the chance VS Code honors them. Workspace-
    level settings are unreliable for port privacy in tunnel sessions.

    In local VS Code terminals we keep the workspace-level attribute so
    auto-forward/silent remains configured when relevant.

    Returns True on success.
    """

    def _strip_json_comments(raw: str) -> str:
        raw = re.sub(r"/\*.*?\*/", "", raw, flags=re.DOTALL)
        raw = re.sub(r"(^|\s)//.*$", "", raw, flags=re.MULTILINE)
        return raw

    def _load_settings(path: str) -> dict:
        if not os.path.exists(path):
            return {}
        try:
            with open(path) as f:
                raw = f.read()
            cleaned = _strip_json_comments(raw)
            return json.loads(cleaned) if cleaned.strip() else {}
        except (json.JSONDecodeError, OSError):
            return {}

    def _write_settings(path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        settings = _load_settings(path)
        attrs = settings.setdefault("remote.portsAttributes", {})
        attrs[str(port)] = {
            "protocol": "http",
            "label": "ArrayView",
            "onAutoForward": "silent",
            "privacy": "public",
        }
        with open(path, "w") as f:
            json.dump(settings, f, indent=2)
            f.write("\n")

    try:
        in_vscode = _in_vscode_terminal()
        is_remote = _is_vscode_remote()

        if is_remote:
            home = os.path.expanduser("~")
            targets: list[str] = []
            for root in (
                os.path.join(home, ".vscode", "cli"),
                os.path.join(home, ".vscode-server"),
            ):
                if os.path.isdir(root):
                    targets.append(
                        os.path.join(root, "data", "Machine", "settings.json")
                    )
                    targets.append(os.path.join(root, "data", "User", "settings.json"))
            if not targets:
                # Fallback: write to the most common paths even if root
                # directories don't exist yet.
                for root in (
                    os.path.join(home, ".vscode-server"),
                    os.path.join(home, ".vscode", "cli"),
                ):
                    targets.append(
                        os.path.join(root, "data", "Machine", "settings.json")
                    )
                    targets.append(os.path.join(root, "data", "User", "settings.json"))

            for settings_path in targets:
                _write_settings(settings_path)
            return True

        if in_vscode:
            settings_path = os.path.join(os.getcwd(), ".vscode", "settings.json")
            _write_settings(settings_path)
        return True
    except Exception as exc:
        print(f"[ArrayView] could not write port settings: {exc}", flush=True)
        return False


def _open_via_signal_file(url: str, delay: float = 0.0) -> bool:
    """Write the URL to the versioned ArrayView opener signal file."""
    return _write_vscode_signal(
        {
            "action": "open-preview",
            "url": url,
            "maxAgeMs": _VSCODE_SIGNAL_MAX_AGE_MS,
        },
        delay=delay,
    )


def _schedule_remote_open_retries(
    url: str, interval: float = 7.0, count: int = 6
) -> None:
    """Re-send the open-preview signal if no viewer WebSocket has connected yet.

    In remote/tunnel sessions the port may be private on first open (user sees
    auth page). Once the user sets Port Visibility → Public, the next retry
    re-opens Simple Browser with the now-public URL — no need to re-run arrayview.
    Retries stop as soon as VIEWER_SOCKETS > 0 (viewer loaded successfully).
    """

    def _loop() -> None:
        for i in range(count):
            time.sleep(interval)
            if VIEWER_SOCKETS > 0:
                return  # viewer connected; port is public and working
            _open_via_signal_file(url)

    threading.Thread(target=_loop, daemon=True).start()


def _write_vscode_signal(payload: dict, delay: float = 0.0) -> bool:
    """Write a versioned control payload for the VS Code opener extension."""
    signal_dir = os.path.expanduser("~/.arrayview")
    try:
        os.makedirs(signal_dir, exist_ok=True)
        if delay > 0:
            time.sleep(delay)
        data = dict(payload)
        data.setdefault("sentAtMs", int(time.time() * 1000))
        data.setdefault("maxAgeMs", _VSCODE_SIGNAL_MAX_AGE_MS)
        data.setdefault("requestId", uuid.uuid4().hex)
        filenames = (_VSCODE_SIGNAL_FILENAME, *_VSCODE_COMPAT_SIGNAL_FILENAMES)
        # Delete any existing signal files before writing the new one.  If the
        # extension failed to unlink a previous file (e.g. a crash) and then
        # restarts (resetting lastHandledRequestId), it would otherwise re-consume
        # the stale signal and open Simple Browser with an old session ID.
        for filename in filenames:
            try:
                os.unlink(os.path.join(signal_dir, filename))
            except FileNotFoundError:
                pass
        for filename in filenames:
            signal_file = os.path.join(signal_dir, filename)
            with open(signal_file, "w") as f:
                json.dump(data, f)
        return True
    except Exception:
        return False


def _print_viewer_location(url: str) -> None:
    """Print a viewer location hint."""
    if not _is_vscode_remote():
        print(f"[ArrayView] {url}", flush=True)


def _open_browser(url: str, blocking: bool = False, force_vscode: bool = False) -> None:
    """Open *url* locally, or configure VS Code remote auto-preview behavior.

    Strategy (see log.txt for what was tried and why):
    1. Remote VS Code terminal:
       a. Configure the port as ``silent`` and ``public`` in
          ``remote.portsAttributes``.
       b. Write the signal file; the workspace extension converts the URL via
          asExternalUri and opens Simple Browser in the local VS Code client.
    2. Local VS Code terminal (or force_vscode=True):
       a. Install the helper extension.
       b. Write the signal file so the extension opens Simple Browser locally.
     3. Fallback: open/xdg-open with the http URL (system browser).
     4. Always print the URL.
    """

    def _do() -> None:
        ipc = _find_vscode_ipc_hook()
        is_remote = _is_vscode_remote()
        opened = False

        try:
            parsed_port = int(url.split(":")[2].split("/")[0].split("?")[0])
        except Exception:
            parsed_port = 8000

        if is_remote:
            # Remote/tunnel: install extension + write signal file.
            # The workspace extension resolves the devtunnel URL via asExternalUri
            # and opens Simple Browser with ?sid= preserved.
            # Port visibility must be set to Public by the user (Ports tab →
            # right-click → Port Visibility → Public).
            print(
                "[ArrayView] Remote tunnel session — set port visibility to Public in the VS Code Ports tab and re-run ArrayView.",
                flush=True,
            )
            ext_ok = _ensure_vscode_extension()
            if ext_ok:
                if _VSCODE_EXT_FRESH_INSTALL:
                    time.sleep(1.5)
                _open_via_signal_file(url)
                _schedule_remote_open_retries(url)
            else:
                print(
                    "[ArrayView] extension install failed — cannot open Simple Browser",
                    flush=True,
                )
            return

        if force_vscode or ipc or _in_vscode_terminal():
            # Local VS Code terminal (or --window vscode forced): install extension + signal file.
            _configure_vscode_port_preview(parsed_port)
            ext_ok = _ensure_vscode_extension()
            if ext_ok:
                if _VSCODE_EXT_FRESH_INSTALL:
                    time.sleep(1.5)
                _open_via_signal_file(url)
                opened = True

        is_plain_ssh = (
            not is_remote
            and not ipc
            and not _in_vscode_terminal()
            and bool(os.environ.get("SSH_CLIENT") or os.environ.get("SSH_CONNECTION"))
        )
        if is_plain_ssh:
            try:
                port_hint = int(url.split(":")[2].split("/")[0].split("?")[0])
            except Exception:
                port_hint = parsed_port
            print(
                f"[ArrayView] SSH session detected — forward the port to access locally:\n"
                f"  ssh -L {port_hint}:localhost:{port_hint} <user>@<remote>\n",
                flush=True,
            )

        if not opened and not is_remote and not force_vscode and not is_plain_ssh:
            # Local fallback: open in system browser
            if sys.platform == "darwin":
                try:
                    r = subprocess.run(["open", url], capture_output=True, timeout=5)
                    opened = r.returncode == 0
                except Exception:
                    pass
            elif sys.platform.startswith("linux"):
                try:
                    r = subprocess.run(
                        ["xdg-open", url], capture_output=True, timeout=5
                    )
                    opened = r.returncode == 0
                except Exception:
                    pass

        # Local sessions still benefit from a clickable terminal URL.
        print(f"\n  \033[1;36m→ {url}\033[0m\n", flush=True)

    if blocking:
        _do()
    else:
        threading.Thread(target=_do, daemon=True).start()


def _server_alive(port: int) -> bool:
    """Return True only if an ArrayView server is responding on the port."""
    url = f"http://127.0.0.1:{port}/ping"
    try:
        with urllib.request.urlopen(url, timeout=0.5) as resp:
            if resp.status != 200:
                return False
            payload = json.loads(resp.read().decode("utf-8"))
            return payload.get("ok") is True and payload.get("service") == "arrayview"
    except Exception:
        return False


def _server_pid(port: int) -> int | None:
    """Return the pid of the responding ArrayView server, or None if unreachable."""
    url = f"http://127.0.0.1:{port}/ping"
    try:
        with urllib.request.urlopen(url, timeout=0.5) as resp:
            if resp.status != 200:
                return None
            payload = json.loads(resp.read().decode("utf-8"))
            if payload.get("ok") is True and payload.get("service") == "arrayview":
                return payload.get("pid")
    except Exception:
        pass
    return None


def _port_in_use(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.3):
            return True
    except OSError:
        return False


def _wait_for_port(port: int, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _server_alive(port):
            return True
        time.sleep(0.05)
    return False


def _find_server_port(port: int, search_range: int = 20) -> tuple[int, bool]:
    """Find a usable server port starting from *port*.

    Returns ``(found_port, already_running)`` where ``already_running`` is True
    if an ArrayView server is already live on ``found_port``.
    """
    if _server_alive(port):
        return (port, True)
    for candidate in range(port, port + search_range + 1):
        if not _port_in_use(candidate):
            return (candidate, False)
    return (port + search_range, False)


async def _serve_background(port: int, stop_when_closed: bool = False):
    global SERVER_LOOP
    SERVER_LOOP = asyncio.get_running_loop()
    import socket as _socket

    # Pre-create the socket with SO_REUSEADDR so we can rebind immediately after
    # a previous server on this port was killed (avoids TIME_WAIT Errno 48).
    sock = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
    sock.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", port))
    sock.listen(128)
    sock.set_inheritable(True)
    config = uvicorn.Config(app, log_level="error", timeout_keep_alive=30)
    server = uvicorn.Server(config)
    if stop_when_closed:
        asyncio.create_task(_stop_server_when_viewer_closes(server))
    await server.serve(sockets=[sock])


def view(
    data,
    name: str = None,
    port: int = 8123,
    inline: bool | None = None,
    height: int = 500,
    window: str | bool | None = None,
    rgb: bool = False,
):
    """
    Launch the viewer. Does not block the main Python process.

    ``window`` controls how the viewer opens:
      - ``None``       auto: native window outside Jupyter, inline IFrame inside Jupyter
      - ``True``       native window (falls back to browser if unavailable)
      - ``False``      no automatic opening; returns URL
      - ``'native'``   open in a native desktop window
      - ``'browser'``  open in the system browser
      - ``'vscode'``   open in VS Code Simple Browser
      - ``'inline'``   return an inline IFrame (Jupyter / VS Code notebook)

    ``rgb`` — treat the array as RGB/RGBA. The first or last dimension must
    have size 3 (RGB) or 4 (RGBA). When True, the colorbar is hidden and each
    slice is composited directly from the colour channels.
    """
    global _jupyter_server_port, _window_process, SERVER_LOOP  # _window_process is a Popen instance

    # Normalise string window modes.
    _force_browser = False
    _force_vscode = False
    if isinstance(window, str):
        _w = window.lower()
        if _w == "inline":
            inline = True
            window = False
        elif _w == "native":
            window = True
        elif _w == "browser":
            window = False
            _force_browser = True
        elif _w == "vscode":
            window = False
            _force_vscode = True
        else:
            raise ValueError(
                f"window must be 'inline', 'native', 'browser', or 'vscode', got {window!r}"
            )

    # Duck-typing: accept PyTorch tensors, JAX arrays, Julia/PythonCall arrays, etc.
    # Zarr and nibabel proxy arrays are already numpy-like and work without conversion.
    if not isinstance(data, np.ndarray):
        _mod = type(data).__module__ or ""
        _is_lazy = "nibabel" in _mod or "zarr" in _mod or "h5py" in _mod
        if not _is_lazy:
            if (
                hasattr(data, "detach")  # PyTorch
                or hasattr(data, "numpy")  # PyTorch / TF / JAX
                or hasattr(data, "__jax_array__")  # JAX
                or "juliacall" in _mod.lower()  # Julia via PythonCall
            ):
                data = _tensor_to_numpy(data, "view()")
            else:
                # Generic fallback: copy to numpy for any array-like
                # (catches PythonCall arrays whose module name varies across versions)
                # Use np.array (copy) so Julia's GC can safely reclaim the source after view() returns.
                try:
                    converted = np.array(data)
                    if converted.dtype != object:
                        data = converted
                except Exception:
                    pass  # keep original; let Session handle it lazily

    if name is None:
        name = f"Array {data.shape}"

    # Julia/PythonCall: must be handled before the is_jupyter defaults because:
    # 1. _in_jupyter() returns False for IJulia kernels (not ipykernel)
    # 2. that would set inline=False and window=True, overriding user intent
    # The subprocess server is always required here due to GIL.
    if _is_julia_env():
        if window is True:
            _inline, _window = False, True
        elif inline is True or window is False:
            # Explicit "not window" → inline (makes sense in Jupyter; harmless elsewhere)
            _inline, _window = True, False
        elif inline is False:
            _inline, _window = False, False  # explicit inline=False → browser
        else:
            # No args: inline in IJulia, window elsewhere.
            _inline = _in_julia_jupyter()
            _window = False
        return _view_julia(
            np.array(data) if not isinstance(data, np.ndarray) else data,
            name,
            port,
            window=_window,
            inline=_inline,
            height=height,
        )

    is_jupyter = _in_jupyter()
    if inline is None:
        inline = is_jupyter
    if window is None:
        window = not is_jupyter
    if window:
        inline = False

    # Julia/PythonCall: the GIL is not released reliably between Julia statements,
    # so an in-process uvicorn thread cannot serve requests once view() returns.
    # Use a fully independent subprocess server instead (same approach as the CLI).
    if not inline and _is_julia_env():
        return _view_julia(
            np.array(data) if not isinstance(data, np.ndarray) else data,
            name,
            port,
            window,
        )

    # VS Code tunnel/remote: the calling Python process may exit shortly after
    # view() returns (one-shot scripts, non-interactive use).  A daemon-thread
    # server would die with the process, leaving the user staring at
    # "session expired" in the browser.  Use a subprocess server that survives
    # the parent's exit.  (In Jupyter inline mode we stay in-process since the
    # notebook kernel is long-lived.)
    if not inline and _is_vscode_remote() and not _in_jupyter():
        return _view_subprocess(
            np.array(data) if not isinstance(data, np.ndarray) else data,
            name,
            port,
            window,
            inline,
            height,
            rgb=rgb,
        )

    session = Session(data, name=name)
    if rgb:
        _setup_rgb(session)
    SESSIONS[session.sid] = session
    win_w, win_h = 1400, 900

    # Start (or restart) the background server if it isn't responding or is stale.
    server_pid = _server_pid(port)
    our_pid = os.getpid()
    if server_pid is not None and server_pid != our_pid:
        # A stale ArrayView server (different process) is on this port — sessions
        # stored in our process won't be visible to it.  Kill it so we can bind.
        print(
            f"[ArrayView] Stale server (pid {server_pid}) on port {port}, terminating it...",
            flush=True,
        )
        import signal as _signal

        try:
            os.kill(server_pid, _signal.SIGTERM)
        except Exception:
            pass
        # Wait up to 1 s for a clean exit, then SIGKILL.
        for _ in range(10):
            if not _port_in_use(port):
                break
            time.sleep(0.1)
        if _port_in_use(port):
            try:
                os.kill(server_pid, _signal.SIGKILL)
            except Exception:
                pass
            # Wait up to 2 more seconds after SIGKILL.
            for _ in range(20):
                if not _port_in_use(port):
                    break
                time.sleep(0.1)
        server_pid = None  # treat as not alive

    if server_pid is None:
        if _port_in_use(port) and not _server_alive(port):
            raise RuntimeError(
                f"Port {port} is already in use by another process. "
                f"Choose a different port in view(..., port=...)."
            )
        SERVER_LOOP = None  # reset so we wait for the new loop below
        _script = _is_script_mode()
        threading.Thread(
            target=lambda: asyncio.run(
                _serve_background(port, stop_when_closed=_script)
            ),
            daemon=not _script,
            name="arrayview-server",
        ).start()
        if not _wait_for_port(port):
            if _port_in_use(port) and not _server_alive(port):
                raise RuntimeError(
                    f"Port {port} is in use by another process (not ArrayView). "
                    f"Choose a different port in view(..., port=...)."
                )
            raise RuntimeError(
                f"ArrayView server did not start on port {port} within timeout."
            )
        _jupyter_server_port = port
    else:
        _jupyter_server_port = port  # server already ours on this port

    # Ensure background server captures the event loop before continuing.
    # If the server was already running in our process, SERVER_LOOP is already set.
    deadline = time.monotonic() + 5.0
    while SERVER_LOOP is None and time.monotonic() < deadline:
        time.sleep(0.01)

    url_viewer = f"http://localhost:{port}/?sid={session.sid}"
    encoded_name = urllib.parse.quote(name)
    url_shell = (
        f"http://localhost:{port}/shell?init_sid={session.sid}&init_name={encoded_name}"
    )

    if inline:
        from IPython.display import IFrame

        return IFrame(src=url_viewer, width="100%", height=height)

    can_native_window = _can_native_window() if window else False
    if window and can_native_window and not _force_browser and not _force_vscode:
        try:
            if _window_process is not None and _window_process.poll() is None:
                # Webview already open — inject new tab
                asyncio.run_coroutine_threadsafe(
                    _notify_shells(session.sid, name), SERVER_LOOP
                )
            else:
                _window_process = _open_webview_with_fallback(url_shell, win_w, win_h)
        except Exception:
            _open_browser(url_viewer, force_vscode=_force_vscode)
    else:
        if (
            window
            and not can_native_window
            and not _force_browser
            and not _force_vscode
        ):
            print(
                "[ArrayView] Native window unavailable; opening browser fallback",
                flush=True,
            )
        _open_browser(url_viewer, force_vscode=_force_vscode)

    _print_viewer_location(url_viewer)
    return url_viewer


def _is_script_mode() -> bool:
    """True when running as a plain Python script (not interactive REPL, not Jupyter, not Julia)."""
    if _in_jupyter() or _is_julia_env():
        return False
    if sys.flags.interactive or hasattr(sys, "ps1"):
        return False
    return True


async def _stop_server_when_viewer_closes(
    server, connect_timeout: float = 20.0, grace_seconds: float = 1.0
) -> None:
    """Asyncio task: signal uvicorn to stop once the viewer window is fully closed.
    Used in script mode so the non-daemon server thread exits cleanly when done."""
    deadline = time.monotonic() + connect_timeout
    while VIEWER_SOCKETS == 0:
        if time.monotonic() > deadline:
            server.should_exit = True  # no viewer connected; give up
            return
        await asyncio.sleep(0.2)
    # At least one viewer connected — now wait for all to disconnect.
    while True:
        while VIEWER_SOCKETS > 0:
            await asyncio.sleep(0.2)
        # Grace period (lets page refreshes reconnect before we shut down).
        frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        fi = 0
        sys.stderr.write("\n")  # start on a fresh line, never paste onto shell prompt
        sys.stderr.flush()
        deadline = time.monotonic() + grace_seconds
        while time.monotonic() < deadline:
            if VIEWER_SOCKETS > 0:
                sys.stderr.write("\r" + " " * 60 + "\r")
                sys.stderr.flush()
                break
            sys.stderr.write(
                f"\r\033[33m{frames[fi % len(frames)]} Shutting down...\033[0m"
            )
            sys.stderr.flush()
            fi += 1
            await asyncio.sleep(0.08)
        else:
            sys.stderr.write("\r" + " " * 60 + "\r")
            sys.stderr.flush()
            print("[ArrayView] Server stopped.", flush=True)
            server.should_exit = True
            return


def _wait_for_viewer_close(grace_seconds: float = 1.0) -> None:
    """Block until all viewer WebSocket connections close.
    Waits for a viewer WebSocket to connect, then all to disconnect, then applies a
    brief grace period so page refreshes don't prematurely kill the server.
    Shows a short shutdown animation during the grace period.
    """
    while VIEWER_SOCKETS == 0:
        time.sleep(0.2)
    while True:
        while VIEWER_SOCKETS > 0:
            time.sleep(0.2)
        # All sockets gone — grace period for page refresh / reconnect
        frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        fi = 0
        sys.stderr.write("\n")  # start on a fresh line, never paste onto shell prompt
        sys.stderr.flush()
        deadline = time.monotonic() + grace_seconds
        while time.monotonic() < deadline:
            if VIEWER_SOCKETS > 0:
                sys.stderr.write("\r" + " " * 60 + "\r")
                sys.stderr.flush()
                break  # reconnected; wait again
            sys.stderr.write(
                f"\r\033[33m{frames[fi % len(frames)]} Shutting down...\033[0m"
            )
            sys.stderr.flush()
            fi += 1
            time.sleep(0.08)
        else:
            sys.stderr.write("\r" + " " * 60 + "\r")
            sys.stderr.flush()
            return  # deadline passed with no reconnect → really closed


def _is_julia_env() -> bool:
    """True when running inside Julia via PythonCall/PyCall.
    In this environment the GIL is not released between Julia statements, so
    daemon threads (uvicorn) cannot serve HTTP requests once view() returns.
    We detect it by checking for juliacall/julia markers in loaded modules or
    the executable path.
    """
    if any("juliacall" in k.lower() or "julia" in k.lower() for k in sys.modules):
        return True
    exe = sys.executable.lower()
    return "julia" in exe


_julia_jupyter_cache: bool | None = None


def _in_julia_jupyter() -> bool:
    """True when running in Julia via PythonCall inside an IJulia Jupyter kernel.

    In IJulia, Julia's stdout is redirected to an IJulia stream object; its type
    name contains "IJulia". In a plain terminal, stdout is a Base.TTY.
    Fallback: try accessing Main.IJulia (present when `using IJulia` was called).
    """
    global _julia_jupyter_cache
    if _julia_jupyter_cache is not None:
        return _julia_jupyter_cache
    try:
        import juliacall as _jl

        r = _jl.Main.seval('occursin("IJulia", string(typeof(stdout)))')
        if str(r).strip().lower() == "true":
            _julia_jupyter_cache = True
            return True
        r2 = _jl.Main.seval("try; Main.IJulia; true; catch; false; end")
        _julia_jupyter_cache = str(r2).strip().lower() == "true"
    except Exception:
        _julia_jupyter_cache = False
    return _julia_jupyter_cache


def _view_julia(
    data: np.ndarray,
    name: str,
    port: int,
    window: bool,
    inline: bool = False,
    height: int = 500,
):
    """Julia-specific view() path: run the server in a subprocess so it is
    completely independent of Julia's GIL.
    """
    return _view_subprocess(data, name, port, window)


def _view_subprocess(
    data: np.ndarray,
    name: str,
    port: int,
    window: bool,
    inline: bool = False,
    height: int = 500,
    rgb: bool = False,
) -> str:
    """Run the viewer in a separate subprocess server.

    Used when the calling process may exit shortly after view() returns
    (Julia, VS Code tunnel one-shot scripts, CLI).  The subprocess server
    survives because it is not a daemon thread.
    """
    import tempfile

    # Persist the array to a temp file so the server subprocess can load it.
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tmp:
        tmp_path = tmp.name
    np.save(tmp_path, data)

    if _server_alive(port):
        # Existing subprocess server — register the new array via /load.
        try:
            body = json.dumps({"filepath": tmp_path, "name": name, "rgb": rgb}).encode()
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/load",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=300) as resp:
                result = json.loads(resp.read())
            if "error" in result:
                raise RuntimeError(result["error"])
            # Data is now in server memory; temp file no longer needed.
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
            sid = result["sid"]
        except Exception as e:
            print(
                f"[ArrayView] Failed to register with existing server: {e}", flush=True
            )
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
            raise
    else:
        if _port_in_use(port):
            raise RuntimeError(
                f"Port {port} is already in use by another process. "
                f"Choose a different port in view(..., port=...)."
            )
        sid = uuid.uuid4().hex
        # Spawn a self-contained server subprocess (same as CLI path).
        script = (
            f"from arrayview._app import _serve_daemon;"
            f"_serve_daemon({repr(tmp_path)}, {port}, {repr(sid)}, "
            f"name={repr(name)}, cleanup=True, rgb={rgb})"
        )
        subprocess.Popen(
            [sys.executable, "-c", script],
        )
        if not _wait_for_port(port, timeout=15.0):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
            raise RuntimeError(f"ArrayView server failed to start on port {port}.")

    url_viewer = f"http://localhost:{port}/?sid={sid}"
    encoded_name = urllib.parse.quote(name)
    url_shell = f"http://localhost:{port}/shell?init_sid={sid}&init_name={encoded_name}"
    _print_viewer_location(url_viewer)

    if inline:
        iframe_html = (
            f"<iframe src='{url_viewer}' width='100%'"
            f" height='{height}' frameborder='0'></iframe>"
        )
        # IJulia kernel: push HTML through Julia's display stack (routes to Jupyter
        # frontend). Must be a side-effect call, not a return value, because
        # PythonCall would convert a Python IFrame object to an opaque Julia value.
        try:
            import juliacall as _jl

            _jl.Main.seval(f'display("text/html", "{iframe_html}")')
            return None
        except Exception:
            pass
        # Plain Python Jupyter kernel fallback.
        try:
            from IPython.display import HTML, display as _ipy_display

            _ipy_display(HTML(iframe_html))
        except Exception:
            pass
        return url_viewer

    can_native = _can_native_window()
    if window and can_native:
        if not _open_webview_cli(url_shell, 1400, 900):
            print("[ArrayView] Falling back to browser", flush=True)
            _open_browser(url_viewer)
    else:
        _open_browser(url_viewer, force_vscode=False)
    return url_viewer


def _serve_empty(port: int) -> None:
    """Background server process with no sessions. Runs until killed.

    Used for ``arrayview --serve`` (pre-warm) and on remote tunnel sessions so
    the port stays alive across multiple tab opens/closes without requiring the
    user to re-run ``--serve`` or re-set port visibility.
    """
    threading.Thread(
        target=lambda: uvicorn.run(
            app, host="127.0.0.1", port=port, log_level="error", timeout_keep_alive=30
        ),
        daemon=True,
    ).start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    os._exit(0)


def _serve_daemon(
    filepath: str,
    port: int,
    sid: str,
    name: str = None,
    cleanup: bool = False,
    overlay_filepath: str = None,
    overlay_sid: str = None,
    compare_filepath: str = None,
    compare_sid: str = None,
    vfield_filepath: str = None,
    persist: bool = False,
    rgb: bool = False,
) -> None:
    """Background server process. Loads data, serves it.
    persist=True: never exits (used on remote tunnel so port stays alive).
    persist=False: exits when the UI closes (default, used locally).
    cleanup=True: delete filepath after loading (used when it is a temp file).
    """
    # Register sid as pending so /metadata can poll while data loads.
    PENDING_SESSIONS.add(sid)

    # Start uvicorn immediately — the window can open before data is ready.
    threading.Thread(
        target=lambda: uvicorn.run(
            app, host="127.0.0.1", port=port, log_level="error", timeout_keep_alive=30
        ),
        daemon=True,
    ).start()

    def _load():
        try:
            data = load_data(filepath)
            if cleanup:
                try:
                    os.unlink(filepath)
                except Exception:
                    pass
            session = Session(data, filepath=None if cleanup else filepath, name=name)
            session.sid = sid
            if rgb:
                _setup_rgb(session)
            if vfield_filepath:
                try:
                    vf_data = load_data(vfield_filepath)
                    session.vfield = vf_data
                    print(
                        f"[ArrayView] Loaded vector field {vfield_filepath} shape {vf_data.shape}",
                        flush=True,
                    )
                except Exception as e:
                    print(
                        f"[ArrayView] Warning: failed to load vector field {vfield_filepath}: {e}",
                        flush=True,
                    )
            SESSIONS[session.sid] = session
            if overlay_filepath and overlay_sid:
                try:
                    ov_data = load_data(overlay_filepath)
                    ov_session = Session(
                        ov_data, filepath=overlay_filepath, name="overlay"
                    )
                    ov_session.sid = overlay_sid
                    SESSIONS[overlay_sid] = ov_session
                except Exception as e:
                    print(
                        f"[ArrayView] Warning: failed to load overlay {overlay_filepath}: {e}",
                        flush=True,
                    )
            if compare_filepath and compare_sid:
                try:
                    cmp_data = load_data(compare_filepath)
                    cmp_name = os.path.basename(compare_filepath) or "compare"
                    cmp_session = Session(
                        cmp_data, filepath=compare_filepath, name=cmp_name
                    )
                    cmp_session.sid = compare_sid
                    SESSIONS[compare_sid] = cmp_session
                except Exception as e:
                    print(
                        f"[ArrayView] Warning: failed to load compare array {compare_filepath}: {e}",
                        flush=True,
                    )
        finally:
            PENDING_SESSIONS.discard(sid)

    threading.Thread(target=_load, daemon=True).start()

    if persist:
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
    else:
        _wait_for_viewer_close()
    os._exit(0)


def _make_demo_array() -> "np.ndarray":
    """Return a (128, 128, 32, 3) float32 RGB plasma animation.

    Dims: 128×128 canvas, 32 animation frames, 3 colour channels (RGB).
    Each frame is a smooth plasma / interference pattern built from
    overlapping sine waves — colourful and visually rich at every zoom level.
    Values are in [0, 255] so the array is ready for direct RGB viewing.
    """
    import numpy as np

    H, W, T = 128, 128, 32
    arr = np.zeros((H, W, T, 3), dtype=np.float32)

    # Spatial grids (normalised to [0, 1])
    xs = np.linspace(0.0, 1.0, W, dtype=np.float32)
    ys = np.linspace(0.0, 1.0, H, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys)  # shape (H, W)

    for ti in range(T):
        ph = 2.0 * np.pi * ti / T  # animation phase

        # Three overlapping plasma waves with distinct spatial frequencies
        p0 = np.sin(6.0 * np.pi * X + ph) + np.sin(6.0 * np.pi * Y + ph * 0.7)
        p1 = np.sin(9.0 * np.pi * (X + Y) + ph * 1.3) + np.sin(
            np.pi * (np.sqrt((X - 0.5) ** 2 + (Y - 0.5) ** 2) * 12.0 - ph)
        )
        p2 = np.sin(7.0 * np.pi * X * Y + ph * 0.5)

        # Each channel driven by a different mix → distinct hue cycling
        r = 0.5 + 0.5 * np.sin(p0 * 1.0 + ph * 0.0)
        g = 0.5 + 0.5 * np.sin(p1 * 1.0 + ph * 0.33)
        b = 0.5 + 0.5 * np.sin(p2 * 1.0 + ph * 0.67)

        arr[:, :, ti, 0] = (r * 255.0).astype(np.float32)
        arr[:, :, ti, 1] = (g * 255.0).astype(np.float32)
        arr[:, :, ti, 2] = (b * 255.0).astype(np.float32)

    return arr


def arrayview():
    """Command Line Interface Entry Point."""
    parser = argparse.ArgumentParser(description="Lightning Fast ND Array Viewer")
    parser.add_argument(
        "files",
        nargs="*",
        metavar="FILE",
        help=(
            "Array paths. First path is the base array; optional additional paths "
            "are preloaded for compare mode (up to 6 total files)."
        ),
    )
    parser.add_argument("--port", type=int, default=8000, help="Port to serve on")
    parser.add_argument(
        "--serve",
        action="store_true",
        help=(
            "Start a persistent server on the given port without loading any data. "
            "Useful on VS Code remote tunnel: run this first, set the port to Public "
            "in the Ports tab, then use 'arrayview FILE' freely."
        ),
    )
    parser.add_argument(
        "--window",
        choices=["browser", "vscode", "native"],
        default=None,
        help="How to open the viewer: browser (system browser), vscode (SimpleBrowser), or native window",
    )
    parser.add_argument(
        "--browser",
        action="store_true",
        help="Deprecated: use --window browser",
    )
    parser.add_argument(
        "--kill",
        action="store_true",
        help="Kill the ArrayView server running on --port (default 8000) and exit",
    )
    parser.add_argument(
        "--overlay",
        metavar="FILE",
        help="Segmentation mask to overlay (binary 0/1 array, same spatial shape)",
    )
    parser.add_argument(
        "--compare",
        metavar="FILE",
        help="Deprecated: second array for side-by-side compare mode",
    )
    parser.add_argument(
        "--vectorfield",
        metavar="FILE",
        help=(
            "Deformation vector field to overlay as arrows. "
            "Must have the same spatial shape as the image plus a trailing dimension of size 3 "
            "(displacements along each spatial axis)."
        ),
    )
    parser.add_argument(
        "--rgb",
        action="store_true",
        help=(
            "Interpret the array as an RGB or RGBA image. "
            "The first or last dimension must have size 3 or 4."
        ),
    )
    args = parser.parse_args()
    if not args.serve and not args.kill and not args.files:
        # No files given: launch the animated pixel-art demo
        import tempfile as _tempfile
        import numpy as _np_demo

        _demo_arr = _make_demo_array()
        _fd, _tmp_path = _tempfile.mkstemp(suffix=".npy")
        import os as _os_demo

        _os_demo.close(_fd)
        _np_demo.save(_tmp_path, _demo_arr)
        args.files = [_tmp_path]
        args._demo_name = "welcome"
        args._demo_cleanup = True
        args.rgb = True
    if args.files and len(args.files) > 6:
        parser.error(
            "At most six FILE arguments are supported; concat arrays first for larger compare sets."
        )
    if args.compare and len(args.files) > 1:
        parser.error("Use either positional compare files or --compare, not both.")

    # ── --kill: stop the server on the given port ──────────────────────────
    if args.kill:
        import signal as _signal

        if sys.platform == "win32":
            result = subprocess.run(
                ["netstat", "-ano", "-p", "TCP"], capture_output=True, text=True
            )
            killed = False
            for line in result.stdout.splitlines():
                if f":{args.port}" in line and "LISTENING" in line:
                    parts = line.split()
                    try:
                        pid = int(parts[-1])
                        subprocess.run(["taskkill", "/F", "/PID", str(pid)], check=True)
                        print(f"[ArrayView] Killed process {pid} on port {args.port}")
                        killed = True
                    except Exception as e:
                        print(f"[ArrayView] Failed to kill process: {e}")
            if not killed:
                print(f"[ArrayView] No process found listening on port {args.port}")
        else:
            result = subprocess.run(
                ["lsof", "-ti", f"tcp:{args.port}", "-sTCP:LISTEN"],
                capture_output=True,
                text=True,
            )
            pids = [
                int(p) for p in result.stdout.strip().split() if p.strip().isdigit()
            ]
            if not pids:
                print(f"[ArrayView] No process found listening on port {args.port}")
            else:
                for pid in pids:
                    try:
                        os.kill(pid, _signal.SIGTERM)
                        print(f"[ArrayView] Killed process {pid} on port {args.port}")
                    except ProcessLookupError:
                        pass
        return

    # ── --serve: start a persistent empty server and exit ──────────────────
    if args.serve:
        if _server_alive(args.port):
            print(
                f"[ArrayView] Server already running on port {args.port}. "
                "Set port to Public in VS Code Ports tab if not done yet, "
                "then run: arrayview your_file.npy"
            )
            return
        if _port_in_use(args.port):
            print(
                f"Error: port {args.port} is in use by another process. "
                "Use --port to pick another."
            )
            sys.exit(1)
        script = f"from arrayview._app import _serve_empty; _serve_empty({args.port})"
        proc = subprocess.Popen([sys.executable, "-c", script])
        if not _wait_for_port(args.port, timeout=15.0):
            print(f"Error: ArrayView server failed to start on port {args.port}.")
            sys.exit(1)
        print(
            f"\n  \033[1;36m→ ArrayView server started on port {args.port} (PID {proc.pid})\033[0m\n"
            f"\n  Remote tunnel setup:\n"
            f"    1. VS Code Ports tab → port {args.port} → right-click → Port Visibility → Public\n"
            f"    2. Then run: arrayview your_file.npy\n"
            f"\n  Server stays running until you kill it (kill {proc.pid}).\n"
        )
        return

    base_file = os.path.abspath(args.files[0])
    compare_files = [os.path.abspath(p) for p in args.files[1:]]
    if args.compare:
        compare_files.append(os.path.abspath(args.compare))

    try:
        data = load_data(base_file)
        try:
            size_str = f" ({data.nbytes // 1024**2} MB)"
        except AttributeError:
            size_str = ""
        print(f"Loaded {base_file} with shape {data.shape}{size_str}")
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

    name = getattr(args, "_demo_name", None) or os.path.basename(base_file)

    is_arrayview_server = _server_alive(args.port)
    if _port_in_use(args.port) and not is_arrayview_server:
        print(
            f"Error: port {args.port} is in use by another process. "
            "Use --port to pick another."
        )
        sys.exit(1)

    # Resolve --window / --browser into a single window_mode
    if args.browser and not args.window:
        args.window = "browser"
    window_mode = args.window  # None = auto-detect (current behaviour)
    if window_mode == "native" and _is_vscode_remote():
        print(
            "[ArrayView] --window native is not supported on remote tunnel; using vscode instead."
        )
        window_mode = "vscode"
    use_webview = (window_mode == "native") or (
        window_mode is None and _can_native_window()
    )

    if is_arrayview_server:
        # Server already running — register the new array.
        # If using webview, notify the existing shell to inject a new tab.
        try:
            # Register overlay first (no notification) to get overlay_sid
            overlay_sid = None
            if args.overlay:
                ov_body = json.dumps(
                    {
                        "filepath": os.path.abspath(args.overlay),
                        "name": "overlay",
                        "notify": False,
                    }
                ).encode()
                ov_req = urllib.request.Request(
                    f"http://127.0.0.1:{args.port}/load",
                    data=ov_body,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(ov_req, timeout=300) as resp:
                    ov_result = json.loads(resp.read())
                if "error" in ov_result:
                    print(
                        f"Error from server while loading overlay: {ov_result['error']}"
                    )
                    sys.exit(1)
                overlay_sid = ov_result.get("sid")

            compare_sids = []
            for compare_file in compare_files:
                cmp_body = json.dumps(
                    {
                        "filepath": compare_file,
                        "name": os.path.basename(compare_file),
                        "notify": False,
                    }
                ).encode()
                cmp_req = urllib.request.Request(
                    f"http://127.0.0.1:{args.port}/load",
                    data=cmp_body,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(cmp_req, timeout=300) as resp:
                    cmp_result = json.loads(resp.read())
                if "error" in cmp_result:
                    print(
                        f"Error from server while loading compare array: {cmp_result['error']}"
                    )
                    sys.exit(1)
                compare_sid = cmp_result.get("sid")
                if compare_sid:
                    compare_sids.append(compare_sid)

            notify_webview = use_webview and overlay_sid is None
            body_dict = {
                "filepath": base_file,
                "name": name,
                "notify": notify_webview,
                "rgb": args.rgb,
            }
            if notify_webview and compare_sids:
                body_dict["compare_sid"] = compare_sids[0]
                body_dict["compare_sids"] = ",".join(compare_sids)
            body = json.dumps(body_dict).encode()
            req = urllib.request.Request(
                f"http://127.0.0.1:{args.port}/load",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=300) as resp:
                result = json.loads(resp.read())
            if "error" in result:
                print(f"Error from server: {result['error']}")
                sys.exit(1)

            # Attach vector field to the newly loaded session
            if args.vectorfield:
                vf_body = json.dumps(
                    {
                        "sid": result["sid"],
                        "filepath": os.path.abspath(args.vectorfield),
                    }
                ).encode()
                vf_req = urllib.request.Request(
                    f"http://127.0.0.1:{args.port}/attach_vectorfield",
                    data=vf_body,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(vf_req, timeout=300) as resp:
                    vf_result = json.loads(resp.read())
                if "error" in vf_result:
                    print(
                        f"Warning: failed to attach vector field: {vf_result['error']}"
                    )
        except Exception as e:
            print(
                f"Error: port {args.port} is in use by another process. "
                f"Use --port to pick another. ({e})"
            )
            sys.exit(1)

        sid = result["sid"]
        encoded_name_inject = urllib.parse.quote(name)
        qs = f"?sid={sid}"
        if overlay_sid:
            qs += f"&overlay_sid={overlay_sid}"
        if compare_sids:
            qs += f"&compare_sid={compare_sids[0]}"
            qs += f"&compare_sids={','.join(compare_sids)}"
        if notify_webview and result.get("notified"):
            # Tab was injected into existing webview window (with or without compare)
            print(f"Injected into existing window (port {args.port})")
        elif notify_webview and not result.get("notified"):
            # Native window was requested but the shell is gone — open a new native window.
            init_qs = f"init_sid={sid}&init_name={encoded_name_inject}"
            if compare_sids:
                init_qs += (
                    f"&init_compare_sid={compare_sids[0]}"
                    f"&init_compare_sids={','.join(compare_sids)}"
                )
            url_shell = f"http://localhost:{args.port}/shell?{init_qs}"
            if not _open_webview_cli(url_shell, 1200, 800):
                print("[ArrayView] Falling back to browser", flush=True)
                url = f"http://localhost:{args.port}/{qs}"
                _print_viewer_location(url)
                _open_browser(url, blocking=True)
        else:
            url = f"http://localhost:{args.port}/{qs}"
            print(f"Open {url} in your browser")
            _open_browser(url, blocking=True, force_vscode=(window_mode == "vscode"))
        return

    sid = uuid.uuid4().hex
    overlay_sid = uuid.uuid4().hex if args.overlay else None
    encoded_name = urllib.parse.quote(name)

    # Configure VS Code port settings before starting the server.
    if not use_webview:
        _configure_vscode_port_preview(args.port)

    # Spawn background server.
    # On remote tunnel: persist=True so the server survives tab closes and the
    # port stays public across multiple arrayview invocations.
    # Locally: persist=False so the port is freed when the last tab closes.
    is_remote = _is_vscode_remote()
    vfield_abs = os.path.abspath(args.vectorfield) if args.vectorfield else None
    demo_name = getattr(args, "_demo_name", None)
    demo_cleanup = getattr(args, "_demo_cleanup", False)
    script = (
        f"from arrayview._app import _serve_daemon;"
        f"_serve_daemon("
        f"{repr(base_file)}, {args.port}, {repr(sid)},"
        f" name={repr(demo_name)},"
        f" cleanup={demo_cleanup},"
        f" overlay_filepath={repr(os.path.abspath(args.overlay) if args.overlay else None)},"
        f" overlay_sid={repr(overlay_sid)},"
        f" vfield_filepath={repr(vfield_abs)},"
        f" persist={is_remote},"
        f" rgb={args.rgb},"
        f")"
    )
    subprocess.Popen(
        [sys.executable, "-c", script],
    )
    if not _wait_for_port(args.port, timeout=15.0):
        print(
            f"Error: ArrayView server failed to start on port {args.port}. "
            "Use --port to pick another."
        )
        sys.exit(1)

    compare_sids = []
    for compare_file in compare_files:
        try:
            cmp_body = json.dumps(
                {
                    "filepath": compare_file,
                    "name": os.path.basename(compare_file),
                    "notify": False,
                }
            ).encode()
            cmp_req = urllib.request.Request(
                f"http://127.0.0.1:{args.port}/load",
                data=cmp_body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(cmp_req, timeout=300) as resp:
                cmp_result = json.loads(resp.read())
            if "error" in cmp_result:
                print(
                    f"Error from server while loading compare array: {cmp_result['error']}"
                )
                sys.exit(1)
            compare_sid = cmp_result.get("sid")
            if compare_sid:
                compare_sids.append(compare_sid)
        except Exception as e:
            print(f"Error while loading compare array {compare_file}: {e}")
            sys.exit(1)

    qs = f"?sid={sid}"
    if overlay_sid:
        qs += f"&overlay_sid={overlay_sid}"
    if compare_sids:
        qs += f"&compare_sid={compare_sids[0]}"
        qs += f"&compare_sids={','.join(compare_sids)}"

    if use_webview and overlay_sid is None:
        init_qs = f"init_sid={sid}&init_name={encoded_name}"
        if compare_sids:
            init_qs += (
                f"&init_compare_sid={compare_sids[0]}"
                f"&init_compare_sids={','.join(compare_sids)}"
            )
        url_shell = f"http://localhost:{args.port}/shell?{init_qs}"
        if not _open_webview_cli(url_shell, 1400, 900):
            print("[ArrayView] Falling back to browser", flush=True)
            url = f"http://localhost:{args.port}/{qs}"
            _print_viewer_location(url)
            _open_browser(url, blocking=False, force_vscode=(window_mode == "vscode"))
    else:
        if use_webview and overlay_sid:
            print(
                "[ArrayView] Overlay mode: opening browser (webview injection not supported with overlay)",
                flush=True,
            )
        url = f"http://localhost:{args.port}/{qs}"
        _print_viewer_location(url)
        _open_browser(url, blocking=True, force_vscode=(window_mode == "vscode"))
