import argparse
import asyncio
import io
import json
import os
import queue as _queue
import socket
import sys
import time
import threading
import subprocess
import uuid
import urllib.parse
import urllib.request
from collections import OrderedDict
from importlib.resources import files as _pkg_files

import numpy as np
import nibabel as nib
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response, WebSocket
from fastapi.responses import HTMLResponse, RedirectResponse
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

        # Phase 2: process is alive — wait up to 8 s for a viewer WebSocket to connect
        for _ in range(80):
            time.sleep(0.1)
            if VIEWER_SOCKETS > 0:
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

        self.raw_cache = OrderedDict()
        self.rgba_cache = OrderedDict()
        self.mosaic_cache = OrderedDict()

        self.RAW_CACHE_BYTES = 512 * 1024 * 1024  # 512 MB
        self.RGBA_CACHE_BYTES = 1024 * 1024 * 1024  # 1 GB
        self.MOSAIC_CACHE_BYTES = 256 * 1024 * 1024  # 256 MB
        self._raw_bytes = self._rgba_bytes = self._mosaic_bytes = 0

        self.preload_gen = 0
        self.preload_done = 0
        self.preload_total = 0
        self.preload_skipped = False
        self.preload_lock = threading.Lock()

        self.mask_level = 0  # 0=off, 1=Otsu, 2=2×Otsu
        self.mask_otsu = None  # cached Otsu threshold (float)
        self.mask_threshold = 0.0  # active threshold applied to rendering

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

COLORMAPS = ["gray", "lipari", "navia", "viridis", "plasma", "RdBu_r"]
DR_PERCENTILES = [(0, 100), (1, 99), (5, 95), (10, 90)]
DR_LABELS = ["0-100%", "1-99%", "5-95%", "10-90%"]

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
    mu_b = np.where(w_b > 0, mu_cum / (w_b * total), 0.0)
    mu_f = np.where(w_f > 0, (mu_cum[-1] / total - mu_b * w_b) / w_f, 0.0)
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
    # Exactly-zero or masked raw values → transparent so the canvas background shows through
    abs_raw = np.abs(raw)
    mask_thr = getattr(session, "mask_threshold", 0.0)
    transparent = (abs_raw == 0) if mask_thr <= 0 else (abs_raw < mask_thr)
    if transparent.any():
        rgba = rgba.copy()
        rgba[transparent, 3] = 0
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
    out_rgb = (ov_a * ov_rgb + (1.0 - ov_a) * base_a[:, None] * base_rgb) / denom[:, None]

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
    n = session.shape[slice_dim]
    H = session.shape[dim_y]
    W = session.shape[dim_x]
    if dim_z >= 0:
        nz = session.shape[dim_z]
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
            render_rgba(
                session, dim_x, dim_y, tuple(idx), colormap, dr, complex_mode, log_scale
            )
        with session.preload_lock:
            session.preload_done = i + 1
        time.sleep(0.005)


# ---------------------------------------------------------------------------
# Shell WebSocket for Webview Tab Management
# ---------------------------------------------------------------------------
async def _notify_shells(sid, name):
    """Push a new-tab message to all connected webview shell windows."""
    for _ in range(200):  # Wait up to 2 s for window to connect
        if SHELL_SOCKETS:
            break
        await asyncio.sleep(0.01)
    for ws in SHELL_SOCKETS.copy():
        try:
            await ws.send_json({"action": "new_tab", "sid": sid, "name": name})
        except Exception:
            pass


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

            if dim_z >= 0:
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
            if dim_z < 0:
                overlay_sid = msg.get("overlay_sid")
                mask = _extract_overlay_mask(
                    overlay_sid, dim_x, dim_y, idx_tuple, expected_shape=(h, w)
                )
                rgba = _composite_overlay_mask(rgba, mask)

            header = np.array([seq, w, h], dtype=np.uint32).tobytes()
            vminmax = np.array([vmin, vmax], dtype=np.float32).tobytes()
            await ws.send_bytes(header + vminmax + rgba.tobytes())
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


@app.get("/metadata/{sid}")
def get_metadata(sid: str):
    session = SESSIONS.get(sid)
    if not session:
        return Response(status_code=404)
    try:
        return {
            "shape": [int(s) for s in session.shape],
            "is_complex": bool(np.iscomplexobj(session.data)),
            "name": session.name,
        }
    except Exception as e:
        import traceback

        traceback.print_exc()
        return Response(
            status_code=500, content=str(e).encode(), media_type="text/plain"
        )


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

    idx_tuple = tuple(int(x) for x in indices.split(","))
    raw = extract_slice(session, dim_x, dim_y, list(idx_tuple))
    data = apply_complex_mode(raw, complex_mode)
    h, w = data.shape
    if 0 <= py < h and 0 <= px < w:
        val = float(data[py, px])
    else:
        val = float("nan")
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
    idx_tuple = tuple(int(v) for v in indices.split(","))
    raw = extract_slice(session, dim_x, dim_y, list(idx_tuple))
    data = apply_complex_mode(raw, complex_mode)
    h, w = data.shape
    ys, xs = np.ogrid[:h, :w]
    mask = (xs - cx) ** 2 + (ys - cy) ** 2 <= r**2
    roi = data[mask]
    if roi.size == 0:
        return {"error": "empty selection"}
    return {
        "min": float(roi.min()),
        "max": float(roi.max()),
        "mean": float(roi.mean()),
        "std": float(roi.std()),
        "n": int(roi.size),
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
    return {
        "min": float(roi.min()),
        "max": float(roi.max()),
        "mean": float(roi.mean()),
        "std": float(roi.std()),
        "n": int(roi.size),
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
    idx_list = [int(x) for x in indices.split(",")]
    n = session.shape[slice_dim]
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
    idx_list = [int(x) for x in indices.split(",")]
    n = session.shape[slice_dim]
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


@app.get("/sessions")
def get_sessions():
    """Returns list of active sessions (used by shell to populate tabs on load)."""
    return [{"sid": s.sid, "name": s.name} for s in SESSIONS.values()]


@app.post("/load")
async def load_file(request: Request):
    """Load a file into a new session. Optionally notify webview shells."""
    body = await request.json()
    filepath = str(body["filepath"])
    name = str(body.get("name") or os.path.basename(filepath))
    notify = bool(body.get("notify", False))
    try:
        data = load_data(filepath)
    except Exception as e:
        return {"error": str(e)}
    session = Session(data, filepath=filepath, name=name)
    SESSIONS[session.sid] = session
    if notify:
        await _notify_shells(session.sid, name)
    return {"sid": session.sid, "name": name}


@app.get("/")
def get_ui(sid: str = None):
    """Viewer page."""
    if not sid:
        # VS Code's onAutoForward opens the root URL without a session ID.
        # Redirect to the most recently created session if one exists.
        if SESSIONS:
            latest_sid = list(SESSIONS.keys())[-1]
            return RedirectResponse(url=f"/?sid={latest_sid}")
        return HTMLResponse(
            content="<html><body style='background:#111;color:#ccc;font-family:monospace;display:flex;align-items:center;justify-content:center;height:100vh;margin:0'>No session ID provided.</body></html>"
        )
    html = (
        _VIEWER_HTML_TEMPLATE.replace("__COLORMAPS__", str(COLORMAPS))
        .replace("__DR_LABELS__", str(DR_LABELS))
        .replace("__COLORMAP_GRADIENT_STOPS__", json.dumps(COLORMAP_GRADIENT_STOPS))
        .replace("__COMPLEX_MODES__", str(COMPLEX_MODES))
        .replace("__REAL_MODES__", str(REAL_MODES))
    )
    return HTMLResponse(content=html)


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
    if os.environ.get("VSCODE_IPC_HOOK_CLI"):
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

    dbg = f"[ArrayView] _find_vscode_ipc_hook: pid={os.getpid()}"
    # Own environment first
    val = os.environ.get("VSCODE_IPC_HOOK_CLI", "")
    dbg += f" own_env={val!r}"
    if val:
        exists = os.path.exists(val)
        dbg += f" exists={exists}"
        if exists:
            print(dbg + " → found in own env", flush=True)
            _VSCODE_IPC_HOOK_CACHE = val
            return val
    else:
        dbg += " TERM_PROGRAM=" + repr(os.environ.get("TERM_PROGRAM", ""))

    # Walk up to 12 ancestor processes
    pid = os.getpid()
    for i in range(12):
        pid = _ppid(pid)
        if pid <= 1:
            dbg += f" [stop at pid={pid}]"
            break
        val = _ipc_from_pid(pid)
        dbg += f" pid{i + 1}={pid}:ipc={val!r}"
        if val:
            exists = os.path.exists(val)
            dbg += f"(exists={exists})"
            if exists:
                print(dbg + " → found in ancestor", flush=True)
                _VSCODE_IPC_HOOK_CACHE = val
                return val

    print(dbg + " → NOT FOUND", flush=True)
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
_VSCODE_EXT_VERSION = "0.4.0"  # must match vscode-extension/package.json
_VSCODE_SIGNAL_FILENAME = "open-request-v0400.json"


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
                print(
                    f"[ArrayView] patched targetPlatform in {package_json}", flush=True
                )
        except Exception as exc:
            print(
                f"[ArrayView] could not patch extension metadata at {package_json}: {exc}",
                flush=True,
            )


def _ensure_vscode_extension() -> bool:
    """Install the bundled arrayview-opener VS Code extension for local VS Code use.

    Force-installs the current VSIX into the running extension host (hot-install).
    Hot-installing immediately activates the new version alongside any older
    version that may still be running. The extension and Python code also use a
    versioned signal filename so stale instances won't consume new requests.

    We do NOT uninstall first: an explicit uninstall causes VS Code to mark the
    extension and skip hot-activation on reinstall (see log.txt attempt 11).

    Remote/tunnel sessions deliberately avoid this path because smooth scrolling
    is better preserved by using the original websocket viewer in a real browser
    via VS Code port forwarding rather than the extension-host proxy viewer.
    """
    global _VSCODE_EXT_INSTALLED, _VSCODE_EXT_FRESH_INSTALL
    if _VSCODE_EXT_INSTALLED:
        return True

    code = _find_code_cli()
    if not code:
        print("[ArrayView] ext install: no 'code' CLI found", flush=True)
        return False

    env = dict(os.environ)
    ipc = _find_vscode_ipc_hook()
    if ipc and "VSCODE_IPC_HOOK_CLI" not in env:
        env["VSCODE_IPC_HOOK_CLI"] = ipc

    print(f"[ArrayView] ext install: code={code!r}", flush=True)

    vsix_path = str(_pkg_files(__package__).joinpath("arrayview-opener.vsix"))
    if not os.path.isfile(vsix_path):
        print(f"[ArrayView] ext vsix not found at {vsix_path}", flush=True)
        return False
    print(f"[ArrayView] ext installing from {vsix_path} ...", flush=True)
    try:
        r = subprocess.run(
            [code, "--install-extension", vsix_path, "--force"],
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
        )
        combined = (r.stdout or "") + (r.stderr or "")
        print(
            f"[ArrayView] ext install rc={r.returncode} out={combined.strip()!r}",
            flush=True,
        )
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
    except Exception as exc:
        print(f"[ArrayView] ext install exception: {exc}", flush=True)
    return False


def _configure_vscode_port_preview(port: int) -> bool:
    """Write port settings to .vscode/settings.json for the arrayview server.

    Remote/tunnel: marks the port as public and asks VS Code to open the port
    in Simple Browser automatically when it is detected.

    Local: removes stale keys written by older versions of arrayview.

    Returns True on success.
    """
    vscode_dir = os.path.join(os.getcwd(), ".vscode")
    settings_path = os.path.join(vscode_dir, "settings.json")
    try:
        os.makedirs(vscode_dir, exist_ok=True)
        settings: dict = {}
        if os.path.exists(settings_path):
            try:
                with open(settings_path) as f:
                    settings = json.load(f)
            except (json.JSONDecodeError, OSError):
                settings = {}
        attrs = settings.setdefault("remote.portsAttributes", {})
        if _is_vscode_remote():
            attrs[str(port)] = {
                "protocol": "http",
                "label": "ArrayView",
                "onAutoForward": "openPreview",
                "privacy": "public",
            }
            print(
                f"[ArrayView] port {port}: onAutoForward=openPreview privacy=public → {settings_path}",
                flush=True,
            )
        else:
            port_attrs = attrs.get(str(port), {})
            changed = False
            for key in ("onAutoForward", "privacy"):
                if key in port_attrs:
                    del port_attrs[key]
                    changed = True
            if changed:
                print(
                    f"[ArrayView] removed stale remote keys from port {port} settings",
                    flush=True,
                )
        with open(settings_path, "w") as f:
            json.dump(settings, f, indent=2)
        return True
    except Exception as exc:
        print(f"[ArrayView] could not write port settings: {exc}", flush=True)
        return False


def _open_via_signal_file(url: str, delay: float = 0.0) -> bool:
    """Write the URL to ~/.arrayview/open-request.json for the VS Code extension."""
    signal_dir = os.path.expanduser("~/.arrayview")
    signal_file = os.path.join(signal_dir, _VSCODE_SIGNAL_FILENAME)
    try:
        os.makedirs(signal_dir, exist_ok=True)
        if delay > 0:
            time.sleep(delay)
        with open(signal_file, "w") as f:
            json.dump({"url": url}, f)
        return True
    except Exception:
        return False


def _print_viewer_location(url: str) -> None:
    """Print a viewer location hint without offering a broken terminal link remotely."""
    if _is_vscode_remote():
        try:
            parsed = urllib.parse.urlparse(url)
            port = parsed.port or 8000
            sid = urllib.parse.parse_qs(parsed.query).get("sid", [None])[0]
        except Exception:
            port = 8000
            sid = None
        if sid:
            print(
                f"[ArrayView] remote viewer ready on port {port} (session {sid})",
                flush=True,
            )
        else:
            print(f"[ArrayView] remote viewer ready on port {port}", flush=True)
        print(
            "[ArrayView] VS Code should auto-open ArrayView in a Simple Browser tab.",
            flush=True,
        )
        print(
            "[ArrayView] do not cmd-click localhost from the tunnel terminal; let the Ports auto-forward/openPreview path handle it.",
            flush=True,
        )
        return
    print(f"[ArrayView] {url}", flush=True)


def _open_browser(url: str, blocking: bool = False) -> None:
    """Open *url* locally, or configure VS Code remote auto-preview behavior.

    Strategy (see log.txt for what was tried and why):
    1. Remote VS Code terminal:
       a. Configure the port as ``openPreview`` and ``public`` in
          ``remote.portsAttributes``.
       b. Let VS Code auto-open the forwarded port in a Simple Browser tab.
    2. Local VS Code terminal:
       a. Install the helper extension.
       b. Write the signal file so the extension opens Simple Browser locally.
    3. Fallback: open/xdg-open with the http URL (system browser).
    4. Always print the URL.
    """

    def _do() -> None:
        ipc = _find_vscode_ipc_hook()
        is_remote = _is_vscode_remote()
        opened = False

        print(
            f"[ArrayView] _open_browser: url={url!r} ipc={ipc!r} is_remote={is_remote}",
            flush=True,
        )

        try:
            parsed_port = int(url.split(":")[2].split("/")[0].split("?")[0])
        except Exception:
            parsed_port = 8000

        if is_remote:
            _configure_vscode_port_preview(parsed_port)
            print(
                "[ArrayView] remote VS Code session detected: relying on onAutoForward=openPreview.",
                flush=True,
            )
            print(
                "[ArrayView] port is forced public so the Simple Browser tab should open without the GitHub auth/postback 403.",
                flush=True,
            )
            return

        if ipc or _in_vscode_terminal():
            # Local VS Code terminal: install extension + signal file.
            _configure_vscode_port_preview(parsed_port)
            ext_ok = _ensure_vscode_extension()
            if ext_ok:
                if _VSCODE_EXT_FRESH_INSTALL:
                    print(
                        "[ArrayView] waiting for extension to activate...", flush=True
                    )
                    time.sleep(1.5)
                _open_via_signal_file(url)
                print(f"[ArrayView] wrote signal file for {url}", flush=True)
                opened = True

        if not opened and not is_remote:
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
    window: bool | None = None,
):
    """
    Launch the viewer. Does not block the main Python process.
    window defaults to False in Jupyter (inline IFrame) and True elsewhere.
    Each call opens a new viewer window/tab.
    Returns an IPython IFrame in inline mode, otherwise returns the viewer URL.
    """
    global _jupyter_server_port, _window_process, SERVER_LOOP  # _window_process is a Popen instance

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
        )

    session = Session(data, name=name)
    SESSIONS[session.sid] = session
    win_w, win_h = 1200, 800

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
    if window and can_native_window:
        try:
            if _window_process is not None and _window_process.poll() is None:
                # Webview already open — inject new tab
                asyncio.run_coroutine_threadsafe(
                    _notify_shells(session.sid, name), SERVER_LOOP
                )
            else:
                _window_process = _open_webview_with_fallback(url_shell, win_w, win_h)
        except Exception:
            _open_browser(url_viewer)
    else:
        if window and not can_native_window:
            print(
                "[ArrayView] Native window unavailable; opening browser fallback",
                flush=True,
            )
        _open_browser(url_viewer)

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


def _view_subprocess(data: np.ndarray, name: str, port: int, window: bool) -> str:
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
            body = json.dumps({"filepath": tmp_path, "name": name}).encode()
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/load",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
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
            f"name={repr(name)}, cleanup=True)"
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
        if not _open_webview_cli(url_shell, 1200, 800):
            print("[ArrayView] Falling back to browser", flush=True)
            _open_browser(url_viewer)
    else:
        _open_browser(url_viewer)
    return url_viewer


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
) -> None:
    """Background server process. Loads data, serves it, exits when the UI closes.
    cleanup=True: delete filepath after loading (used when it is a temp file).
    """
    data = load_data(filepath)
    if cleanup:
        try:
            os.unlink(filepath)
        except Exception:
            pass
    session = Session(data, filepath=None if cleanup else filepath, name=name)
    session.sid = sid
    SESSIONS[session.sid] = session
    if overlay_filepath and overlay_sid:
        try:
            ov_data = load_data(overlay_filepath)
            ov_session = Session(ov_data, filepath=overlay_filepath, name="overlay")
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
            cmp_session = Session(cmp_data, filepath=compare_filepath, name=cmp_name)
            cmp_session.sid = compare_sid
            SESSIONS[compare_sid] = cmp_session
        except Exception as e:
            print(
                f"[ArrayView] Warning: failed to load compare array {compare_filepath}: {e}",
                flush=True,
            )
    threading.Thread(
        target=lambda: uvicorn.run(
            app, host="127.0.0.1", port=port, log_level="error", timeout_keep_alive=30
        ),
        daemon=True,
    ).start()
    _wait_for_viewer_close()
    print(
        f"\033[32m[ArrayView] Server stopped. Port {port} is now available.\033[0m",
        flush=True,
    )
    os._exit(0)


def arrayview():
    """Command Line Interface Entry Point."""
    parser = argparse.ArgumentParser(description="Lightning Fast ND Array Viewer")
    parser.add_argument(
        "file",
        help="Path to array file (.npy, .npz, .nii/.nii.gz, .zarr, .pt/.pth, .h5/.hdf5, .tif/.tiff, .mat)",
    )
    parser.add_argument("--port", type=int, default=8000, help="Port to serve on")
    parser.add_argument(
        "--browser",
        action="store_true",
        help="Open in web browser instead of native window",
    )
    parser.add_argument(
        "--overlay",
        metavar="FILE",
        help="Segmentation mask to overlay (binary 0/1 array, same spatial shape)",
    )
    parser.add_argument(
        "--compare",
        metavar="FILE",
        help="Second array for side-by-side compare mode (must have same shape as main array)",
    )
    args = parser.parse_args()

    try:
        data = load_data(args.file)
        try:
            size_str = f" ({data.nbytes // 1024**2} MB)"
        except AttributeError:
            size_str = ""
        print(f"Loaded {args.file} with shape {data.shape}{size_str}")
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

    name = os.path.basename(args.file)

    is_arrayview_server = _server_alive(args.port)
    if _port_in_use(args.port) and not is_arrayview_server:
        print(
            f"Error: port {args.port} is in use by another process. "
            "Use --port to pick another."
        )
        sys.exit(1)

    use_webview = not args.browser and _can_native_window()

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
                with urllib.request.urlopen(ov_req, timeout=5) as resp:
                    ov_result = json.loads(resp.read())
                if "error" in ov_result:
                    print(f"Error from server while loading overlay: {ov_result['error']}")
                    sys.exit(1)
                overlay_sid = ov_result.get("sid")

            compare_sid = None
            if args.compare:
                cmp_body = json.dumps(
                    {
                        "filepath": os.path.abspath(args.compare),
                        "name": os.path.basename(args.compare),
                        "notify": False,
                    }
                ).encode()
                cmp_req = urllib.request.Request(
                    f"http://127.0.0.1:{args.port}/load",
                    data=cmp_body,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(cmp_req, timeout=5) as resp:
                    cmp_result = json.loads(resp.read())
                if "error" in cmp_result:
                    print(f"Error from server while loading compare array: {cmp_result['error']}")
                    sys.exit(1)
                compare_sid = cmp_result.get("sid")

            notify_webview = (
                use_webview and overlay_sid is None and compare_sid is None
            )  # webview inject skipped when overlay/compare used
            body = json.dumps(
                {
                    "filepath": os.path.abspath(args.file),
                    "name": name,
                    "notify": notify_webview,
                }
            ).encode()
            req = urllib.request.Request(
                f"http://127.0.0.1:{args.port}/load",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                result = json.loads(resp.read())
            if "error" in result:
                print(f"Error from server: {result['error']}")
                sys.exit(1)
        except Exception as e:
            print(
                f"Error: port {args.port} is in use by another process. "
                f"Use --port to pick another. ({e})"
            )
            sys.exit(1)

        sid = result["sid"]
        qs = f"?sid={sid}"
        if overlay_sid:
            qs += f"&overlay_sid={overlay_sid}"
        if compare_sid:
            qs += f"&compare_sid={compare_sid}"
        if use_webview and overlay_sid is None and compare_sid is None:
            # Tab was injected into existing webview window
            print(f"Injected into existing window (port {args.port})")
        else:
            url = f"http://localhost:{args.port}/{qs}"
            if use_webview and compare_sid:
                print(
                    "[ArrayView] Compare mode: opening browser (webview tab injection does not carry compare_sid)",
                    flush=True,
                )
            print(f"Open {url} in your browser")
            _open_browser(url, blocking=True)
        return

    sid = uuid.uuid4().hex
    overlay_sid = uuid.uuid4().hex if args.overlay else None
    compare_sid = uuid.uuid4().hex if args.compare else None
    encoded_name = urllib.parse.quote(name)

    # Configure .vscode/settings.json: in tunnel mode writes
    # onAutoForward=openPreview so VS Code auto-opens Simple Browser when it
    # detects the forwarded port.  Locally cleans up stale remote keys.
    if not use_webview:
        _configure_vscode_port_preview(args.port)

    # Spawn background server — exits automatically when the window/tab is closed
    script = (
        f"from arrayview._app import _serve_daemon;"
        f"_serve_daemon("
        f"{repr(os.path.abspath(args.file))}, {args.port}, {repr(sid)},"
        f" overlay_filepath={repr(os.path.abspath(args.overlay) if args.overlay else None)},"
        f" overlay_sid={repr(overlay_sid)},"
        f" compare_filepath={repr(os.path.abspath(args.compare) if args.compare else None)},"
        f" compare_sid={repr(compare_sid)}"
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

    qs = f"?sid={sid}"
    if overlay_sid:
        qs += f"&overlay_sid={overlay_sid}"
    if compare_sid:
        qs += f"&compare_sid={compare_sid}"

    if use_webview and overlay_sid is None and compare_sid is None:
        url_shell = f"http://localhost:{args.port}/shell?init_sid={sid}&init_name={encoded_name}"
        if not _open_webview_cli(url_shell, 1200, 800):
            print("[ArrayView] Falling back to browser", flush=True)
            url = f"http://localhost:{args.port}/{qs}"
            _print_viewer_location(url)
            _open_browser(url, blocking=False)
    else:
        if use_webview and overlay_sid:
            print(
                "[ArrayView] Overlay mode: opening browser (webview injection not supported with overlay)",
                flush=True,
            )
        if use_webview and compare_sid:
            print(
                "[ArrayView] Compare mode: opening browser (webview tab injection not supported with compare)",
                flush=True,
            )
        url = f"http://localhost:{args.port}/{qs}"
        _print_viewer_location(url)
        _open_browser(url, blocking=True)
