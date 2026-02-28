import argparse
import asyncio
import io
import json
import os
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
from fastapi import FastAPI, Request, Response, WebSocket
from fastapi.responses import HTMLResponse
from PIL import Image
from matplotlib import colormaps as mpl_colormaps
import qmricolors  # registers lipari, navia colormaps with matplotlib  # noqa: F401


# ---------------------------------------------------------------------------
# Subprocess GUI Launcher
# ---------------------------------------------------------------------------
def _open_webview(
    url: str, win_w: int, win_h: int, capture_stderr: bool = False
) -> subprocess.Popen:
    """Launch pywebview in a fresh subprocess. Uses subprocess.Popen to avoid
    multiprocessing bootstrap errors when called from a Jupyter kernel."""
    # Qt WebEngine renders at device-pixel-ratio scale, producing a thin
    # scrollbar at zoom=1.0.  A slightly lower zoom prevents it.
    script = (
        "import sys,webview;"
        "u,w,h=sys.argv[1],int(sys.argv[2]),int(sys.argv[3]);"
        "webview.create_window('ArrayView',u,width=w,height=h,background_color='#111111');"
        "webview.start(**({'gui':'qt'} if sys.platform.startswith('linux') else {}))"
    )
    return subprocess.Popen(
        [sys.executable, "-c", script, url, str(win_w), str(win_h)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE if capture_stderr else subprocess.DEVNULL,
    )


def _open_webview_with_fallback(url: str, win_w: int, win_h: int) -> subprocess.Popen:
    """Launch pywebview, falling back to _open_browser if the subprocess exits immediately
    OR if no shell WebSocket connects within ~10 s (catches macOS non-framework Python
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

        # Phase 2: process is alive — wait up to 8 s for a shell WebSocket to connect
        for _ in range(80):
            time.sleep(0.1)
            if SHELL_SOCKETS:
                print("[ArrayView] Native window connected successfully", flush=True)
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
SHELL_SOCKETS = []
_window_process = None
_PENDING_OPENS = 0  # incremented by /load, decremented when shell WS connects


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
                chunks = []
                for i in range(0, self.shape[0], step):
                    chunks.append(np.array(self.data[i]).ravel())
                    if sum(c.size for c in chunks) >= max_samples:
                        break
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
    if complex_mode == 0 and dr in session.global_stats:
        return session.global_stats[dr]
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


def _prepare_display(
    session, raw, complex_mode, dr, log_scale, vmin_override=None, vmax_override=None
):
    data = apply_complex_mode(raw, complex_mode)
    if vmin_override is not None and vmax_override is not None:
        return data, vmin_override, vmax_override
    if log_scale:
        data = np.log1p(np.abs(data)).astype(np.float32)
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
    return lut[(normalized * 255).astype(np.uint8)]


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
        key = (dim_x, dim_y, idx_tuple, colormap, dr, complex_mode, log_scale)
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
# Shell WebSocket for Tab Management
# ---------------------------------------------------------------------------
async def _notify_shells(sid, name):
    """Pushes a message to the open shell UI to dynamically spawn a new tab."""
    for _ in range(200):  # Wait up to 2 seconds if window just launched
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
    global _PENDING_OPENS
    await ws.accept()
    SHELL_SOCKETS.append(ws)
    if _PENDING_OPENS > 0:
        _PENDING_OPENS -= 1
    try:
        while True:
            msg = await ws.receive_json()
            if msg.get("action") == "close":
                sid = msg.get("sid")
                if sid in SESSIONS:
                    # Clear Memory instantly when tab closes
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
                rgba = await loop.run_in_executor(
                    None,
                    render_mosaic,
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
            else:
                rgba = await loop.run_in_executor(
                    None,
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

            header = np.array([seq, w, h], dtype=np.uint32).tobytes()
            vminmax = np.array([vmin, vmax], dtype=np.float32).tobytes()
            await ws.send_bytes(header + vminmax + rgba.tobytes())
    except Exception:
        pass


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
):
    session = SESSIONS.get(sid)
    if not session:
        return Response(status_code=404)
    idx_tuple = tuple(int(x) for x in indices.split(","))
    rgba = render_rgba(session, dim_x, dim_y, idx_tuple, colormap, dr)
    img = Image.fromarray(rgba[:, :, :3], mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return Response(
        content=buf.getvalue(),
        media_type="image/jpeg",
        headers={"Cache-Control": "max-age=300"},
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
    """Returns the Master Tabbed Window UI."""
    return HTMLResponse(content=_SHELL_HTML)


@app.get("/ping")
def ping():
    """Health marker so clients can verify this is an ArrayView server."""
    return {"ok": True, "service": "arrayview", "pid": os.getpid()}


@app.get("/sessions")
def get_sessions():
    """Returns list of active sessions (used by shell to auto-load on cold open)."""
    return [{"sid": s.sid, "name": s.name} for s in SESSIONS.values()]


@app.post("/load")
async def load_file(request: Request):
    """Load a file into a new session and push a new tab to all open shell windows."""
    body = await request.json()
    filepath = str(body["filepath"])
    name = str(body.get("name") or os.path.basename(filepath))
    try:
        data = load_data(filepath)
    except Exception as e:
        return {"error": str(e)}
    session = Session(data, filepath=filepath, name=name)
    global _PENDING_OPENS
    SESSIONS[session.sid] = session
    _PENDING_OPENS += 1
    await _notify_shells(session.sid, name)
    return {"sid": session.sid, "name": name}


@app.get("/")
def get_ui(sid: str = None):
    """Viewer iframe page. Redirects to /shell when no sid is given (e.g. VSCode popup)."""
    from fastapi.responses import RedirectResponse

    if not sid:
        return RedirectResponse(url="/shell")
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


def _is_vscode_remote() -> bool:
    """True when running inside a VS Code remote/tunnel session."""
    # VSCODE_IPC_HOOK_CLI is always present in VS Code tunnel/remote terminals.
    if os.environ.get("VSCODE_IPC_HOOK_CLI"):
        # On the local machine, VSCODE_IPC_HOOK_CLI is also set when VS Code
        # terminal is used, but we only treat it as "remote" when there's no
        # local desktop (i.e. not macOS / Windows native).
        if sys.platform not in ("darwin", "win32"):
            return True
        # macOS / Windows: only remote if SSH or explicit remote indicator.
        if os.environ.get("SSH_CONNECTION") or os.environ.get("SSH_CLIENT"):
            return True
    # TERM_PROGRAM=vscode + SSH_CONNECTION = VS Code SSH remote
    if os.environ.get("TERM_PROGRAM") == "vscode" and (
        os.environ.get("SSH_CONNECTION") or os.environ.get("SSH_CLIENT")
    ):
        return True
    return False


def _can_native_window() -> bool:
    """True if a pywebview native window can be opened."""
    # In a VS Code tunnel/remote there is no local display even if DISPLAY is set.
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


def _find_vscode_ipc_hook() -> str | None:
    """Return the value of VSCODE_IPC_HOOK_CLI, searching ancestor processes.

    uv run (and similar launchers) strip environment variables before executing
    Python.  Walking up the process tree lets us recover VSCODE_IPC_HOOK_CLI
    from the shell that originally invoked the command.
    """

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
        return val

    # Walk up to 12 ancestor processes
    pid = os.getpid()
    for _ in range(12):
        pid = _ppid(pid)
        if pid <= 1:
            break
        val = _ipc_from_pid(pid)
        if val and os.path.exists(val):
            return val

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
_VSCODE_EXT_VERSION = "0.0.2"  # must match vscode-extension/package.json


def _ensure_vscode_extension() -> bool:
    """Install the bundled arrayview-opener VS Code extension if not present
    or outdated.  Returns True if the extension is (now) installed.
    """
    global _VSCODE_EXT_INSTALLED
    if _VSCODE_EXT_INSTALLED:
        return True

    code = _find_code_cli()
    if not code:
        return False

    # Ensure the IPC hook is available so the remote-cli `code` helper can
    # communicate with VS Code (uv run and similar launchers strip env vars).
    env = dict(os.environ)
    ipc = _find_vscode_ipc_hook()
    if ipc:
        env["VSCODE_IPC_HOOK_CLI"] = ipc

    # Quick check: is it already installed with the right version?
    try:
        r = subprocess.run(
            [code, "--list-extensions", "--show-versions"],
            capture_output=True,
            text=True,
            timeout=10,
            env=env,
        )
        if r.returncode == 0:
            for line in r.stdout.splitlines():
                if line.startswith("arrayview.arrayview-opener@"):
                    installed_ver = line.split("@", 1)[1].strip()
                    if installed_ver == _VSCODE_EXT_VERSION:
                        _VSCODE_EXT_INSTALLED = True
                        return True
                    # Wrong version — will reinstall below
                    break
    except Exception:
        pass

    # Not installed — install from the .vsix bundled inside our package.
    vsix_path = str(_pkg_files(__package__).joinpath("arrayview-opener.vsix"))
    if not os.path.isfile(vsix_path):
        return False
    try:
        r = subprocess.run(
            [code, "--install-extension", vsix_path, "--force"],
            capture_output=True,
            timeout=30,
            env=env,
        )
        if r.returncode == 0:
            _VSCODE_EXT_INSTALLED = True
            return True
    except Exception:
        pass
    return False


def _open_browser(url: str, blocking: bool = False) -> None:
    """Open *url* in VS Code's Simple Browser via the arrayview-opener extension,
    or fall back to ``webbrowser.open``.

    The arrayview-opener VS Code extension registers a URI handler so that
    ``open "vscode://arrayview.arrayview-opener/open?url=<encoded>"`` (macOS)
    or ``code --open-url vscode://...`` (Linux / tunnel) triggers
    ``simpleBrowser.show`` inside VS Code.  The extension also resolves the
    URL through ``vscode.env.asExternalUri`` so port forwarding works in
    remote/tunnel contexts.

    The extension is automatically installed from a bundled .vsix on first use.

    blocking=True runs synchronously (CLI); blocking=False uses a daemon thread.
    """

    def _do() -> None:
        from urllib.parse import quote as _quote

        vscode_uri = None

        # Ensure the companion extension is installed before trying the URI.
        if _ensure_vscode_extension():
            vscode_uri = "vscode://arrayview.arrayview-opener/open?url=" + _quote(
                url, safe=""
            )

        opened = False

        if vscode_uri:
            if sys.platform == "darwin" and not _is_vscode_remote():
                # macOS local: `open` routes vscode:// URIs via Launch Services
                try:
                    r = subprocess.run(
                        ["open", vscode_uri],
                        capture_output=True,
                        timeout=5,
                    )
                    opened = r.returncode == 0
                except Exception:
                    pass
            else:
                # Linux / macOS-remote / tunnel: use `code --open-url`.
                # _find_code_cli() prefers the tunnel's remote-cli helper
                # when VSCODE_IPC_HOOK_CLI is set, so we route through the
                # tunnel instead of opening a new desktop VS Code window.
                code = _find_code_cli()
                ipc = _find_vscode_ipc_hook()
                if code and ipc:
                    env = {**os.environ, "VSCODE_IPC_HOOK_CLI": ipc}
                    try:
                        r = subprocess.run(
                            [code, "--open-url", vscode_uri],
                            env=env,
                            capture_output=True,
                            timeout=8,
                        )
                        opened = r.returncode == 0
                    except Exception:
                        pass

                    # Fallback: open the HTTP URL directly via code CLI.
                    # With simpleBrowser.useIntegratedBrowser VS Code opens
                    # http:// URLs in Simple Browser.  Port forwarding is
                    # handled automatically by VS Code's tunnel.
                    if not opened:
                        try:
                            r = subprocess.run(
                                [code, "--open-url", url],
                                env=env,
                                capture_output=True,
                                timeout=8,
                            )
                            opened = r.returncode == 0
                        except Exception:
                            pass

                # Fallback: xdg-open on desktop Linux
                if not opened and not _is_vscode_remote():
                    try:
                        r = subprocess.run(
                            ["xdg-open", vscode_uri],
                            capture_output=True,
                            timeout=5,
                        )
                        opened = r.returncode == 0
                    except Exception:
                        pass

        if not opened:
            # Final fallback: print URL prominently (cmd-clickable in VS Code terminal)
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


async def _serve_background(port: int):
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
    await server.serve(sockets=[sock])


def view(
    data,
    name: str = None,
    port: int = 8123,
    inline: bool | None = None,
    height: int = 500,
    window: bool | None = None,
    new_window: bool = True,
):
    """
    Launch the viewer. Does not block the main Python process.
    window defaults to False in Jupyter (inline IFrame) and True elsewhere.
    If window=True and new_window=True (default), each call opens a fresh native window.
    If window=True and new_window=False, repeated calls inject new tabs into the existing window.
    Returns an IPython IFrame in inline mode, otherwise returns the opened shell URL.
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
        threading.Thread(
            target=lambda: asyncio.run(_serve_background(port)),
            daemon=True,
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

    url_inline = f"http://localhost:{port}/?sid={session.sid}"
    encoded_name = urllib.parse.quote(name)
    url_shell = (
        f"http://localhost:{port}/shell?init_sid={session.sid}&init_name={encoded_name}"
    )

    if inline:
        from IPython.display import IFrame

        return IFrame(src=url_inline, width="100%", height=height)

    # Always print the URL so it's accessible regardless of environment.
    print(f"[ArrayView] {url_shell}", flush=True)

    can_native_window = _can_native_window() if window else False
    if window and can_native_window:
        try:
            if (
                not new_window
                and _window_process is not None
                and _window_process.poll() is None
            ):
                asyncio.run_coroutine_threadsafe(
                    _notify_shells(session.sid, name), SERVER_LOOP
                )
            else:
                _window_process = _open_webview_with_fallback(url_shell, win_w, win_h)
        except Exception:
            _open_browser(url_shell)
    else:
        if window and not can_native_window:
            print(
                "[ArrayView] Native window unavailable; opening browser fallback",
                flush=True,
            )
        _open_browser(url_shell)
    return url_shell


def _wait_for_shell_close(grace_seconds: float = 8.0) -> None:
    """Block until the browser/window closes.
    Waits for a shell WebSocket to connect, then disconnect, then applies a
    grace period so page refreshes don't prematurely kill the server.
    """
    while not SHELL_SOCKETS:
        time.sleep(0.2)
    while True:
        while SHELL_SOCKETS:
            time.sleep(0.2)
        # All sockets gone — grace period for page refresh / reconnect
        deadline = time.monotonic() + grace_seconds
        while time.monotonic() < deadline:
            if SHELL_SOCKETS:
                break  # reconnected; wait again
            if _PENDING_OPENS > 0:
                deadline = max(deadline, time.monotonic() + grace_seconds)
            time.sleep(0.2)
        else:
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


def _view_julia(data: np.ndarray, name: str, port: int, window: bool) -> str:
    """Julia-specific view() path: run the server in a subprocess so it is
    completely independent of Julia's GIL.
    """
    import tempfile

    # Persist the array to a temp file so the server subprocess can load it.
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tmp:
        tmp_path = tmp.name
    np.save(tmp_path, data)

    encoded_name = urllib.parse.quote(name)

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
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if not _wait_for_port(port, timeout=15.0):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
            raise RuntimeError(f"ArrayView server failed to start on port {port}.")

    url = f"http://localhost:{port}/shell?init_sid={sid}&init_name={encoded_name}"
    print(f"[ArrayView] {url}", flush=True)

    can_native = _can_native_window()
    if window and can_native:
        if not _open_webview_cli(url, 1200, 800):
            print("[ArrayView] Falling back to browser", flush=True)
            _open_browser(url)
    else:
        _open_browser(url)
    return url


def _serve_daemon(
    filepath: str, port: int, sid: str, name: str = None, cleanup: bool = False
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
    threading.Thread(
        target=lambda: uvicorn.run(
            app, host="127.0.0.1", port=port, log_level="error", timeout_keep_alive=30
        ),
        daemon=True,
    ).start()
    _wait_for_shell_close()
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
        "--tab",
        action="store_true",
        help="When a server is already running, inject into the existing window as a new tab without opening a new browser window",
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

    if is_arrayview_server:
        # Server already running — register the new array
        try:
            body = json.dumps(
                {
                    "filepath": os.path.abspath(args.file),
                    "name": name,
                    "notify": args.tab,  # push into existing window only with --tab
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
        encoded_name = urllib.parse.quote(name)
        url = f"http://localhost:{args.port}/shell?init_sid={sid}&init_name={encoded_name}"

        if args.tab:
            print(f"Injected as new tab in existing window (port {args.port})")
        elif args.browser or not _can_native_window():
            print(f"Open {url} in your browser")
            _open_browser(url, blocking=True)
        else:
            print(f"Open {url} in your browser")
            if not _open_webview_cli(url, 1200, 800):
                print("[ArrayView] Falling back to browser", flush=True)
                _open_browser(url, blocking=True)
        return

    sid = uuid.uuid4().hex
    encoded_name = urllib.parse.quote(name)
    url = f"http://localhost:{args.port}/shell?init_sid={sid}&init_name={encoded_name}"

    # Spawn background server — exits automatically when the window/tab is closed
    script = (
        f"from arrayview._app import _serve_daemon;"
        f"_serve_daemon({repr(os.path.abspath(args.file))}, {args.port}, {repr(sid)})"
    )
    subprocess.Popen(
        [sys.executable, "-c", script],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if not _wait_for_port(args.port, timeout=15.0):
        print(
            f"Error: ArrayView server failed to start on port {args.port}. "
            "Use --port to pick another."
        )
        sys.exit(1)

    can_native = _can_native_window()
    print(f"Open {url} in your browser")
    if args.browser or not can_native:
        _open_browser(url, blocking=True)
    else:
        if not _open_webview_cli(url, 1200, 800):
            print("[ArrayView] Falling back to browser", flush=True)
            _open_browser(url, blocking=True)
