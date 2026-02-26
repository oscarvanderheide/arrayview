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
import webbrowser
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
def _open_webview(url: str, win_w: int, win_h: int, capture_stderr: bool = False) -> subprocess.Popen:
    """Launch pywebview in a fresh subprocess. Uses subprocess.Popen to avoid
    multiprocessing bootstrap errors when called from a Jupyter kernel."""
    # Qt WebEngine renders at device-pixel-ratio scale, producing a thin
    # scrollbar at zoom=1.0.  A slightly lower zoom prevents it.
    zoom = 0.9 if sys.platform.startswith("linux") else 1.0
    script = (
        "import sys,webview;"
        "u,w,h,z=sys.argv[1],int(sys.argv[2]),int(sys.argv[3]),float(sys.argv[4]);"
        "webview.create_window('ArrayView',u,width=w,height=h,background_color='#111111',zoom=z);"
        "webview.start(**({'gui':'qt'} if sys.platform.startswith('linux') else {}))"
    )
    return subprocess.Popen(
        [sys.executable, "-c", script, url, str(win_w), str(win_h), str(zoom)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE if capture_stderr else subprocess.DEVNULL,
    )


def _open_webview_with_fallback(url: str, win_w: int, win_h: int) -> subprocess.Popen:
    """Launch pywebview, falling back to _open_browser if the subprocess exits immediately.

    Used from view() (Python API) where the host process stays alive.
    Uses a daemon watchdog thread — do NOT use from the CLI where the process exits immediately.
    """
    proc = _open_webview(url, win_w, win_h)

    def _watchdog():
        for _ in range(20):
            time.sleep(0.1)
            if proc.poll() is not None:
                print(f"[ArrayView] Native window exited (code {proc.returncode}), opening in browser", flush=True)
                _open_browser(url)
                return

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
            print(f"[ArrayView] Native window exited immediately (code {proc.returncode})", flush=True)
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


class Session:
    def __init__(self, data, filepath=None, name=None):
        self.sid = uuid.uuid4().hex
        self.data = data
        self.shape = data.shape
        self.filepath = filepath
        self.name = name or (os.path.basename(filepath) if filepath else f"Array {data.shape}")
        self.global_stats = {}
        self.fft_original_data = None
        self.fft_axes = None

        self.raw_cache = OrderedDict()
        self.rgba_cache = OrderedDict()
        self.mosaic_cache = OrderedDict()

        self.RAW_CACHE_BYTES  = 512  * 1024 * 1024   # 512 MB
        self.RGBA_CACHE_BYTES = 1024 * 1024 * 1024   # 1 GB
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
_SHELL_HTML: str = _pkg_files(__package__).joinpath("_shell.html").read_text(encoding="utf-8")
_VIEWER_HTML_TEMPLATE: str = _pkg_files(__package__).joinpath("_viewer.html").read_text(encoding="utf-8")


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
        arrays = {k: v for k, v in mat.items() if not k.startswith("_") and isinstance(v, np.ndarray)}
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
        return np.asarray(obj)
    raise ValueError(
        f"Cannot convert {type(obj)} from {source} to a numpy array."
    )


def mosaic_shape(batch):
    mshape = [int(batch**0.5), batch // int(batch**0.5)]
    while mshape[0] * mshape[1] < batch:
        mshape[1] += 1
    if (mshape[0] - 1) * (mshape[1] + 1) == batch:
        mshape[0] -= 1
        mshape[1] += 1
    return tuple(mshape)


def _compute_vmin_vmax(session, data, dr, complex_mode=0):
    if complex_mode == 1:
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


def _prepare_display(session, raw, complex_mode, dr, log_scale):
    data = apply_complex_mode(raw, complex_mode)
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


def apply_colormap_rgba(session, raw, colormap, dr, complex_mode=0, log_scale=False):
    data, vmin, vmax = _prepare_display(session, raw, complex_mode, dr, log_scale)
    if vmax > vmin:
        normalized = np.clip((data - vmin) / (vmax - vmin), 0, 1)
    else:
        normalized = np.zeros_like(data)
    _ensure_lut(colormap)
    lut = LUTS.get(colormap, LUTS["gray"])
    return lut[(normalized * 255).astype(np.uint8)]


def render_rgba(
    session, dim_x, dim_y, idx_tuple, colormap, dr, complex_mode=0, log_scale=False
):
    key = (dim_x, dim_y, idx_tuple, colormap, dr, complex_mode, log_scale)
    if key in session.rgba_cache:
        session.rgba_cache.move_to_end(key)
        return session.rgba_cache[key]
    raw = extract_slice(session, dim_x, dim_y, list(idx_tuple))
    rgba = apply_colormap_rgba(session, raw, colormap, dr, complex_mode, log_scale)
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
    padded = np.zeros((rows * cols, H, W), dtype=np.float32)
    padded[:n] = all_data
    grid = (
        padded.reshape(rows, cols, H, W)
        .transpose(0, 2, 1, 3)
        .reshape(rows * H, cols * W)
    )

    if vmax > vmin:
        normalized = np.clip((grid - vmin) / (vmax - vmin), 0, 1)
    else:
        normalized = np.zeros_like(grid)

    _ensure_lut(colormap)
    lut = LUTS.get(colormap, LUTS["gray"])
    rgba = lut[(normalized * 255).astype(np.uint8)]
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
    await ws.accept()
    SHELL_SOCKETS.append(ws)
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
                    SESSIONS[sid]._raw_bytes = SESSIONS[sid]._rgba_bytes = SESSIONS[sid]._mosaic_bytes = 0
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
                    render_rgba,
                    session,
                    dim_x,
                    dim_y,
                    idx_tuple,
                    colormap,
                    dr,
                    complex_mode,
                    log_scale,
                )

            h, w = rgba.shape[:2]
            raw = extract_slice(session, dim_x, dim_y, list(idx_tuple))
            _, vmin, vmax = _prepare_display(session, raw, complex_mode, dr, log_scale)

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
    return {
        "shape": list(session.shape),
        "is_complex": bool(np.iscomplexobj(session.data)),
    }


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
    padded = np.zeros((rows * cols, H, W), dtype=np.float32)
    padded[:n] = all_data
    mosaic = (
        padded.reshape(rows, cols, H, W)
        .transpose(0, 2, 1, 3)
        .reshape(rows * H, cols * W)
    )

    if vmax > vmin:
        normalized = np.clip((mosaic - vmin) / (vmax - vmin), 0, 1)
    else:
        normalized = np.zeros_like(mosaic)

    lut = LUTS.get(colormap if colormap in LUTS else "gray", LUTS["gray"])
    rgba = lut[(normalized * 255).astype(np.uint8)]
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
    return {"ok": True, "service": "arrayview"}


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
    SESSIONS[session.sid] = session
    await _notify_shells(session.sid, name)
    return {"sid": session.sid, "name": name}


@app.get("/")
def get_ui(sid: str = None):
    """Viewer iframe page. Redirects to /shell when no sid is given (e.g. VSCode popup)."""
    from fastapi.responses import RedirectResponse
    if not sid:
        return RedirectResponse(url="/shell")
    html = (
        _VIEWER_HTML_TEMPLATE
        .replace("__COLORMAPS__", str(COLORMAPS))
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


def _can_native_window() -> bool:
    """True if a pywebview native window can be opened."""
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


def _open_browser(url: str, blocking: bool = False) -> None:
    """Open URL in the user's browser.

    On headless/remote environments (no DISPLAY/WAYLAND_DISPLAY), tries
    ``code --open-url`` first so it works on VSCode tunnels and Remote-SSH
    (where it forwards the request to the local machine's browser).
    When a local display is available, goes straight to Python's webbrowser
    module (``code --open-url`` returns 0 locally but may not open anything).

    blocking=True runs synchronously (use from CLI where the process exits immediately).
    blocking=False (default) runs in a daemon thread (use from view()).
    """
    import shutil

    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))

    def _do():
        code_cli = shutil.which("code")
        # Only try code --open-url on headless/remote environments where it
        # actually forwards the URL to the local browser.  On machines with a
        # local display it returns 0 silently without opening anything.
        if code_cli and not has_display:
            print(f"[ArrayView] Trying: code --open-url {url}", flush=True)
            try:
                r = subprocess.run(
                    [code_cli, "--open-url", url],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=5,
                )
                if r.returncode == 0:
                    print("[ArrayView] code --open-url succeeded", flush=True)
                    return
                print(f"[ArrayView] code --open-url failed (exit {r.returncode})", flush=True)
            except Exception as e:
                print(f"[ArrayView] code --open-url error: {e}", flush=True)
        print(f"[ArrayView] Trying: webbrowser.open({url})", flush=True)
        try:
            webbrowser.open(url)
            print("[ArrayView] webbrowser.open called", flush=True)
        except Exception as e:
            print(f"[ArrayView] webbrowser.open failed: {e}", flush=True)

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
            return (
                payload.get("ok") is True
                and payload.get("service") == "arrayview"
            )
    except Exception:
        return False


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
    config = uvicorn.Config(
        app, host="127.0.0.1", port=port, log_level="error", timeout_keep_alive=30
    )
    server = uvicorn.Server(config)
    await server.serve()


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
    if not isinstance(data, np.ndarray) and (
        hasattr(data, "detach")        # PyTorch
        or hasattr(data, "numpy")      # PyTorch / TF / JAX
        or hasattr(data, "__jax_array__")  # JAX
        or (type(data).__module__ or "").startswith("juliacall")  # Julia via PythonCall
    ):
        data = _tensor_to_numpy(data, "view()")

    if name is None:
        name = f"Array {data.shape}"

    session = Session(data, name=name)
    SESSIONS[session.sid] = session

    win_w = 1200
    win_h = 800

    is_jupyter = _in_jupyter()
    if inline is None:
        inline = is_jupyter
    if window is None:
        window = not is_jupyter
    if window:
        inline = False

    # Start (or restart) the background server if it isn't responding.
    if _jupyter_server_port != port or not _server_alive(port):
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

    # Ensure background server captures the event loop before continuing
    while SERVER_LOOP is None:
        time.sleep(0.01)

    url_inline = f"http://127.0.0.1:{port}/?sid={session.sid}"
    encoded_name = urllib.parse.quote(name)
    url_shell = (
        f"http://127.0.0.1:{port}/shell?init_sid={session.sid}&init_name={encoded_name}"
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
                break       # reconnected; wait again
            time.sleep(0.2)
        else:
            return          # deadline passed with no reconnect → really closed


def _serve_daemon(filepath: str, port: int, sid: str) -> None:
    """Background server process. Loads data, serves it, exits when the UI closes."""
    data = load_data(filepath)
    session = Session(data, filepath=filepath)
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
    parser.add_argument("file", help="Path to array file (.npy, .npz, .nii/.nii.gz, .zarr, .pt/.pth, .h5/.hdf5, .tif/.tiff, .mat)")
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
            body = json.dumps({
                "filepath": os.path.abspath(args.file),
                "name": name,
                "notify": args.tab,  # push into existing window only with --tab
            }).encode()
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
            print(f"Error: port {args.port} is in use by another process. "
                  f"Use --port to pick another. ({e})")
            sys.exit(1)

        sid = result["sid"]
        encoded_name = urllib.parse.quote(name)
        url = f"http://127.0.0.1:{args.port}/shell?init_sid={sid}&init_name={encoded_name}"

        if args.tab:
            print(f"Injected as new tab in existing window (port {args.port})")
        else:
            print(f"Open {url} in your browser")
            _open_browser(url, blocking=True)
        return

    sid = uuid.uuid4().hex
    encoded_name = urllib.parse.quote(name)
    url = f"http://127.0.0.1:{args.port}/shell?init_sid={sid}&init_name={encoded_name}"

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
    print(f"[ArrayView] DISPLAY={os.environ.get('DISPLAY')!r}  WAYLAND_DISPLAY={os.environ.get('WAYLAND_DISPLAY')!r}  can_native_window={can_native}", flush=True)
    print(f"Open {url} in your browser")
    if args.browser or not can_native:
        _open_browser(url, blocking=True)
    else:
        if not _open_webview_cli(url, 1200, 800):
            print("[ArrayView] Falling back to browser", flush=True)
            _open_browser(url, blocking=True)
