import argparse
import asyncio
import io
import json
import socket
import sys
import time
import threading
import subprocess
import uuid
import urllib.parse
import webbrowser
from collections import OrderedDict

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
def _open_webview(url: str, win_w: int, win_h: int) -> subprocess.Popen:
    """Launch pywebview in a fresh subprocess. Uses subprocess.Popen to avoid
    multiprocessing bootstrap errors when called from a Jupyter kernel."""
    script = (
        "import sys,webview;"
        "u,w,h=sys.argv[1],int(sys.argv[2]),int(sys.argv[3]);"
        "webview.create_window('ArrayView',u,width=w,height=h,background_color='#111111');"
        "webview.start()"
    )
    return subprocess.Popen(
        [sys.executable, "-c", script, url, str(win_w), str(win_h)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


# ---------------------------------------------------------------------------
# Session & Global State Management
# ---------------------------------------------------------------------------
SERVER_LOOP = None
SHELL_SOCKETS = []
_window_process = None


class Session:
    def __init__(self, data, filepath=None):
        self.sid = uuid.uuid4().hex
        self.data = data
        self.shape = data.shape
        self.filepath = filepath
        self.global_stats = {}
        self.fft_original_data = None
        self.fft_axes = None

        self.raw_cache = OrderedDict()
        self.rgba_cache = OrderedDict()
        self.mosaic_cache = OrderedDict()

        self.RAW_CACHE_MAX = 200
        self.RGBA_CACHE_MAX = 512
        self.MOSAIC_CACHE_MAX = 64

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


def load_data(filepath):
    if filepath.endswith(".npy"):
        return np.load(filepath, mmap_mode="r")
    elif filepath.endswith(".nii") or filepath.endswith(".nii.gz"):
        return nib.load(filepath).dataobj
    elif filepath.endswith(".zarr") or filepath.endswith(".zarr.zip"):
        import zarr

        return zarr.open(filepath, mode="r")
    else:
        raise ValueError(
            "Unsupported format. Please provide a .npy, .nii/.nii.gz, or .zarr file"
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
    if len(session.raw_cache) > session.RAW_CACHE_MAX:
        session.raw_cache.popitem(last=False)
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


def apply_colormap_rgba(session, raw, colormap, dr, complex_mode=0, log_scale=False):
    data, vmin, vmax = _prepare_display(session, raw, complex_mode, dr, log_scale)
    if vmax > vmin:
        normalized = np.clip((data - vmin) / (vmax - vmin), 0, 1)
    else:
        normalized = np.zeros_like(data)
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
    if len(session.rgba_cache) > session.RGBA_CACHE_MAX:
        session.rgba_cache.popitem(last=False)
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

    lut = LUTS.get(colormap, LUTS["gray"])
    rgba = lut[(normalized * 255).astype(np.uint8)]
    session.mosaic_cache[key] = rgba
    if len(session.mosaic_cache) > session.MOSAIC_CACHE_MAX:
        session.mosaic_cache.popitem(last=False)
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
        if dim_z < 0:
            session.RGBA_CACHE_MAX = max(512, n * 4)

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
    return {"status": "ok"}


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
    html_content = """<!DOCTYPE html>
<html>
<head>
    <title>ArrayView</title>
    <style>
        body { 
            margin: 0; padding: 0; background: #000; color: #ccc; 
            font-family: sans-serif; overflow: hidden; height: 100vh; 
            display: flex; flex-direction: column; 
        }
        #tab-bar { 
            height: 34px; background: #161616; display: flex; align-items: flex-end; 
            padding: 0 8px; user-select: none; flex-shrink: 0; overflow-x: auto; 
            border-bottom: 1px solid #333;
        }
        .tab { 
            background: #2a2a2a; margin-right: 4px; padding: 6px 14px; 
            border-radius: 6px 6px 0 0; cursor: pointer; display: flex; 
            align-items: center; font-size: 13px; max-width: 250px; 
            white-space: nowrap; overflow: hidden; text-overflow: ellipsis; 
            border-top: 2px solid transparent; transition: background 0.1s;
        }
        .tab.active { background: #111; border-top: 2px solid #55aaff; color: #fff; }
        .tab:hover:not(.active) { background: #3a3a3a; }
        .tab span { overflow: hidden; text-overflow: ellipsis; }
        .tab-close { 
            margin-left: 10px; font-size: 16px; opacity: 0.5; cursor: pointer; 
            line-height: 1; padding: 0 4px; border-radius: 50%;
        }
        .tab-close:hover { opacity: 1; color: #ff5555; background: rgba(255,255,255,0.1); }
        
        #content { flex: 1; position: relative; background: #111; }
        iframe { 
            width: 100%; height: 100%; border: none; position: absolute; 
            top: 0; left: 0; visibility: hidden; background: #111;
        }
        iframe.active { visibility: visible; }
        
        #tab-bar::-webkit-scrollbar { height: 4px; }
        #tab-bar::-webkit-scrollbar-thumb { background: #555; border-radius: 4px; }
    </style>
</head>
<body>
    <div id="tab-bar"></div>
    <div id="content"></div>
    <script>
        const tabBar = document.getElementById('tab-bar');
        const content = document.getElementById('content');
        let tabs = {}; // sid -> { button, iframe }
        let activeSid = null;
        
        function addTab(sid, name) {
            if (tabs[sid]) { activateTab(sid); return; }
            
            const btn = document.createElement('div');
            btn.className = 'tab';
            btn.innerHTML = `<span>${name}</span><span class="tab-close">&times;</span>`;
            
            const iframe = document.createElement('iframe');
            iframe.src = `/?sid=${sid}`;
            
            tabs[sid] = { btn, iframe };
            
            btn.onclick = () => activateTab(sid);
            btn.querySelector('.tab-close').onclick = (e) => {
                e.stopPropagation();
                closeTab(sid);
            };
            
            tabBar.appendChild(btn);
            content.appendChild(iframe);
            activateTab(sid);
        }
        
        function activateTab(sid) {
            activeSid = sid;
            for (let k in tabs) {
                if (k === sid) {
                    tabs[k].btn.classList.add('active');
                    tabs[k].iframe.classList.add('active');
                    // Give focus back to the iframe so keybindings still work!
                    tabs[k].iframe.contentWindow?.focus();
                } else {
                    tabs[k].btn.classList.remove('active');
                    tabs[k].iframe.classList.remove('active');
                }
            }
        }
        
        function closeTab(sid) {
            const tab = tabs[sid];
            if (!tab) return;
            tabBar.removeChild(tab.btn);
            content.removeChild(tab.iframe);
            delete tabs[sid];
            
            // Notify backend to instantly wipe this array from python memory
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({action: "close", sid: sid}));
            }
            
            // Activate adjacent tab automatically
            if (activeSid === sid) {
                const remaining = Object.keys(tabs);
                if (remaining.length > 0) {
                    activateTab(remaining[remaining.length - 1]);
                } else {
                    activeSid = null;
                }
            }
        }

        // Init WebSocket to listen for python view() calls injecting new tabs
        const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
        let ws = new WebSocket(`${proto}//${location.host}/ws/shell`);
        ws.onmessage = (e) => {
            const msg = JSON.parse(e.data);
            if (msg.action === 'new_tab') {
                addTab(msg.sid, msg.name);
            }
        };
        
        // Auto-load first array requested in query param
        const params = new URLSearchParams(window.location.search);
        const initSid = params.get('init_sid');
        const initName = params.get('init_name');
        if (initSid) {
            addTab(initSid, initName || 'Array');
        }
    </script>
</body>
</html>
"""
    return HTMLResponse(content=html_content)


@app.get("/")
def get_ui():
    """Returns the standalone array viewer embedded inside the iframe."""
    html_content = (
        """<!DOCTYPE html>
<html>
<head>
    <style>
        :root {
            --bg: #111; --surface: #1e1e1e; --border: #444;
            --text: #ccc; --muted: #777; --subtle: #444;
            --highlight: #fff; --canvas-border: #555;
        }
        html, body { 
            background: var(--bg); margin: 0; padding: 0; 
            width: 100%; height: 100%; overflow: hidden; 
        }
        #wrapper {
            background: var(--bg); color: var(--text); font-family: monospace;
            display: flex; flex-direction: column; align-items: center;
            padding: 16px 20px 20px; width: 100%; height: 100%; box-sizing: border-box;
        }
        #wrapper.light {
            --bg: #f0f0f0; --surface: #e0e0e0; --border: #bbb;
            --text: #333; --muted: #888; --subtle: #bbb;
            --highlight: #000; --canvas-border: #999;
        }
        #info { margin-bottom: 12px; font-size: 16px; white-space: nowrap; text-align: left; flex-shrink: 0; }
        
        #viewer-row { 
            display: flex; align-items: center; justify-content: center; flex: 1; 
            min-height: 0; width: 100%; overflow: auto; padding: 20px; box-sizing: border-box;
        }
        
        #canvas-wrap { position: relative; display: inline-flex; justify-content: center; align-items: center; }
        canvas { border: 1px solid var(--canvas-border); image-rendering: pixelated; outline: none; cursor: crosshair; }
        #colorbar { display: none; position: absolute; left: 100%; top: 0; margin-left: 6px; border: none; cursor: default; }
        .highlight { color: var(--highlight); font-weight: bold; }
        .muted { color: var(--muted); }
        #status { margin-top: 8px; font-size: 13px; color: var(--muted); min-height: 1.2em; flex-shrink: 0; }
        #pixel-info { margin-top: 2px; font-size: 12px; color: var(--text); min-height: 1em; font-family: monospace; flex-shrink: 0; }
        #preload-status { margin-top: 4px; font-size: 12px; color: var(--subtle); min-height: 1em; flex-shrink: 0; }
        #toast {
            margin-top: 8px; font-size: 13px; color: var(--text);
            min-height: 1.2em; opacity: 0; transition: opacity 0.8s ease; flex-shrink: 0;
        }
        #help-overlay {
            display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%;
            background: rgba(0,0,0,0.75); z-index: 10; justify-content: center; align-items: center;
        }
        #help-overlay.visible { display: flex; }
        #help-box {
            background: var(--surface); border: 1px solid var(--border); border-radius: 10px;
            padding: 30px 40px; font-size: 15px; line-height: 2; color: var(--text);
            white-space: pre;
        }
        #help-box .key { color: var(--highlight); font-weight: bold; display: inline-block; min-width: 140px; }
        #help-hint { position: fixed; bottom: 12px; right: 16px; color: var(--muted); font-size: 14px; cursor: pointer; font-family: monospace; user-select: none; }
        #data-info { margin-top: 8px; font-size: 13px; color: var(--text); white-space: pre; opacity: 0; transition: opacity 0.4s ease; pointer-events: none; }
    </style>
</head>
<body>
<div id="wrapper">
    <div id="info">Connecting...</div>
    <div id="viewer-row">
        <div id="canvas-wrap">
            <canvas id="viewer" tabindex="0"></canvas>
            <canvas id="colorbar"></canvas>
        </div>
    </div>
    <textarea id="keyboard-sink" autocomplete="off" autocorrect="off" spellcheck="false"
              style="position:fixed;top:-1px;left:-1px;width:1px;height:1px;opacity:0;border:none;padding:0;margin:0;resize:none;overflow:hidden;"></textarea>
    <div id="status"></div>
    <div id="pixel-info"></div>
    <div id="toast"></div>
    <div id="preload-status"></div>
    <div id="help-hint">?</div>
    <div id="data-info"></div>
    <div id="help-overlay">
        <div id="help-box"><span class="key">scroll</span>  previous / next slice (active dim)
<span class="key">h / l / ← / →</span>  move cursor to prev / next dim
<span class="key">j / ↓</span>  on x/y: flip axis  |  else: prev index
<span class="key">k / ↑</span>  on x/y: flip axis  |  else: next index
<span class="key">L</span>  toggle log scale
<span class="key">x</span>  swap horizontal dim with slice dim
<span class="key">y</span>  swap vertical dim with slice dim
<span class="key">Space</span>  toggle auto-play
<span class="key">z</span>  claim dim as z (grid), scroll through next dim
<span class="key">m</span>  cycle complex mode (mag/phase/real/imag)
<span class="key">i</span>  show data info overlay
<span class="key">f</span>  toggle centred FFT (prompts for axes)
<span class="key">c</span>  cycle colormap
<span class="key">d</span>  cycle dynamic range
<span class="key">b</span>  toggle colorbar
<span class="key">t</span>  toggle dark / light theme
<span class="key">s</span>  save screenshot (PNG)
<span class="key">g</span>  save GIF of current slice dim
<span class="key">+ / -</span>  zoom in / out
<span class="key">0</span>  reset zoom (fit to window)
<span class="key">hover</span>  show pixel value
<span class="key">?</span>  toggle this help</div>
    </div>
</div>
    <script>
        const COLORMAPS = """
        + str(COLORMAPS)
        + """;
        const DR_LABELS = """
        + str(DR_LABELS)
        + """;
        const COLORMAP_GRADIENT_STOPS = """
        + json.dumps(COLORMAP_GRADIENT_STOPS)
        + """;
        const COMPLEX_MODES = """
        + str(COMPLEX_MODES)
        + """;
        const REAL_MODES = """
        + str(REAL_MODES)
        + """;

        const urlParams = new URLSearchParams(window.location.search);
        const sid = urlParams.get('sid');

        let shape = [];
        let dim_x = 0, dim_y = 1, current_slice_dim = 2;
        let activeDim = 2;
        let indices = [];
        let colormap_idx = 0, dr_idx = 1;
        let isPlaying = false, playInterval = null;
        let dim_z = -1;
        let lastDirection = 1;
        let isDark = true;
        let isComplex = false;
        let complexMode = 0;

        let showColorbar = false;
        let currentVmin = 0, currentVmax = 1;
        let lastImageData = null, lastImgW = 0, lastImgH = 0;

        let ws = null, wsReady = false, wsSentSeq = 0;
        let isRendering = false;
        let pendingRequest = false;

        let preloadPolling = null;
        let preloadActiveDim = -1;
        let toastTimer = null;
        let dataInfoTimer = null;
        
        let userZoom = 1.0;
        let pixelHoverPending = false;
        let flip_x = false, flip_y = false;
        let _fftActive = false;
        let logScale = false;

        const canvas = document.getElementById('viewer');
        const ctx = canvas.getContext('2d');
        const colorbarCanvas = document.getElementById('colorbar');
        const cbCtx = colorbarCanvas.getContext('2d');
        const sink = document.getElementById('keyboard-sink');

        if (!sid) {
            document.getElementById('info').textContent = "No session ID provided in URL.";
        }

        function getBaseScale(w, h) {
            const row = document.getElementById('viewer-row');
            const maxW = Math.max(100, row.clientWidth - 40);
            const maxH = Math.max(100, row.clientHeight - 40);
            if (maxW <= 0 || maxH <= 0) return 1.0;
            return Math.min(maxW / w, maxH / h);
        }

        function scaleCanvas(w, h) {
            const baseScale = getBaseScale(w, h);
            const finalScale = baseScale * userZoom;
            canvas.style.width  = Math.round(w * finalScale) + 'px';
            canvas.style.height = Math.round(h * finalScale) + 'px';
            if (showColorbar) drawColorbar();
        }
        
        window.addEventListener('resize', () => {
            if (lastImgW && lastImgH) scaleCanvas(lastImgW, lastImgH);
        });

        function showToast(msg) {
            const el = document.getElementById('toast');
            el.textContent = msg;
            el.style.transition = 'none';
            el.style.opacity = '1';
            if (toastTimer) clearTimeout(toastTimer);
            toastTimer = setTimeout(() => {
                el.style.transition = 'opacity 0.8s ease';
                el.style.opacity = '0';
            }, 1500);
        }

        function drawColorbar() {
            const stops = COLORMAP_GRADIENT_STOPS[COLORMAPS[colormap_idx]];
            const n = stops.length;
            const dpr = window.devicePixelRatio || 1;
            const cssH = parseInt(canvas.style.height);
            const cbCSSW = 50, barW = 14, barX = 8;
            const barH = Math.max(60, cssH - 40);
            const barY = Math.floor((cssH - barH) / 2);

            colorbarCanvas.width = Math.round(cbCSSW * dpr);
            colorbarCanvas.height = Math.round(cssH * dpr);
            colorbarCanvas.style.width = cbCSSW + 'px';
            colorbarCanvas.style.height = cssH + 'px';
            cbCtx.scale(dpr, dpr);

            for (let row = 0; row < barH; row++) {
                const t = 1 - row / (barH - 1);
                const fi = t * (n - 1);
                const lo = Math.floor(fi), hi = Math.min(lo + 1, n - 1);
                const frac = fi - lo;
                const r = Math.round(stops[lo][0] * (1 - frac) + stops[hi][0] * frac);
                const g = Math.round(stops[lo][1] * (1 - frac) + stops[hi][1] * frac);
                const b = Math.round(stops[lo][2] * (1 - frac) + stops[hi][2] * frac);
                cbCtx.fillStyle = `rgb(${r},${g},${b})`;
                cbCtx.fillRect(barX, barY + row, barW, 1);
            }

            cbCtx.strokeStyle = '#888'; cbCtx.lineWidth = 1;
            cbCtx.strokeRect(barX - 0.5, barY - 0.5, barW + 1, barH + 1);

            const fmt = v => {
                const av = Math.abs(v);
                if (av === 0) return '0';
                if (av >= 1e4 || (av < 1e-2 && av > 0)) return v.toExponential(2);
                return parseFloat(v.toPrecision(3)).toString();
            };
            cbCtx.font = '10px monospace';
            cbCtx.textAlign = 'left';
            cbCtx.fillStyle = isDark ? '#ddd' : '#222';
            const labelX = barX + barW + 3;
            cbCtx.fillText(fmt(currentVmax), labelX, barY + 9);
            cbCtx.fillText(fmt(currentVmin), labelX, barY + barH);
        }

        function showDataInfo(text) {
            const el = document.getElementById('data-info');
            el.textContent = text;
            el.style.transition = 'none';
            el.style.opacity = '1';
            if (dataInfoTimer) clearTimeout(dataInfoTimer);
            dataInfoTimer = setTimeout(() => {
                el.style.transition = 'opacity 0.8s ease';
                el.style.opacity = '0';
            }, 4000);
        }

        function applyFlips(imageData, w, h) {
            if (!flip_x && !flip_y) return imageData;
            const src = imageData.data;
            const out = new Uint8ClampedArray(src.length);
            for (let row = 0; row < h; row++) {
                const srcRow = flip_y ? (h - 1 - row) : row;
                for (let col = 0; col < w; col++) {
                    const srcCol = flip_x ? (w - 1 - col) : col;
                    const si = (srcRow * w + srcCol) * 4;
                    const di = (row * w + col) * 4;
                    out[di] = src[si]; out[di+1] = src[si+1];
                    out[di+2] = src[si+2]; out[di+3] = src[si+3];
                }
            }
            return new ImageData(out, w, h);
        }

        function initWebSocket() {
            const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${proto}//${location.host}/ws/${sid}`);
            ws.binaryType = 'arraybuffer';

            ws.onopen = () => {
                wsReady = true;
                setStatus('');
                sink.focus();
                updateView();
            };

            ws.onmessage = (event) => {
                const buf = event.data;
                const headerU32 = new Uint32Array(buf, 0, 3);
                const seq    = headerU32[0];
                const width  = headerU32[1];
                const height = headerU32[2];

                if (seq === wsSentSeq) {
                    const headerF32 = new Float32Array(buf, 12, 2);
                    currentVmin = headerF32[0];
                    currentVmax = headerF32[1];

                    const rgba = new Uint8ClampedArray(buf.slice(20));
                    lastImageData = new ImageData(rgba, width, height);
                    lastImgW = width; lastImgH = height;
                    canvas.width  = width;
                    canvas.height = height;
                    ctx.putImageData(applyFlips(lastImageData, width, height), 0, 0);
                    scaleCanvas(width, height);
                }

                isRendering = false;
                if (pendingRequest) {
                    pendingRequest = false;
                    updateView();
                }
            };

            ws.onclose = () => {
                wsReady = false;
                setStatus('WebSocket closed.');
            };

            ws.onerror = () => ws.close();
        }

        function triggerPreload() {
            if (shape.length < 3) return;
            preloadActiveDim = current_slice_dim;
            fetch(`/preload/${sid}`, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    dim_x, dim_y, dim_z,
                    indices: [...indices],
                    colormap: COLORMAPS[colormap_idx],
                    dr: dr_idx,
                    complex_mode: complexMode,
                    log_scale: logScale,
                    slice_dim: current_slice_dim,
                })
            });
            if (preloadPolling) clearInterval(preloadPolling);
            preloadPolling = setInterval(pollPreloadStatus, 500);
        }

        function pollPreloadStatus() {
            fetch(`/preload_status/${sid}`).then(r => r.json()).then(data => {
                const el = document.getElementById('preload-status');
                if (data.skipped) {
                    el.textContent = 'Array too large for full preload (>500 MB).';
                    clearInterval(preloadPolling); preloadPolling = null;
                } else if (data.total > 0 && data.done >= data.total) {
                    el.textContent = '';
                    clearInterval(preloadPolling); preloadPolling = null;
                } else if (data.total > 0) {
                    const pct = Math.round(data.done / data.total * 100);
                    el.textContent = `Preloading dim ${preloadActiveDim}: ${data.done}/${data.total} (${pct}%)`;
                }
            }).catch(() => {
                clearInterval(preloadPolling); preloadPolling = null;
            });
        }

        async function init() {
            if (!sid) return;
            const res = await fetch(`/metadata/${sid}`);
            if (!res.ok) {
                document.getElementById('info').textContent = "Session expired or invalid.";
                return;
            }
            const data = await res.json();
            shape = data.shape;
            isComplex = data.is_complex || false;
            indices = shape.map(s => Math.floor(s / 2));
            dim_x = 0; dim_y = 1;
            current_slice_dim = shape.length > 2 ? 2 : 0;
            activeDim = current_slice_dim;
            initWebSocket();
            triggerPreload();
        }

        function getModeLabel() {
            return isComplex ? COMPLEX_MODES[complexMode] : REAL_MODES[complexMode];
        }

        function renderInfo() {
            const idxStr = indices.map((v, i) => {
                const active = (i === activeDim);
                if (i === dim_x) {
                    const inner = (flip_x ? '<span class="muted">-</span>' : '') + 'x';
                    return active ? `<span class="highlight">${inner}</span>` : inner;
                }
                if (i === dim_y) {
                    const inner = (flip_y ? '<span class="muted">-</span>' : '') + 'y';
                    return active ? `<span class="highlight">${inner}</span>` : inner;
                }
                if (i === dim_z) return active ? `<span class="highlight">z</span>` : 'z';
                return active ? `<span class="highlight">[${v}]</span>` : `${v}`;
            }).join(', ');
            let text = `[${idxStr}]`;
            if (isComplex || complexMode !== 0)
                text += `  <span class="muted">${getModeLabel()}</span>`;
            if (logScale)
                text += `  <span class="muted">log</span>`;
            if (_fftActive)
                text += `  <span class="muted">FFT</span>`;
            document.getElementById('info').innerHTML = text;
        }

        function setStatus(msg) { document.getElementById('status').textContent = msg; }

        function updateView() {
            renderInfo();
            if (!wsReady) return;
            
            if (isRendering) {
                pendingRequest = true;
                return;
            }
            
            isRendering = true;
            wsSentSeq++;
            ws.send(JSON.stringify({
                seq: wsSentSeq,
                dim_x, dim_y, dim_z,
                indices: [...indices],
                colormap: COLORMAPS[colormap_idx],
                dr: dr_idx,
                complex_mode: complexMode,
                log_scale: logScale,
                slice_dim: current_slice_dim,
                direction: lastDirection,
            }));
        }

        function stopPlay() {
            clearInterval(playInterval); playInterval = null;
            isPlaying = false; setStatus('');
        }

        function togglePlay() {
            if (isPlaying) { stopPlay(); return; }
            isPlaying = true;
            setStatus('▶ playing  (Space to stop)');
            playInterval = setInterval(() => {
                indices[current_slice_dim] = (indices[current_slice_dim] + 1) % shape[current_slice_dim];
                updateView();
            }, 80);
        }

        function saveScreenshot() {
            const link = document.createElement('a');
            link.download = `slice_${indices.join('-')}.png`;
            link.href = canvas.toDataURL('image/png');
            link.click();
            setStatus('Screenshot saved.');
            setTimeout(() => setStatus(''), 2000);
        }

        async function saveGif() {
            setStatus('Generating GIF...');
            const url = `/gif/${sid}?dim_x=${dim_x}&dim_y=${dim_y}&indices=${indices.join(',')}&colormap=${COLORMAPS[colormap_idx]}&dr=${dr_idx}&slice_dim=${current_slice_dim}`;
            const res = await fetch(url);
            const blob = await res.blob();
            const link = document.createElement('a');
            link.download = `dim${current_slice_dim}.gif`;
            link.href = URL.createObjectURL(blob);
            link.click();
            setStatus('GIF saved.');
            setTimeout(() => setStatus(''), 2000);
        }

        const helpOverlay = document.getElementById('help-overlay');

        // Allow clicks on canvas to auto-refocus keystrokes
        canvas.addEventListener('click', () => sink.focus());
        document.addEventListener('click', () => sink.focus());
        
        // This is crucial for tabs so key inputs don't get lost when swapping
        window.addEventListener('focus', () => sink.focus()); 

        canvas.addEventListener('mousemove', (e) => {
            if (dim_z >= 0) return;
            if (pixelHoverPending) return;
            pixelHoverPending = true;
            setTimeout(() => { pixelHoverPending = false; }, 50);
            const rect = canvas.getBoundingClientRect();
            const px = Math.floor((e.clientX - rect.left) * canvas.width / rect.width);
            const py = Math.floor((e.clientY - rect.top) * canvas.height / rect.height);
            fetch(`/pixel/${sid}?dim_x=${dim_x}&dim_y=${dim_y}&indices=${indices.join(',')}&px=${px}&py=${py}&complex_mode=${complexMode}`)
                .then(r => r.json())
                .then(d => {
                    const el = document.getElementById('pixel-info');
                    if (d.value !== undefined && isFinite(d.value)) {
                        const av = Math.abs(d.value);
                        const fmt = v => (av >= 1e4 || (av < 1e-2 && av > 0))
                            ? v.toExponential(3) : parseFloat(v.toPrecision(4)).toString();
                        el.textContent = `(${px}, ${py}) = ${fmt(d.value)}`;
                    } else {
                        el.textContent = '';
                    }
                });
        });
        canvas.addEventListener('mouseleave', () => {
            document.getElementById('pixel-info').textContent = '';
        });

        sink.addEventListener('keydown', (e) => {
            e.preventDefault();
            e.stopImmediatePropagation();
            if (e.key === '?') { helpOverlay.classList.toggle('visible'); return; }
            if (e.key === 'Escape') {
                helpOverlay.classList.remove('visible');
                return;
            }
            if (e.key === '+' || e.key === '=') {
                userZoom = Math.min(userZoom * 1.1, 10.0);
                scaleCanvas(canvas.width, canvas.height);
                showToast(`zoom: ${Math.round(userZoom * 100)}%`);
            } else if (e.key === '-') {
                userZoom = Math.max(userZoom / 1.1, 0.1);
                scaleCanvas(canvas.width, canvas.height);
                showToast(`zoom: ${Math.round(userZoom * 100)}%`);
            } else if (e.key === '0') {
                userZoom = 1.0;
                scaleCanvas(canvas.width, canvas.height);
                showToast(`zoom: fit`);
            } else if (e.key === 'b') {
                showColorbar = !showColorbar;
                colorbarCanvas.style.display = showColorbar ? 'block' : 'none';
                if (showColorbar && lastImageData) drawColorbar();
                showToast(showColorbar ? 'colorbar: on' : 'colorbar: off');
            } else if (e.key === 'z') {
                if (dim_z >= 0) {
                    current_slice_dim = dim_z;
                    dim_z = -1;
                } else {
                    if (shape.length < 4) return;
                    dim_z = current_slice_dim;
                    do { current_slice_dim = (current_slice_dim + 1) % shape.length; }
                    while (current_slice_dim === dim_x || current_slice_dim === dim_y || current_slice_dim === dim_z);
                }
                activeDim = current_slice_dim;
                updateView(); triggerPreload();
            } else if (e.key === ' ') {
                e.preventDefault();
                togglePlay();
            } else if (e.key === 't') {
                isDark = !isDark;
                document.getElementById('wrapper').classList.toggle('light', !isDark);
            } else if (e.key === 's') {
                saveScreenshot();
            } else if (e.key === 'g') {
                saveGif();
            } else if (e.key === 'm') {
                const modeCount = isComplex ? COMPLEX_MODES.length : REAL_MODES.length;
                complexMode = (complexMode + 1) % modeCount;
                updateView(); triggerPreload();
                showToast(`mode: ${getModeLabel()}`);
            } else if (e.key === 'c') {
                colormap_idx = (colormap_idx + 1) % COLORMAPS.length;
                fetch(`/clearcache/${sid}`); updateView(); triggerPreload();
                showToast(`colormap: ${COLORMAPS[colormap_idx]}`);
            } else if (e.key === 'd') {
                dr_idx = (dr_idx + 1) % DR_LABELS.length;
                fetch(`/clearcache/${sid}`); updateView(); triggerPreload();
                showToast(`range: ${DR_LABELS[dr_idx]}`);
            } else if (e.key === 'j' || e.key === 'ArrowDown') {
                e.preventDefault();
                if (activeDim === dim_x) {
                    flip_x = !flip_x;
                    if (lastImageData) ctx.putImageData(applyFlips(lastImageData, lastImgW, lastImgH), 0, 0);
                    renderInfo();
                } else if (activeDim === dim_y) {
                    flip_y = !flip_y;
                    if (lastImageData) ctx.putImageData(applyFlips(lastImageData, lastImgW, lastImgH), 0, 0);
                    renderInfo();
                } else {
                    lastDirection = -1;
                    indices[activeDim] = Math.max(0, indices[activeDim] - 1);
                    updateView();
                }
            } else if (e.key === 'k' || e.key === 'ArrowUp') {
                e.preventDefault();
                if (activeDim === dim_x) {
                    flip_x = !flip_x;
                    if (lastImageData) ctx.putImageData(applyFlips(lastImageData, lastImgW, lastImgH), 0, 0);
                    renderInfo();
                } else if (activeDim === dim_y) {
                    flip_y = !flip_y;
                    if (lastImageData) ctx.putImageData(applyFlips(lastImageData, lastImgW, lastImgH), 0, 0);
                    renderInfo();
                } else {
                    lastDirection = 1;
                    indices[activeDim] = Math.min(shape[activeDim] - 1, indices[activeDim] + 1);
                    updateView();
                }
            } else if (e.key === 'h' || e.key === 'ArrowLeft') {
                e.preventDefault();
                activeDim = (activeDim - 1 + shape.length) % shape.length;
                if (activeDim !== dim_x && activeDim !== dim_y && activeDim !== dim_z) {
                    current_slice_dim = activeDim; triggerPreload();
                }
                renderInfo();
            } else if (e.key === 'l' || e.key === 'ArrowRight') {
                e.preventDefault();
                activeDim = (activeDim + 1) % shape.length;
                if (activeDim !== dim_x && activeDim !== dim_y && activeDim !== dim_z) {
                    current_slice_dim = activeDim; triggerPreload();
                }
                renderInfo();
            } else if (e.key === 'L') {
                logScale = !logScale;
                fetch(`/clearcache/${sid}`); updateView(); triggerPreload();
                showToast(logScale ? 'log scale: on' : 'log scale: off');
            } else if (e.key === 'i') {
                fetch(`/info/${sid}`).then(r => r.json()).then(d => {
                    const lines = [
                        `Shape:    [${d.shape.join(', ')}]`,
                        `Dtype:    ${d.dtype}`,
                        `Elements: ${d.total_elements.toLocaleString()}`,
                        `Size:     ${d.size_mb !== null ? d.size_mb + ' MB' : 'unknown'}`,
                    ];
                    if (d.filepath) lines.push(`File:     ${d.filepath}`);
                    if (d.fft_axes) lines.push(`FFT axes: [${d.fft_axes.join(', ')}]`);
                    showDataInfo(lines.join('\\n'));
                });
            } else if (e.key === 'f') {
                if (_fftActive) {
                    fetch(`/fft/${sid}`, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({axes: ''})})
                        .then(r => r.json()).then(d => {
                            _fftActive = false;
                            isComplex = d.is_complex || false;
                            if (!isComplex && complexMode >= REAL_MODES.length) complexMode = 0;
                            updateView(); triggerPreload();
                            showToast('FFT: off');
                        });
                } else {
                    const axesStr = window.prompt('FFT axes (comma-separated, e.g. 0,1):', '0,1');
                    if (!axesStr) return;
                    fetch(`/fft/${sid}`, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({axes: axesStr})})
                        .then(r => r.json()).then(d => {
                            if (d.error) { showToast('FFT error: ' + d.error); return; }
                            _fftActive = true;
                            isComplex = d.is_complex || false;
                            complexMode = 0;
                            updateView(); triggerPreload();
                            showToast(`FFT: [${d.axes.join(',')}]`);
                        });
                }
            } else if (e.key === 'x') {
                if (shape.length < 3) return;
                [dim_x, current_slice_dim] = [current_slice_dim, dim_x];
                dim_z = -1;
                updateView(); triggerPreload();
            } else if (e.key === 'y') {
                if (shape.length < 3) return;
                [dim_y, current_slice_dim] = [current_slice_dim, dim_y];
                dim_z = -1;
                updateView(); triggerPreload();
            }
        });

        helpOverlay.addEventListener('click', () => { helpOverlay.classList.remove('visible'); sink.focus(); });
        document.getElementById('help-hint').addEventListener('click', () => { helpOverlay.classList.toggle('visible'); sink.focus(); });

        window.addEventListener('wheel', (e) => {
            e.preventDefault();
            if (e.deltaY > 0) {
                lastDirection = -1;
                indices[current_slice_dim] = Math.max(0, indices[current_slice_dim] - 1);
            } else {
                lastDirection = 1;
                indices[current_slice_dim] = Math.min(shape[current_slice_dim] - 1, indices[current_slice_dim] + 1);
            }
            updateView();
        }, {passive: false});

        function closeSocket() {
            if (ws) {
                ws.onclose = null;
                ws.close();
            }
        }
        window.addEventListener('beforeunload', closeSocket);
        window.addEventListener('unload', closeSocket);
        window.addEventListener('pagehide', closeSocket);

        init();
    </script>
</body>
</html>"""
    )
    return HTMLResponse(content=html_content)


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


def _wait_for_port(port: int, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                return
        except OSError:
            time.sleep(0.05)


async def _serve_background(port: int):
    global SERVER_LOOP
    SERVER_LOOP = asyncio.get_running_loop()
    config = uvicorn.Config(
        app, host="127.0.0.1", port=port, log_level="error", timeout_keep_alive=1
    )
    server = uvicorn.Server(config)
    await server.serve()


def view(
    data,
    name: str = None,
    port: int = 8123,
    inline: bool | None = None,
    height: int = 500,
    window: bool = True,
    new_window: bool = True,
):
    """
    Launch the viewer. Does not block the main Python process.
    If window=True and new_window=True (default), each call opens a fresh native window.
    If window=True and new_window=False, repeated calls inject new tabs into the existing window.
    """
    global _jupyter_server_port, _window_process, SERVER_LOOP  # _window_process is a Popen instance

    if name is None:
        name = f"Array {data.shape}"

    session = Session(data)
    SESSIONS[session.sid] = session

    win_w = 1200
    win_h = 800

    is_jupyter = _in_jupyter()
    if inline is None:
        inline = is_jupyter

    if window:
        inline = False

    if _jupyter_server_port != port:
        threading.Thread(
            target=lambda: asyncio.run(_serve_background(port)),
            daemon=True,
        ).start()
        _wait_for_port(port)
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

    if window:
        try:
            if (
                not new_window
                and _window_process is not None
                and _window_process.poll() is None
            ):
                # Tab mode: a window is already running, push a new tab into it over WebSockets
                asyncio.run_coroutine_threadsafe(
                    _notify_shells(session.sid, name), SERVER_LOOP
                )
            else:
                # New-window mode (or no existing window): spawn a fresh isolated native window.
                # The init_sid URL param loads the session — no _notify_shells needed and it would
                # bleed into any other open shell connections (old windows, browser tabs, etc.).
                _window_process = _open_webview(url_shell, win_w, win_h)
        except Exception as e:
            print(f"[ArrayView] Failed to spawn native window: {e}")
    else:
        # Open in standard web browser (with our custom tab bar!)
        webbrowser.open(url_shell)


def arrayview():
    """Command Line Interface Entry Point."""
    parser = argparse.ArgumentParser(description="Lightning Fast ND Array Viewer")
    parser.add_argument("file", help="Path to .npy, .nii/.nii.gz, or .zarr file")
    parser.add_argument("--port", type=int, default=8000, help="Port to serve on")
    parser.add_argument(
        "--browser",
        action="store_true",
        help="Open in web browser instead of native window",
    )
    args = parser.parse_args()

    try:
        data = load_data(args.file)
        session = Session(data, filepath=args.file)
        SESSIONS[session.sid] = session

        try:
            size_str = f" ({data.nbytes // 1024**2} MB)"
        except AttributeError:
            size_str = ""
        print(f"Loaded {args.file} with shape {session.shape}{size_str}")
        if args.browser:
            print(
                f"Open http://127.0.0.1:{args.port}/shell?init_sid={session.sid} in your browser"
            )
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

    url = f"http://127.0.0.1:{args.port}/shell?init_sid={session.sid}&init_name=Array"

    if args.browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    else:
        threading.Timer(0.5, lambda: _open_webview(url, 1200, 800)).start()

    uvicorn.run(
        app, host="127.0.0.1", port=args.port, log_level="warning", timeout_keep_alive=1
    )
