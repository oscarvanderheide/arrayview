import argparse
import asyncio
import io
import json
import socket
import sys
import time
import threading
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

DATA = None
SHAPE = None
GLOBAL_STATS = {}  # {dr_idx: (vmin, vmax)} sampled once at startup
_data_filepath: str | None = None  # set when data is loaded from a file
_fft_original_data = None  # stored before FFT so we can toggle back
_fft_axes: tuple | None = None

COLORMAPS = ["gray", "lipari", "navia", "viridis", "plasma", "RdBu_r"]
DR_PERCENTILES = [(0, 100), (1, 99), (5, 95), (10, 90)]
DR_LABELS = ["0-100%", "1-99%", "5-95%", "10-90%"]

# RGBA lookup tables (256 x 4)
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


# 32-stop RGB gradient for each colormap (embedded in the JS for colorbar drawing)
COLORMAP_GRADIENT_STOPS = {
    name: _lut_to_gradient_stops(LUTS[name]) for name in COLORMAPS
}

# Complex-mode labels.  For complex data all 4 are valid; for real data only the first 2.
COMPLEX_MODES = ["mag", "phase", "real", "imag"]
REAL_MODES = ["real", "mag"]

app = FastAPI()


def load_data(filepath):
    if filepath.endswith(".npy"):
        return np.load(filepath, mmap_mode="r")
    elif filepath.endswith(".nii") or filepath.endswith(".nii.gz"):
        img = nib.load(filepath)
        return img.dataobj
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


def _sample_for_stats(max_samples=200_000):
    """Sample data without loading the full array (mmap-friendly)."""
    total = int(np.prod(SHAPE))
    if total <= max_samples:
        sample = np.array(DATA).ravel()
    else:
        # Walk along dim 0 (sequential = mmap-friendly for C-contiguous arrays)
        n_take = min(10, SHAPE[0])
        step = max(1, SHAPE[0] // n_take)
        chunks = []
        for i in range(0, SHAPE[0], step):
            chunks.append(np.array(DATA[i]).ravel())
            if sum(c.size for c in chunks) >= max_samples:
                break
        sample = np.concatenate(chunks)
    if np.iscomplexobj(sample):
        sample = np.abs(sample)
    return np.nan_to_num(sample).astype(np.float32)


def compute_global_stats():
    global GLOBAL_STATS
    try:
        print("Computing global contrast statistics...", end="", flush=True)
        sample = _sample_for_stats()
        GLOBAL_STATS = {
            i: (float(np.percentile(sample, lo)), float(np.percentile(sample, hi)))
            for i, (lo, hi) in enumerate(DR_PERCENTILES)
        }
        print(f" done. ({len(sample):,} samples)")
    except Exception as e:
        print(f" failed ({e}), using per-slice stats.")
        GLOBAL_STATS = {}


def _compute_vmin_vmax(data, dr, complex_mode=0):
    """Return (vmin, vmax) for the given float32 data array."""
    if complex_mode == 1:  # phase: fixed physical range
        return (-float(np.pi), float(np.pi))
    if complex_mode == 0 and dr in GLOBAL_STATS:
        return GLOBAL_STATS[dr]
    pct_lo, pct_hi = DR_PERCENTILES[dr % len(DR_PERCENTILES)]
    return float(np.percentile(data, pct_lo)), float(np.percentile(data, pct_hi))


# ---------------------------------------------------------------------------
# Two-level cache
# ---------------------------------------------------------------------------
_raw_cache = OrderedDict()
_RAW_CACHE_MAX = 200

_rgba_cache = OrderedDict()
_RGBA_CACHE_MAX = 512

_mosaic_cache = OrderedDict()
_MOSAIC_CACHE_MAX = 64  # mosaics are large; 64 is plenty

# ---------------------------------------------------------------------------
# Background preload state
# ---------------------------------------------------------------------------
_preload_gen = 0  # increment to cancel the running preload thread
_preload_done = 0  # slices rendered so far
_preload_total = 0  # total slices in current preload
_preload_skipped = False  # True when array is too large to preload
_preload_lock = threading.Lock()


def extract_slice(dim_x, dim_y, idx_list):
    """Return the raw slice as float32 (real data) or complex64 (complex data)."""
    key = (dim_x, dim_y, tuple(idx_list))
    if key in _raw_cache:
        _raw_cache.move_to_end(key)
        return _raw_cache[key]

    slicer = [
        slice(None) if i in (dim_x, dim_y) else idx_list[i] for i in range(len(SHAPE))
    ]
    extracted = np.array(DATA[tuple(slicer)])
    if dim_x < dim_y:
        extracted = extracted.T
    if np.iscomplexobj(extracted):
        result = np.nan_to_num(extracted).astype(np.complex64)
    else:
        result = np.nan_to_num(extracted).astype(np.float32)

    _raw_cache[key] = result
    if len(_raw_cache) > _RAW_CACHE_MAX:
        _raw_cache.popitem(last=False)
    return result


def apply_complex_mode(raw, complex_mode):
    """Apply the requested view mode and return a float32 array."""
    if np.iscomplexobj(raw):
        if complex_mode == 1:
            result = np.angle(raw)
        elif complex_mode == 2:
            result = raw.real.copy()
        elif complex_mode == 3:
            result = raw.imag.copy()
        else:  # 0 = magnitude
            result = np.abs(raw)
    else:
        result = np.abs(raw) if complex_mode == 1 else raw
    return np.nan_to_num(result).astype(np.float32)


def _prepare_display(raw, complex_mode, dr, log_scale):
    """Apply complex mode + optional log transform; return (data_f32, vmin, vmax)."""
    data = apply_complex_mode(raw, complex_mode)
    if log_scale:
        data = np.log1p(np.abs(data)).astype(np.float32)
        pct_lo, pct_hi = DR_PERCENTILES[dr % len(DR_PERCENTILES)]
        vmin = float(np.percentile(data, pct_lo))
        vmax = float(np.percentile(data, pct_hi))
    else:
        vmin, vmax = _compute_vmin_vmax(data, dr, complex_mode)
    return data, vmin, vmax


def apply_colormap_rgba(raw, colormap, dr, complex_mode=0, log_scale=False):
    """Apply complex mode, (optional log), normalise, and map → RGBA uint8 (H, W, 4)."""
    data, vmin, vmax = _prepare_display(raw, complex_mode, dr, log_scale)
    if vmax > vmin:
        normalized = np.clip((data - vmin) / (vmax - vmin), 0, 1)
    else:
        normalized = np.zeros_like(data)
    lut = LUTS.get(colormap, LUTS["gray"])
    return lut[(normalized * 255).astype(np.uint8)]  # (H, W, 4)


def render_rgba(dim_x, dim_y, idx_tuple, colormap, dr, complex_mode=0, log_scale=False):
    """Return cached RGBA (H, W, 4) uint8 array."""
    key = (dim_x, dim_y, idx_tuple, colormap, dr, complex_mode, log_scale)
    if key in _rgba_cache:
        _rgba_cache.move_to_end(key)
        return _rgba_cache[key]
    raw = extract_slice(dim_x, dim_y, list(idx_tuple))
    rgba = apply_colormap_rgba(raw, colormap, dr, complex_mode, log_scale)
    _rgba_cache[key] = rgba
    if len(_rgba_cache) > _RGBA_CACHE_MAX:
        _rgba_cache.popitem(last=False)
    return rgba


def render_mosaic(
    dim_x, dim_y, dim_z, idx_tuple, colormap, dr, complex_mode=0, log_scale=False
):
    """Return cached RGBA mosaic of all dim_z slices."""
    idx_norm = list(idx_tuple)
    idx_norm[dim_z] = 0  # dim_z position in idx doesn't affect the mosaic
    key = (dim_x, dim_y, dim_z, tuple(idx_norm), colormap, dr, complex_mode, log_scale)
    if key in _mosaic_cache:
        _mosaic_cache.move_to_end(key)
        return _mosaic_cache[key]

    n = SHAPE[dim_z]
    frames_raw = [
        extract_slice(
            dim_x, dim_y, [i if j == dim_z else idx_tuple[j] for j in range(len(SHAPE))]
        )
        for i in range(n)
    ]
    frames = [apply_complex_mode(f, complex_mode) for f in frames_raw]
    if log_scale:
        frames = [np.log1p(np.abs(f)).astype(np.float32) for f in frames]
    all_data = np.stack(frames)  # (n, H, W) float32

    if log_scale:
        pct_lo, pct_hi = DR_PERCENTILES[dr % len(DR_PERCENTILES)]
        vmin = float(np.percentile(all_data, pct_lo))
        vmax = float(np.percentile(all_data, pct_hi))
    else:
        vmin, vmax = _compute_vmin_vmax(all_data, dr, complex_mode)

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
    _mosaic_cache[key] = rgba
    if len(_mosaic_cache) > _MOSAIC_CACHE_MAX:
        _mosaic_cache.popitem(last=False)
    return rgba


def _run_preload(
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
    """Background thread: pre-render every slice of slice_dim into cache."""
    global _preload_done, _preload_total, _preload_skipped, _RGBA_CACHE_MAX

    n = SHAPE[slice_dim]
    H = SHAPE[dim_y]
    W = SHAPE[dim_x]
    if dim_z >= 0:
        nz = SHAPE[dim_z]
        mrows, mcols = mosaic_shape(nz)
        size_bytes = n * (mrows * H) * (mcols * W) * 4
    else:
        size_bytes = n * H * W * 4

    with _preload_lock:
        _preload_total = n
        _preload_done = 0
        if size_bytes > 500 * 1024 * 1024:
            _preload_skipped = True
            return
        _preload_skipped = False
        if dim_z < 0:
            _RGBA_CACHE_MAX = max(512, n * 4)

    for i in range(n):
        if _preload_gen != gen:
            return
        idx = list(idx_list)
        idx[slice_dim] = i
        if dim_z >= 0:
            render_mosaic(
                dim_x, dim_y, dim_z, tuple(idx), colormap, dr, complex_mode, log_scale
            )
        else:
            render_rgba(dim_x, dim_y, tuple(idx), colormap, dr, complex_mode, log_scale)
        with _preload_lock:
            _preload_done = i + 1

        time.sleep(0.005)  # Yield GIL so window UI isn't starved


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    loop = asyncio.get_running_loop()

    latest_msg: dict | None = None
    latest_seq: int = 0
    new_request = asyncio.Event()

    async def receiver():
        nonlocal latest_msg, latest_seq
        try:
            while True:
                msg = await ws.receive_json()
                seq = int(msg.get("seq", 0))
                if seq > latest_seq:
                    latest_seq = seq
                    latest_msg = msg
                    new_request.set()
        except Exception:
            new_request.set()

    receiver_task = asyncio.create_task(receiver())
    try:
        while True:
            await new_request.wait()
            new_request.clear()
            if latest_msg is None:
                break

            msg = latest_msg
            seq = latest_seq
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
                    dim_x,
                    dim_y,
                    idx_tuple,
                    colormap,
                    dr,
                    complex_mode,
                    log_scale,
                )

            if seq == latest_seq:
                h, w = rgba.shape[:2]
                raw = extract_slice(dim_x, dim_y, list(idx_tuple))
                _, vmin, vmax = _prepare_display(raw, complex_mode, dr, log_scale)

                header = np.array([seq, w, h], dtype=np.uint32).tobytes()
                vminmax = np.array([vmin, vmax], dtype=np.float32).tobytes()
                await ws.send_bytes(header + vminmax + rgba.tobytes())

                # Removed inline ThreadPoolExecutor _prefetch block here!
                # This fixes the pywebview unresponsive closing issue entirely.
                # The _run_preload thread handles all caching safely in the background now.

    except Exception:
        pass
    finally:
        receiver_task.cancel()
        try:
            await receiver_task
        except asyncio.CancelledError:
            pass


@app.get("/clearcache")
def clear_cache():
    _raw_cache.clear()
    _rgba_cache.clear()
    _mosaic_cache.clear()
    return {"status": "ok"}


@app.post("/preload")
async def start_preload(request: Request):
    global _preload_gen
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

    _preload_gen += 1
    gen = _preload_gen
    threading.Thread(
        target=_run_preload,
        args=(
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


@app.get("/preload_status")
def get_preload_status():
    with _preload_lock:
        return {
            "done": _preload_done,
            "total": _preload_total,
            "skipped": _preload_skipped,
        }


@app.get("/metadata")
def get_metadata():
    return {"shape": list(SHAPE), "is_complex": bool(np.iscomplexobj(DATA))}


@app.get("/pixel")
def get_pixel(
    dim_x: int, dim_y: int, indices: str, px: int, py: int, complex_mode: int = 0
):
    idx_tuple = tuple(int(x) for x in indices.split(","))
    raw = extract_slice(dim_x, dim_y, list(idx_tuple))
    data = apply_complex_mode(raw, complex_mode)
    h, w = data.shape
    if 0 <= py < h and 0 <= px < w:
        val = float(data[py, px])
    else:
        val = float("nan")
    return {"value": val}


@app.get("/info")
def get_info():
    try:
        dtype_str = str(DATA.dtype)
    except AttributeError:
        dtype_str = "unknown"
    info: dict = {
        "shape": list(SHAPE),
        "dtype": dtype_str,
        "ndim": len(SHAPE),
        "total_elements": int(np.prod(SHAPE)),
        "is_complex": bool(np.iscomplexobj(DATA)),
        "filepath": _data_filepath,
    }
    try:
        info["size_mb"] = round(DATA.nbytes / 1024**2, 2)
    except AttributeError:
        info["size_mb"] = None
    if _fft_axes is not None:
        info["fft_axes"] = list(_fft_axes)
    return info


@app.post("/fft")
async def toggle_fft(request: Request):
    global DATA, SHAPE, _fft_original_data, _fft_axes
    body = await request.json()
    axes_str = str(body.get("axes", "")).strip()

    if _fft_original_data is not None:
        DATA = _fft_original_data
        SHAPE = DATA.shape
        _fft_original_data = None
        _fft_axes = None
        _raw_cache.clear()
        _rgba_cache.clear()
        _mosaic_cache.clear()
        compute_global_stats()
        return {"status": "restored", "is_complex": bool(np.iscomplexobj(DATA))}

    try:
        axes = tuple(int(a.strip()) for a in axes_str.split(",") if a.strip())
        if not axes:
            raise ValueError("No axes specified")
    except Exception as e:
        return {"error": str(e)}

    _fft_original_data = DATA
    full = np.array(DATA)
    DATA = np.fft.fftshift(np.fft.fftn(full, axes=axes), axes=axes)
    SHAPE = DATA.shape
    _fft_axes = axes
    _raw_cache.clear()
    _rgba_cache.clear()
    _mosaic_cache.clear()
    compute_global_stats()
    return {
        "status": "fft_applied",
        "axes": list(axes),
        "is_complex": bool(np.iscomplexobj(DATA)),
    }


@app.get("/slice")
def get_slice(
    dim_x: int,
    dim_y: int,
    indices: str,
    colormap: str = "gray",
    dr: int = 1,
    slice_dim: int = -1,
):
    idx_tuple = tuple(int(x) for x in indices.split(","))
    rgba = render_rgba(dim_x, dim_y, idx_tuple, colormap, dr)
    img = Image.fromarray(rgba[:, :, :3], mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return Response(
        content=buf.getvalue(),
        media_type="image/jpeg",
        headers={"Cache-Control": "max-age=300"},
    )


@app.get("/grid")
def get_grid(
    dim_x: int,
    dim_y: int,
    indices: str,
    slice_dim: int,
    colormap: str = "gray",
    dr: int = 1,
):
    idx_list = [int(x) for x in indices.split(",")]
    n = SHAPE[slice_dim]
    frames = []
    for i in range(n):
        idx_list[slice_dim] = i
        frames.append(extract_slice(dim_x, dim_y, idx_list))

    all_data = np.stack(frames)
    if dr in GLOBAL_STATS:
        vmin, vmax = GLOBAL_STATS[dr]
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


@app.get("/gif")
def get_gif(
    dim_x: int,
    dim_y: int,
    indices: str,
    slice_dim: int,
    colormap: str = "gray",
    dr: int = 1,
):
    idx_list = [int(x) for x in indices.split(",")]
    n = SHAPE[slice_dim]
    frames = []
    for i in range(n):
        idx_list[slice_dim] = i
        frames.append(extract_slice(dim_x, dim_y, idx_list))

    all_data = np.stack(frames)
    if dr in GLOBAL_STATS:
        vmin, vmax = GLOBAL_STATS[dr]
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


@app.get("/")
def get_ui():
    html_content = (
        """<!DOCTYPE html>
<html>
<head>
    <title>ArrayView</title>
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
        
        /* viewer-row expands to fill available space, with auto scrolling for when image exceeds zoom */
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

        // Dynamically compute the maximum zoom scale so the image perfectly fits the available window space
        function getBaseScale(w, h) {
            const row = document.getElementById('viewer-row');
            // Allow 40px of breathing room minus the viewer bounds
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
        
        // Auto resize image smoothly if the user stretches the window manually
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
            ws = new WebSocket(`${proto}//${location.host}/ws`);
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

                if (seq !== wsSentSeq) return;

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
            };

            ws.onclose = () => {
                wsReady = false;
                setStatus('WebSocket closed — reconnecting...');
                setTimeout(initWebSocket, 1000);
            };

            ws.onerror = () => ws.close();
        }

        function triggerPreload() {
            if (shape.length < 3) return;
            preloadActiveDim = current_slice_dim;
            fetch('/preload', {
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
            fetch('/preload_status').then(r => r.json()).then(data => {
                const el = document.getElementById('preload-status');
                if (data.skipped) {
                    el.textContent = 'Array too large for full preload (>500 MB) — using prefetch.';
                    clearInterval(preloadPolling); preloadPolling = null;
                } else if (data.total > 0 && data.done >= data.total) {
                    el.textContent = '';
                    clearInterval(preloadPolling); preloadPolling = null;
                } else if (data.total > 0) {
                    const pct = Math.round(data.done / data.total * 100);
                    el.textContent = `Preloading dim ${preloadActiveDim}: ${data.done}/${data.total} (${pct}%)`;
                }
            });
        }

        async function init() {
            const res = await fetch('/metadata');
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
            const url = `/gif?dim_x=${dim_x}&dim_y=${dim_y}&indices=${indices.join(',')}&colormap=${COLORMAPS[colormap_idx]}&dr=${dr_idx}&slice_dim=${current_slice_dim}`;
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

        canvas.addEventListener('click', () => sink.focus());
        document.addEventListener('click', () => sink.focus());

        canvas.addEventListener('mousemove', (e) => {
            if (dim_z >= 0) return;
            if (pixelHoverPending) return;
            pixelHoverPending = true;
            setTimeout(() => { pixelHoverPending = false; }, 50);
            const rect = canvas.getBoundingClientRect();
            const px = Math.floor((e.clientX - rect.left) * canvas.width / rect.width);
            const py = Math.floor((e.clientY - rect.top) * canvas.height / rect.height);
            fetch(`/pixel?dim_x=${dim_x}&dim_y=${dim_y}&indices=${indices.join(',')}&px=${px}&py=${py}&complex_mode=${complexMode}`)
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
                fetch('/clearcache'); updateView(); triggerPreload();
                showToast(`colormap: ${COLORMAPS[colormap_idx]}`);
            } else if (e.key === 'd') {
                dr_idx = (dr_idx + 1) % DR_LABELS.length;
                fetch('/clearcache'); updateView(); triggerPreload();
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
                fetch('/clearcache'); updateView(); triggerPreload();
                showToast(logScale ? 'log scale: on' : 'log scale: off');
            } else if (e.key === 'i') {
                fetch('/info').then(r => r.json()).then(d => {
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
                    fetch('/fft', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({axes: ''})})
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
                    fetch('/fft', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({axes: axesStr})})
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
                activeDim = current_slice_dim; dim_z = -1;
                updateView(); triggerPreload();
            } else if (e.key === 'y') {
                if (shape.length < 3) return;
                [dim_y, current_slice_dim] = [current_slice_dim, dim_y];
                activeDim = current_slice_dim; dim_z = -1;
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
_jupyter_server_port: int | None = None  # port of the running background server


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
    """Block until the local TCP port is accepting connections."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                return
        except OSError:
            time.sleep(0.05)


async def _serve_background(port: int):
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    await server.serve()


def _set_data(data):
    """Update the global DATA/SHAPE and flush all caches."""
    global DATA, SHAPE, _data_filepath, _fft_original_data, _fft_axes
    if isinstance(data, str):
        _data_filepath = data
        data = load_data(data)
    else:
        _data_filepath = None
    _fft_original_data = None
    _fft_axes = None
    DATA = data
    SHAPE = data.shape
    _raw_cache.clear()
    _rgba_cache.clear()
    _mosaic_cache.clear()
    compute_global_stats()


def view(
    data,
    port: int = 8123,
    inline: bool | None = None,
    height: int = 500,
    window: bool = True,
):
    global _jupyter_server_port

    _set_data(data)

    # Use a fixed initial window size as requested
    win_w = 1200
    win_h = 800

    is_jupyter = _in_jupyter()
    if inline is None:
        inline = is_jupyter

    if window:
        inline = False

    url = f"http://127.0.0.1:{port}"

    if inline:
        if _jupyter_server_port != port:
            threading.Thread(
                target=lambda: asyncio.run(_serve_background(port)),
                daemon=True,
            ).start()
            _wait_for_port(port)
            _jupyter_server_port = port

        from IPython.display import IFrame

        return IFrame(src=url, width="100%", height=height)
    else:

        def on_closing():
            """Ensure background computations safely halt when window closes."""
            global _preload_gen
            _preload_gen += 1

        if is_jupyter:
            if window:
                try:
                    import webview
                except Exception:
                    print("pywebview not installed; falling back to browser tab.")
                    threading.Timer(0.5, lambda: webbrowser.open(url)).start()
                    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
                    return

                threading.Thread(
                    target=lambda: asyncio.run(_serve_background(port)),
                    daemon=True,
                ).start()
                _wait_for_port(port)
                try:
                    native_win = webview.create_window(
                        "ArrayView",
                        url,
                        width=win_w,
                        height=win_h,
                        background_color="#111111",
                    )
                    native_win.events.closing += on_closing
                    webview.start()
                except Exception as e:
                    print(
                        "pywebview failed to open window, falling back to browser:", e
                    )
                    threading.Timer(0.5, lambda: webbrowser.open(url)).start()
                    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
            else:
                threading.Timer(0.5, lambda: webbrowser.open(url)).start()
                uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
        else:
            try:
                import webview
            except Exception:
                print("pywebview not installed; falling back to browser tab.")
                threading.Timer(0.5, lambda: webbrowser.open(url)).start()
                uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
                return

            threading.Thread(
                target=lambda: asyncio.run(_serve_background(port)),
                daemon=True,
            ).start()
            _wait_for_port(port)
            try:
                native_win = webview.create_window(
                    "ArrayView",
                    url,
                    width=win_w,
                    height=win_h,
                    background_color="#111111",
                )
                native_win.events.closing += on_closing
                webview.start()
            except Exception as e:
                print("pywebview failed to open window, falling back to browser:", e)
                threading.Timer(0.5, lambda: webbrowser.open(url)).start()
                uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


def arrayview():
    global DATA, SHAPE, _data_filepath

    parser = argparse.ArgumentParser(description="Lightning Fast ND Array Viewer")
    parser.add_argument("file", help="Path to .npy, .nii/.nii.gz, or .zarr file")
    parser.add_argument("--port", type=int, default=8000, help="Port to serve on")
    args = parser.parse_args()

    try:
        DATA = load_data(args.file)
        SHAPE = DATA.shape
        _data_filepath = args.file
        try:
            size_str = f" ({DATA.nbytes // 1024**2} MB)"
        except AttributeError:
            size_str = ""
        print(f"Loaded {args.file} with shape {SHAPE}{size_str}")
        compute_global_stats()
        print(f"Open http://127.0.0.1:{args.port} in your browser")
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

    url = f"http://127.0.0.1:{args.port}"
    threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
