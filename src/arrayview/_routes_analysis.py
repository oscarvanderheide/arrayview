import itertools
import math

import numpy as np
from fastapi import Body, Depends, Response

from arrayview._analysis import (
    _lebesgue_slice,
    _pixel_value,
    _safe_float,
    _slice_histogram,
    _volume_histogram,
)
from arrayview._render import apply_complex_mode, extract_slice


def register_analysis_routes(app, get_session_or_404) -> None:
    @app.get("/pixel/{sid}")
    def get_pixel(
        sid: str,
        dim_x: int,
        dim_y: int,
        indices: str,
        px: int,
        py: int,
        complex_mode: int = 0,
        session=Depends(get_session_or_404),
    ):
        return {
            "value": _pixel_value(
                session, dim_x, dim_y, indices, px, py, complex_mode
            )
        }

    @app.post("/roi_freehand/{sid}")
    def get_roi_freehand(
        sid: str,
        dim_x: int,
        dim_y: int,
        indices: str,
        complex_mode: int = 0,
        body: dict = Body(...),
        session=Depends(get_session_or_404),
    ):
        if session.rgb_axis is not None:
            return {"error": "not supported for RGB sessions"}
        points = body.get("points", [])
        if len(points) < 3:
            return {"error": "need at least 3 points"}
        idx_tuple = tuple(int(v) for v in indices.split(","))
        raw = extract_slice(session, dim_x, dim_y, list(idx_tuple))
        data = apply_complex_mode(raw, complex_mode)
        h, w = data.shape
        from PIL import Image as _PILImage, ImageDraw as _PILDraw

        mask_img = _PILImage.new("L", (w, h), 0)
        _PILDraw.Draw(mask_img).polygon([(p[0], p[1]) for p in points], fill=255)
        mask = np.array(mask_img) > 0
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
        session=Depends(get_session_or_404),
    ):
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
        session=Depends(get_session_or_404),
    ):
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

    @app.get("/roi_multi/{sid}")
    def get_roi_multi(
        sid: str,
        dim_x: int,
        dim_y: int,
        indices: str,
        x0: int = 0,
        y0: int = 0,
        x1: int = 0,
        y1: int = 0,
        complex_mode: int = 0,
        session=Depends(get_session_or_404),
    ):
        if session.rgb_axis is not None:
            return {"error": "not supported for RGB sessions"}
        arr = session.data
        ndim = arr.ndim
        idx_list = [int(v) for v in indices.split(",")]

        roi_dims = {dim_x, dim_y}
        other_dims = [d for d in range(ndim) if d not in roi_dims]

        def extract_roi(data_2d):
            h, w = data_2d.shape
            xa, xb = max(0, min(x0, x1, w - 1)), min(w, max(x0, x1) + 1)
            ya, yb = max(0, min(y0, y1, h - 1)), min(h, max(y0, y1) + 1)
            roi = data_2d[ya:yb, xa:xb]
            if roi.size == 0:
                return np.array([])
            return roi[np.isfinite(roi)]

        rows = []

        bitmask = ["0"] * ndim
        bitmask[dim_x] = "1"
        bitmask[dim_y] = "1"
        base_slice = extract_slice(session, dim_x, dim_y, idx_list)
        base_data = apply_complex_mode(base_slice, complex_mode)
        finite = extract_roi(base_data)
        if finite.size:
            rows.append(
                {
                    "dims": "".join(bitmask),
                    "min": _safe_float(finite.min()),
                    "max": _safe_float(finite.max()),
                    "mean": _safe_float(finite.mean()),
                    "std": _safe_float(finite.std()),
                    "n": int(finite.size),
                }
            )

        for ext_dim in other_dims:
            bitmask_ext = list(bitmask)
            bitmask_ext[ext_dim] = "1"
            all_finite = []
            for val in range(arr.shape[ext_dim]):
                idx_copy = list(idx_list)
                idx_copy[ext_dim] = val
                sl = extract_slice(session, dim_x, dim_y, idx_copy)
                data = apply_complex_mode(sl, complex_mode)
                finite = extract_roi(data)
                if finite.size:
                    all_finite.append(finite)
            if all_finite:
                combined = np.concatenate(all_finite)
                rows.append(
                    {
                        "dims": "".join(bitmask_ext),
                        "min": _safe_float(combined.min()),
                        "max": _safe_float(combined.max()),
                        "mean": _safe_float(combined.mean()),
                        "std": _safe_float(combined.std()),
                        "n": int(combined.size),
                    }
                )

        if len(other_dims) > 1:
            bitmask_all = ["1"] * ndim
            all_finite = []
            ranges = [range(arr.shape[d]) for d in other_dims]
            for combo in itertools.product(*ranges):
                idx_copy = list(idx_list)
                for d, val in zip(other_dims, combo):
                    idx_copy[d] = val
                sl = extract_slice(session, dim_x, dim_y, idx_copy)
                data = apply_complex_mode(sl, complex_mode)
                finite = extract_roi(data)
                if finite.size:
                    all_finite.append(finite)
            if all_finite:
                combined = np.concatenate(all_finite)
                rows.append(
                    {
                        "dims": "".join(bitmask_all),
                        "min": _safe_float(combined.min()),
                        "max": _safe_float(combined.max()),
                        "mean": _safe_float(combined.mean()),
                        "std": _safe_float(combined.std()),
                        "n": int(combined.size),
                    }
                )

        return {"rows": rows}

    @app.get("/roi_floodfill/{sid}")
    def get_roi_floodfill(
        sid: str,
        dim_x: int,
        dim_y: int,
        indices: str,
        px: int,
        py: int,
        tolerance: float = 0.1,
        complex_mode: int = 0,
        session=Depends(get_session_or_404),
    ):
        if session.rgb_axis is not None:
            return {"error": "not supported for RGB sessions"}
        idx_tuple = tuple(int(v) for v in indices.split(","))
        raw = extract_slice(session, dim_x, dim_y, list(idx_tuple))
        data = apply_complex_mode(raw, complex_mode)
        h, w = data.shape
        if not (0 <= py < h and 0 <= px < w):
            return {"error": "seed out of bounds"}
        seed_val = float(data[py, px])
        try:
            from scipy.ndimage import label

            abs_tol = tolerance * (
                np.nanmax(np.abs(data)) - np.nanmin(np.abs(data)) + 1e-10
            )
            mask = np.abs(data - seed_val) <= abs_tol
            labeled, _n_features = label(mask)
            seed_label = labeled[py, px]
            if seed_label == 0:
                return {"error": "seed outside tolerance region"}
            component = labeled == seed_label
        except ImportError:
            abs_tol = tolerance * (
                np.nanmax(np.abs(data)) - np.nanmin(np.abs(data)) + 1e-10
            )
            component = np.zeros((h, w), dtype=bool)
            stack = [(py, px)]
            component[py, px] = True
            while stack:
                cy, cx = stack.pop()
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and not component[ny, nx]:
                        if abs(float(data[ny, nx]) - seed_val) <= abs_tol:
                            component[ny, nx] = True
                            stack.append((ny, nx))
        roi = data[component]
        finite = roi[np.isfinite(roi)]
        ys, xs = np.where(component)
        bbox = (
            {
                "x0": int(xs.min()),
                "y0": int(ys.min()),
                "x1": int(xs.max()),
                "y1": int(ys.max()),
            }
            if len(xs) > 0
            else {"x0": 0, "y0": 0, "x1": 0, "y1": 0}
        )
        return {
            "min": _safe_float(finite.min()) if finite.size else None,
            "max": _safe_float(finite.max()) if finite.size else None,
            "mean": _safe_float(finite.mean()) if finite.size else None,
            "std": _safe_float(finite.std()) if finite.size else None,
            "n": int(finite.size),
            "seed_value": _safe_float(seed_val),
            "tolerance": tolerance,
            "bbox": bbox,
            "mask_b64": _encode_mask_b64(component, bbox),
        }

    @app.get("/line_profile/{sid}")
    def get_line_profile(
        sid: str,
        dim_x: int,
        dim_y: int,
        indices: str,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
        complex_mode: int = 0,
        log_scale: bool = False,
        session=Depends(get_session_or_404),
    ):
        idx_tuple = tuple(int(v) for v in indices.split(","))
        raw = extract_slice(session, dim_x, dim_y, list(idx_tuple))
        data = apply_complex_mode(raw, complex_mode)
        if log_scale:
            data = np.where(data > 0, np.log10(data), 0.0)
        h, w = data.shape
        n_samples = 200
        xs = np.linspace(float(x0), float(x1), n_samples)
        ys = np.linspace(float(y0), float(y1), n_samples)
        xi = np.clip(xs.astype(int), 0, w - 1)
        yi = np.clip(ys.astype(int), 0, h - 1)
        values = data[yi, xi]
        distance = float(math.hypot(x1 - x0, y1 - y0))
        return {
            "values": [_safe_float(v) for v in values],
            "distance": distance,
        }

    @app.get("/histogram/{sid}")
    def get_histogram(
        sid: str,
        dim_x: int,
        dim_y: int,
        indices: str,
        complex_mode: int = 0,
        bins: int = 128,
        session=Depends(get_session_or_404),
    ):
        return _slice_histogram(session, dim_x, dim_y, indices, complex_mode, bins)

    @app.get("/volume-histogram/{sid}")
    def get_volume_histogram(
        sid: str,
        dim_x: int,
        dim_y: int,
        scroll_dim: int = -1,
        scroll_dims: str = "",
        fixed_indices: str = "",
        complex_mode: int = 0,
        bins: int = 64,
        session=Depends(get_session_or_404),
    ):
        return _volume_histogram(
            session,
            dim_x,
            dim_y,
            scroll_dim,
            scroll_dims,
            fixed_indices,
            complex_mode,
            bins,
        )

    @app.get("/volume_data/{sid}")
    def get_volume_data(
        sid: str,
        dims: str = "",
        indices: str = "",
        complex_mode: int = 0,
        session=Depends(get_session_or_404),
    ):
        if not dims:
            return Response(status_code=400, content="dims required")
        dim_list = [int(d) for d in dims.split(",")]
        if len(dim_list) != 3:
            return Response(status_code=400, content="dims must be exactly 3 integers")

        idx_list = (
            [int(x) for x in indices.split(",")]
            if indices
            else [s // 2 for s in session.shape]
        )

        slicer = []
        for i in range(len(session.shape)):
            if i in dim_list:
                slicer.append(slice(None))
            else:
                idx = idx_list[i] if i < len(idx_list) else session.shape[i] // 2
                slicer.append(min(max(idx, 0), session.shape[i] - 1))
        vol = np.array(session.data[tuple(slicer)])

        free_axes_sorted = sorted(dim_list)
        perm = [free_axes_sorted.index(d) for d in dim_list]
        vol = np.transpose(vol, perm)
        vol = apply_complex_mode(vol, complex_mode)

        max_dim = 256
        strides = []
        for s in vol.shape:
            strides.append(max(1, (s + max_dim - 1) // max_dim))
        if any(st > 1 for st in strides):
            vol = vol[::strides[0], ::strides[1], ::strides[2]]

        vol = np.ascontiguousarray(vol, dtype=np.float32)

        finite = vol[np.isfinite(vol)]
        if finite.size > 0:
            vmin = float(np.percentile(finite, 1))
            vmax = float(np.percentile(finite, 99))
        else:
            vmin, vmax = 0.0, 1.0

        return Response(
            content=vol.tobytes(),
            media_type="application/octet-stream",
            headers={
                "X-Shape": ",".join(str(s) for s in vol.shape),
                "X-Vmin": str(vmin),
                "X-Vmax": str(vmax),
            },
        )

    @app.get("/lebesgue/{sid}")
    def get_lebesgue_slice(
        sid: str,
        dim_x: int,
        dim_y: int,
        indices: str,
        complex_mode: int = 0,
        log_scale: bool = False,
        session=Depends(get_session_or_404),
    ):
        data = _lebesgue_slice(session, dim_x, dim_y, indices, complex_mode, log_scale)
        return Response(
            content=data.tobytes(),
            media_type="application/octet-stream",
            headers={
                "X-ArrayView-Width": str(data.shape[1]),
                "X-ArrayView-Height": str(data.shape[0]),
                "Cache-Control": "no-cache",
            },
        )


def _encode_mask_b64(mask: np.ndarray, bbox: dict) -> str:
    import base64

    x0, y0 = bbox["x0"], bbox["y0"]
    x1, y1 = bbox["x1"], bbox["y1"]
    sub = mask[y0 : y1 + 1, x0 : x1 + 1].astype(np.uint8)
    return base64.b64encode(sub.tobytes()).decode("ascii")
