import base64
import io
import itertools
import math
import os
import threading

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
        qmri_role: str = "",
        session=Depends(get_session_or_404),
    ):
        return {
            "value": _pixel_value(
                session, dim_x, dim_y, indices, px, py, complex_mode, qmri_role
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

    @app.get("/roi_floodfill_warm/{sid}")
    def get_roi_floodfill_warm(
        sid: str,
        dim_x: int,
        dim_y: int,
        indices: str,
        scope_dim: int,
        complex_mode: int = 0,
        smooth: float = 0.0,
        session=Depends(get_session_or_404),
    ):
        """Gather (and smooth) the 3D wand volume while the cursor hovers."""
        if session.rgb_axis is not None:
            return {"ok": False}
        shape = session.shape
        idx = [int(v) for v in indices.split(",")][: len(shape)]
        idx += [s // 2 for s in shape[len(idx):]]
        idx = [min(max(v, 0), shape[i] - 1) for i, v in enumerate(idx)]
        if scope_dim in {dim_x, dim_y} or not (0 <= scope_dim < len(shape)):
            return {"ok": False}
        volume, _r = _scoped_floodfill_volume(session, dim_x, dim_y, idx, complex_mode, int(scope_dim))
        smooth = round(max(0.0, min(_FLOODFILL_SMOOTH_MAX, float(smooth))), 2)
        if smooth > 0:
            # Whole volume: every box the grow asks for is then a slice of it.
            _blurred_box(volume, smooth, [0, 0, 0], list(volume.shape))
        return {"ok": True}

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
        scope_dim: int | None = None,
        preview: bool = False,
        smooth: float = 0.0,
        session=Depends(get_session_or_404),
    ):
        smooth = round(max(0.0, min(_FLOODFILL_SMOOTH_MAX, float(smooth))), 2)
        if session.rgb_axis is not None:
            return {"error": "not supported for RGB sessions"}
        idx_tuple = tuple(int(v) for v in indices.split(","))
        if scope_dim is not None:
            return _scoped_floodfill_response(
                session,
                dim_x,
                dim_y,
                list(idx_tuple),
                px,
                py,
                tolerance,
                complex_mode,
                scope_dim,
                preview=preview,
                smooth=smooth,
            )
        raw = extract_slice(session, dim_x, dim_y, list(idx_tuple))
        data = apply_complex_mode(raw, complex_mode)
        h, w = data.shape
        if not (0 <= py < h and 0 <= px < w):
            return {"error": "seed out of bounds"}
        abs_tol = tolerance * (
            np.nanmax(np.abs(data)) - np.nanmin(np.abs(data)) + 1e-10
        )
        component, seed_val, _box, _crop = _grow_floodfill(data[None], (0, py, px), abs_tol, smooth)
        component = component[0]
        if not component.any():
            return {"error": "seed outside tolerance region"}
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
            "smooth": smooth,
            "abs_tol": _safe_float(abs_tol),
            "bbox": bbox,
            "mask_b64": _encode_mask_b64(component, bbox),
        }

    @app.post("/roi_stats/{sid}")
    def get_roi_stats_structured(
        sid: str,
        body: dict = Body(...),
        session=Depends(get_session_or_404),
    ):
        if session.rgb_axis is not None:
            return {"error": "not supported for RGB sessions"}
        dim_x = int(body.get("dim_x", 1 if session.data.ndim > 1 else 0))
        dim_y = int(body.get("dim_y", 0))
        indices = _roi_indices_from_body(body, session.data.ndim, session.shape)
        complex_mode = int(body.get("complex_mode", 0))
        rois = body.get("rois", [])
        if not isinstance(rois, list):
            return {"error": "rois must be a list"}

        results = []
        for roi_idx, roi in enumerate(rois):
            rows = []
            values = []
            for idx in _roi_scope_indices(roi, indices, session.shape, {dim_x, dim_y}):
                raw = extract_slice(session, dim_x, dim_y, idx)
                data = apply_complex_mode(raw, complex_mode)
                mask = _roi_mask_for_shape(roi, data.shape, idx)
                finite = _roi_finite_values(data, mask)
                if finite.size:
                    values.append(finite)
                    rows.append(
                        {
                            "indices": list(idx),
                            "dims": _roi_scope_dims_string(idx, indices, {dim_x, dim_y}),
                            "min": _safe_float(finite.min()),
                            "max": _safe_float(finite.max()),
                            "mean": _safe_float(finite.mean()),
                            "std": _safe_float(finite.std()),
                            "n": int(finite.size),
                        }
                    )
            combined = _combine_roi_rows(rows)
            if combined is not None:
                combined["hist"] = _roi_value_histogram(values, combined["min"], combined["max"])
            results.append(
                {
                    "roi": roi.get("id", roi_idx),
                    "name": roi.get("name", f"ROI {roi_idx + 1}"),
                    "shape": roi.get("type", "rect"),
                    "rows": rows,
                    "stats": combined,
                }
            )
        return {"results": results}

    @app.post("/roi_stats_diff/{sid_a}/{sid_b}")
    def get_roi_stats_diff(sid_a: str, sid_b: str, body: dict = Body(...)):
        """ROI numbers on the side-by-side difference image (A−B, |A−B|,
        |A−B|/|A|) at the current position. Same payload as /roi_stats plus
        diff_mode and optional indices_a / indices_b."""
        from arrayview._diff import _compute_diff
        from arrayview._session import SESSIONS

        session_a, session_b = SESSIONS.get(sid_a), SESSIONS.get(sid_b)
        if not session_a or not session_b:
            return Response(status_code=404)
        dim_x = int(body.get("dim_x", 1))
        dim_y = int(body.get("dim_y", 0))
        indices = _roi_indices_from_body(body, session_a.data.ndim, session_a.shape)
        join = lambda v: ",".join(str(int(i)) for i in v) if v else None  # noqa: E731
        try:
            raw = _compute_diff(
                session_a, session_b, dim_x, dim_y, join(indices), -1, 0,
                int(body.get("complex_mode", 0)), False, int(body.get("diff_mode", 1)),
                indices_a=join(body.get("indices_a")), indices_b=join(body.get("indices_b")),
            )[0]
        except Exception:
            return Response(status_code=422)
        results = []
        for roi_idx, roi in enumerate(body.get("rois") or []):
            mask = _roi_mask_for_shape(roi, raw.shape, tuple(indices))
            finite = _roi_finite_values(raw, mask)
            stats = None
            if finite.size:
                stats = {
                    "min": _safe_float(finite.min()),
                    "max": _safe_float(finite.max()),
                    "mean": _safe_float(finite.mean()),
                    "std": _safe_float(finite.std()),
                    "n": int(finite.size),
                }
                stats["hist"] = _roi_value_histogram([finite], stats["min"], stats["max"])
            results.append({"roi": roi.get("id", roi_idx), "stats": stats})
        return {"results": results}

    @app.post("/roi_mask/{sid}")
    def export_roi_mask_structured(
        sid: str,
        body: dict = Body(...),
        session=Depends(get_session_or_404),
    ):
        if session.rgb_axis is not None:
            return Response(status_code=400, content="not supported for RGB sessions")
        dim_x = int(body.get("dim_x", 1 if session.data.ndim > 1 else 0))
        dim_y = int(body.get("dim_y", 0))
        indices = _roi_indices_from_body(body, session.data.ndim, session.shape)
        rois = body.get("rois", [])
        if not isinstance(rois, list):
            return Response(status_code=400, content="rois must be a list")

        labels = np.zeros(session.shape, dtype=np.uint16)
        for roi_idx, roi in enumerate(rois):
            label = roi_idx + 1
            for idx in _roi_scope_indices(roi, indices, session.shape, {dim_x, dim_y}):
                raw = extract_slice(session, dim_x, dim_y, idx)
                mask = _roi_mask_for_shape(roi, raw.shape, idx)
                _write_roi_mask(labels, mask, idx, dim_x, dim_y, label)

        buf = io.BytesIO()
        np.save(buf, labels)
        return Response(
            content=buf.getvalue(),
            media_type="application/octet-stream",
            headers={"Content-Disposition": "attachment; filename=roi_mask.npy"},
        )

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
        qmri_role: str = "",
        session=Depends(get_session_or_404),
    ):
        return _slice_histogram(
            session, dim_x, dim_y, indices, complex_mode, bins, qmri_role
        )

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
        qmri_role: str = "",
        exclude_zeros: int = 0,
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
            qmri_role,
            exclude_zeros=bool(exclude_zeros),
        )

    @app.get("/volume_data/{sid}")
    def get_volume_data(
        sid: str,
        dims: str = "",
        indices: str = "",
        complex_mode: int = 0,
        log_scale: bool = False,
        session=Depends(get_session_or_404),
    ):
        if not dims:
            return Response(status_code=400, content="dims required")
        dim_list = [int(d) for d in dims.split(",")]
        if len(dim_list) != 3:
            return Response(status_code=400, content="dims must be exactly 3 integers")
        if session.rgb_axis is not None:
            return Response(status_code=400, content="RGB data is not supported")

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
        if log_scale:
            vol = np.log1p(np.abs(vol)).astype(np.float32)

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

        voxel_sizes = tuple(
            (getattr(session, "spatial_meta", None) or {}).get(
                "voxel_sizes", (1.0,) * len(session.shape)
            )
        )
        volume_spacing = [
            float(voxel_sizes[d]) if d < len(voxel_sizes) else 1.0
            for d in dim_list
        ]

        return Response(
            content=vol.tobytes(),
            media_type="application/octet-stream",
            headers={
                "X-Shape": ",".join(str(s) for s in vol.shape),
                "X-Vmin": str(vmin),
                "X-Vmax": str(vmax),
                "X-Spacing": ",".join(str(v) for v in volume_spacing),
                "X-Downsample": ",".join(str(int(v)) for v in strides),
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
        qmri_role: str = "",
        session=Depends(get_session_or_404),
    ):
        data = _lebesgue_slice(session, dim_x, dim_y, indices, complex_mode, log_scale, qmri_role)
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
    x0, y0 = bbox["x0"], bbox["y0"]
    x1, y1 = bbox["x1"], bbox["y1"]
    sub = mask[y0 : y1 + 1, x0 : x1 + 1]
    # Bit-packed, lowest bit first. Readers tell packed from one-byte-per-pixel
    # masks by length (see _decode_mask_bytes).
    return base64.b64encode(
        np.packbits(sub, axis=None, bitorder="little").tobytes()
    ).decode("ascii")


def _decode_mask_bytes(encoded: str, n: int) -> np.ndarray:
    """Decode a base64 mask of *n* pixels, packed or one byte per pixel."""
    raw = np.frombuffer(base64.b64decode(encoded), dtype=np.uint8)
    if raw.size != n and raw.size == (n + 7) // 8:
        return np.unpackbits(raw, count=n, bitorder="little").astype(bool)
    return raw.astype(bool)


# Last stacked volume a scoped flood fill grew through. Dragging the
# sensitivity re-runs the fill many times on the same volume; restacking it and
# rescanning its value range each step is wasted work. One entry only.
_FLOODFILL_VOLUME_CACHE: dict = {}
_FLOODFILL_VOLUME_LOCK = threading.Lock()
# Last scoped grow, so the release request reuses the region the last drag
# step already grew instead of growing it again (the grow dominates release).
_FLOODFILL_GROW_CACHE: dict = {}
# Recent smoothed copies of the image: (volume, smooth, lo, hi, blurred).
# Dragging only the sensitivity reuses them; a box inside one is a slice of it.
_FLOODFILL_BLUR_CACHE: dict = {}


def _blurred_box(volume, smooth, elo, ehi):
    """In-plane blur of volume[elo:ehi] (sigma smooth/2), reusing a recent one.

    Two entries: the seed-plane sizing pass and the 3D box would evict each
    other from a single slot on every sensitivity step.
    """
    entries = _FLOODFILL_BLUR_CACHE.get("entries", [])
    for vol, sm, lo, hi, arr in entries:
        if vol is volume and sm == smooth and all(lo[i] <= elo[i] and ehi[i] <= hi[i] for i in range(3)):
            return arr[
                elo[0] - lo[0]: ehi[0] - lo[0],
                elo[1] - lo[1]: ehi[1] - lo[1],
                elo[2] - lo[2]: ehi[2] - lo[2],
            ]
    sub = volume[elo[0]:ehi[0], elo[1]:ehi[1], elo[2]:ehi[2]]
    out = _blur_planes(sub, smooth / 2.0)
    _FLOODFILL_BLUR_CACHE["entries"] = [(volume, smooth, list(elo), list(ehi), out)] + entries[:1]
    return out


# Edge smoothing for the magic wand, a continuous level s in 0 (off) ..
# _FLOODFILL_SMOOTH_MAX. Compare a copy of the image blurred in-plane with
# sigma s/2, then blur the grown mask in-plane with sigma s, keep > 0.5, keep
# the seed's piece and fill in-plane holes. The viewer repeats this exactly for
# its instant single-slice preview (_roiLocalFloodFill) — keep the two in step.
_FLOODFILL_SMOOTH_MAX = 5.0


_FLOODFILL_POOL = None


def _by_plane_chunks(fn, arr: np.ndarray) -> np.ndarray:
    """fn applied to z-chunks of a [z, y, x] array on a few threads (scipy
    releases the GIL). Only for per-plane work: the result equals fn(arr)."""
    global _FLOODFILL_POOL
    nt = min(4, os.cpu_count() or 1, arr.shape[0])
    if nt < 2 or arr.size < 1_000_000:
        return fn(arr)
    if _FLOODFILL_POOL is None:
        from concurrent.futures import ThreadPoolExecutor

        _FLOODFILL_POOL = ThreadPoolExecutor(4, thread_name_prefix="av-floodfill")
    bounds = np.linspace(0, arr.shape[0], nt + 1).astype(int)
    parts = list(_FLOODFILL_POOL.map(lambda i: fn(arr[bounds[i]:bounds[i + 1]]), range(nt)))
    return np.concatenate(parts, axis=0)


def _blur_planes(values: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur of each [y, x] plane of a [z, y, x] array; NaNs ignored."""
    return _by_plane_chunks(lambda a: _blur_planes_one(a, sigma), values)


def _blur_planes_one(values: np.ndarray, sigma: float) -> np.ndarray:
    from scipy.ndimage import gaussian_filter

    sig = (0, sigma, sigma)
    valid = np.isfinite(values)
    if valid.all():
        return gaussian_filter(values.astype(np.float32), sig, mode="reflect", truncate=4.0)
    num = gaussian_filter(np.where(valid, values, 0).astype(np.float32), sig, mode="reflect", truncate=4.0)
    den = gaussian_filter(valid.astype(np.float32), sig, mode="reflect", truncate=4.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    out[~valid] = np.nan
    return out


def _grow_in_box(volume, seed, abs_tol, smooth, lo, hi):
    """Grow inside box [lo, hi). Returns (crop mask, seed value, axes hit).

    *axes hit* lists the axes where the region could reach past an inner box
    face (one that is not the volume edge): then the box result may differ
    from the whole-volume one and the caller must retry with a bigger box.
    """
    shape = volume.shape
    if smooth > 0:
        r_img = int(4.0 * (smooth / 2.0) + 0.5)
        r_mask = int(4.0 * smooth + 0.5)
    else:
        r_img = r_mask = 0
    # The image blur needs r_img of real neighbours in-plane around the box.
    elo = [lo[0]] + [max(0, lo[i] - r_img) for i in (1, 2)]
    ehi = [hi[0]] + [min(shape[i], hi[i] + r_img) for i in (1, 2)]
    img = _blurred_box(volume, smooth, elo, ehi) if smooth > 0 else volume[elo[0]:ehi[0], elo[1]:ehi[1], elo[2]:ehi[2]]
    img = img[
        :,
        lo[1] - elo[1]: lo[1] - elo[1] + (hi[1] - lo[1]),
        lo[2] - elo[2]: lo[2] - elo[2] + (hi[2] - lo[2]),
    ]
    local = tuple(seed[i] - lo[i] for i in range(3))
    seed_val = float(img[local])
    grown = _connected_component(np.abs(img - seed_val) <= abs_tol, local)
    # Clearance from inner faces: one voxel along the stack (the grow could
    # step through), r_mask + 1 in-plane (the mask blur must see only zeros).
    hit = []
    for axis, margin in ((0, 1), (1, r_mask + 1), (2, r_mask + 1)):
        n = grown.shape[axis]
        if (lo[axis] > 0 and np.take(grown, range(min(margin, n)), axis=axis).any()) or (
            hi[axis] < shape[axis] and np.take(grown, range(max(0, n - margin), n), axis=axis).any()
        ):
            hit.append(axis)
    if hit or smooth <= 0 or not grown.any():
        return grown, seed_val, hit
    # Round the edge only around the region: its bounding box plus the mask
    # blur reach (outside it the blurred mask is exactly zero).
    zz, yy, xx = (np.flatnonzero(grown.any(axis=a)) for a in ((1, 2), (0, 2), (0, 1)))
    sub = (
        slice(zz[0], zz[-1] + 1),
        slice(max(0, yy[0] - r_mask - 1), min(grown.shape[1], yy[-1] + r_mask + 2)),
        slice(max(0, xx[0] - r_mask - 1), min(grown.shape[2], xx[-1] + r_mask + 2)),
    )
    sub_seed = tuple(local[i] - sub[i].start for i in range(3))
    rounded = _blur_planes(grown[sub].astype(np.float32), float(smooth)) > 0.5
    rounded = _connected_component(rounded, sub_seed)
    if not rounded.any():
        return grown, seed_val, hit
    rounded = _fill_holes_in_plane(rounded)
    out = np.zeros_like(grown)
    out[sub] = rounded
    return out, seed_val, hit


def _fill_holes_in_plane(mask):
    """binary_fill_holes on every [y, x] plane of a [z, y, x] mask."""
    return _by_plane_chunks(_fill_holes_in_plane_one, mask)


def _fill_holes_in_plane_one(mask):
    """binary_fill_holes on every [y, x] plane of a [z, y, x] mask, in one pass.

    Background is labelled with 4-connectivity within each plane only; pieces
    that touch a plane's y/x border are outside, the rest are holes.
    """
    from scipy.ndimage import label

    st = np.zeros((3, 3, 3), dtype=bool)
    st[1, 1, :] = st[1, :, 1] = True
    bg = ~mask
    lab, n = label(bg, structure=st)
    outside = np.zeros(n + 1, dtype=bool)
    for face in (lab[:, 0, :], lab[:, -1, :], lab[:, :, 0], lab[:, :, -1]):
        outside[face.ravel()] = True
    outside[0] = True  # label 0 is the mask itself
    return mask | ~outside[lab]


def _grow_floodfill(volume, seed, abs_tol, smooth=0.0):
    """Region of [z, y, x] *volume* joined to *seed*.

    Returns (full mask, seed value, box slices, mask inside the box). Works in
    a box around the seed and doubles it until the region fits, so a compact
    region in a big volume costs only its own neighbourhood.
    """
    if smooth > 0:
        try:
            import scipy.ndimage  # noqa: F401
        except ImportError:
            smooth = 0.0
    shape = volume.shape
    half = [24, 32, 32]
    if shape[0] > 1:
        # Size the first box from the region on the seed's own plane, so a
        # big region does not walk up through several small boxes.
        plo = [seed[0], 0, 0]
        phi = [seed[0] + 1, shape[1], shape[2]]
        plane, _v, _h = _grow_in_box(volume, seed, abs_tol, smooth, plo, phi)
        ys, xs = np.flatnonzero(plane[0].any(axis=1)), np.flatnonzero(plane[0].any(axis=0))
        if ys.size:
            reach = max(seed[1] - ys[0], ys[-1] - seed[1], seed[2] - xs[0], xs[-1] - seed[2]) + 8
            half = [max(24, reach), max(32, reach), max(32, reach)]
    while True:
        lo = [max(0, seed[i] - half[i]) for i in range(3)]
        hi = [min(shape[i], seed[i] + half[i] + 1) for i in range(3)]
        crop, seed_val, hit = _grow_in_box(volume, seed, abs_tol, smooth, lo, hi)
        if not hit:
            break
        for axis in hit:
            half[axis] *= 2
    box = tuple(slice(lo[i], hi[i]) for i in range(3))
    full = np.zeros(shape, dtype=bool)
    full[box] = crop
    return full, seed_val, box, crop


def _scoped_floodfill_volume(session, dim_x, dim_y, indices, complex_mode, scope_dim):
    """Return ``(volume, abs_range)`` for a scoped grow; volume is [scope, y, x]."""
    other = tuple(v for i, v in enumerate(indices) if i not in (dim_x, dim_y, scope_dim))
    key = (
        session.sid,
        id(session.data),
        getattr(session, "data_version", 0),
        int(dim_x),
        int(dim_y),
        int(scope_dim),
        int(complex_mode),
        other,
    )
    with _FLOODFILL_VOLUME_LOCK:
        hit = _FLOODFILL_VOLUME_CACHE.get("entry")
        if hit is not None and hit[0] == key:
            return hit[1], hit[2]
    planes = []
    for value in range(session.shape[scope_dim]):
        idx = list(indices)
        idx[scope_dim] = value
        planes.append(
            apply_complex_mode(extract_slice(session, dim_x, dim_y, idx), complex_mode)
        )
    volume = np.stack(planes, axis=0)
    mag = np.abs(volume)
    abs_range = float(np.nanmax(mag) - np.nanmin(mag) + 1e-10)
    with _FLOODFILL_VOLUME_LOCK:
        _FLOODFILL_VOLUME_CACHE["entry"] = (key, volume, abs_range)
    return volume, abs_range


def _scoped_floodfill_response(
    session,
    dim_x: int,
    dim_y: int,
    indices: list[int],
    px: int,
    py: int,
    tolerance: float,
    complex_mode: int,
    scope_dim: int,
    preview: bool = False,
    smooth: float = 0.0,
) -> dict:
    shape = session.shape
    if len(indices) < len(shape):
        indices.extend(s // 2 for s in shape[len(indices) :])
    indices = [
        min(max(int(v), 0), shape[i] - 1)
        for i, v in enumerate(indices[: len(shape)])
    ]
    scope_dim = int(scope_dim)
    if scope_dim in {dim_x, dim_y} or not (0 <= scope_dim < len(shape)):
        return {"error": "scope_dim must be exactly one non-display dimension"}

    values = list(range(shape[scope_dim]))
    volume, abs_range = _scoped_floodfill_volume(
        session, dim_x, dim_y, indices, complex_mode, scope_dim
    )
    depth, h, w = volume.shape
    if not (0 <= py < h and 0 <= px < w):
        return {"error": "seed out of bounds"}

    seed_z = values.index(indices[scope_dim])
    abs_tol = tolerance * abs_range
    grow_key = ((seed_z, py, px), float(abs_tol), float(smooth))
    with _FLOODFILL_VOLUME_LOCK:
        hit = _FLOODFILL_GROW_CACHE.get("entry")
    if hit is not None and hit[0] is volume and hit[1] == grow_key:
        component, seed_val, box, crop = hit[2]
    else:
        component, seed_val, box, crop = _grow_floodfill(
            volume, (seed_z, py, px), abs_tol, smooth
        )
        with _FLOODFILL_VOLUME_LOCK:
            _FLOODFILL_GROW_CACHE["entry"] = (volume, grow_key, (component, seed_val, box, crop))
    roi_values = volume[box][crop]
    finite = roi_values[np.isfinite(roi_values)]

    if preview:
        # While the sensitivity is being dragged: stats of the whole grown
        # region, but the mask of the shown plane only. The full 3D mask is
        # megabytes per step and only needed once, on release.
        plane = component[seed_z]
        ys, xs = np.where(plane)
        bbox = (
            {"x0": int(xs.min()), "y0": int(ys.min()), "x1": int(xs.max()), "y1": int(ys.max())}
            if len(xs) else {"x0": 0, "y0": 0, "x1": 0, "y1": 0}
        )
        return {
            "min": _safe_float(finite.min()) if finite.size else None,
            "max": _safe_float(finite.max()) if finite.size else None,
            "mean": _safe_float(finite.mean()) if finite.size else None,
            "std": _safe_float(finite.std()) if finite.size else None,
            "n": int(finite.size),
            "seed_value": _safe_float(seed_val),
            "tolerance": tolerance,
            "smooth": smooth,
            "abs_tol": _safe_float(abs_tol),
            "bbox": bbox,
            "mask_b64": _encode_mask_b64(plane, bbox),
            "preview": True,
        }

    # Release: the drag is over, so free the held volume.
    with _FLOODFILL_VOLUME_LOCK:
        _FLOODFILL_VOLUME_CACHE.clear()
        _FLOODFILL_GROW_CACHE.clear()
        _FLOODFILL_BLUR_CACHE.clear()

    slices = []
    scoped_values = []
    for z in range(depth):
        plane_mask = component[z]
        if not plane_mask.any():
            continue
        ys, xs = np.where(plane_mask)
        bbox = {
            "x0": int(xs.min()),
            "y0": int(ys.min()),
            "x1": int(xs.max()),
            "y1": int(ys.max()),
        }
        scope_value = values[z]
        scoped_values.append(scope_value)
        slices.append(
            {
                "index": int(scope_value),
                "bbox": bbox,
                "mask_b64": _encode_mask_b64(plane_mask, bbox),
            }
        )

    roi = {
        "type": "floodfill",
        "scope_dim": int(scope_dim),
        "scope": {
            "indices": list(indices),
            "broadcast_dims": [int(scope_dim)],
            "values": {str(scope_dim): scoped_values},
        },
        "slices": slices,
        # Full 3D component so the frontend can render the grown region in
        # every multiview pane, not just the plane it was seeded on. Shape
        # axes are [scope_dim, dim_y, dim_x] (the grow axes); grow_dim_x /
        # grow_dim_y record which data dims those are.
        # Bit-packed (LSB first, C order): 8x smaller than one byte per voxel.
        "mask3d_b64": base64.b64encode(
            np.packbits(component, axis=None, bitorder="little").tobytes()
        ).decode("ascii"),
        "mask3d_shape": [int(s) for s in component.shape],
        "grow_dim_x": int(dim_x),
        "grow_dim_y": int(dim_y),
    }
    current_slice = next(
        (entry for entry in slices if int(entry["index"]) == int(indices[scope_dim])),
        None,
    )
    return {
        "min": _safe_float(finite.min()) if finite.size else None,
        "max": _safe_float(finite.max()) if finite.size else None,
        "mean": _safe_float(finite.mean()) if finite.size else None,
        "std": _safe_float(finite.std()) if finite.size else None,
        "n": int(finite.size),
        "seed_value": _safe_float(seed_val),
        "tolerance": tolerance,
        "smooth": smooth,
        "abs_tol": _safe_float(abs_tol),
        "bbox": current_slice["bbox"] if current_slice else {"x0": 0, "y0": 0, "x1": 0, "y1": 0},
        "mask_b64": current_slice["mask_b64"] if current_slice else "",
        "scope": roi["scope"],
        "slices": slices,
        "roi": roi,
    }


def _connected_component(mask: np.ndarray, seed: tuple[int, int, int]) -> np.ndarray:
    try:
        from scipy.ndimage import label

        labeled, _n_features = label(mask)
        seed_label = labeled[seed]
        if seed_label == 0:
            return np.zeros(mask.shape, dtype=bool)
        return labeled == seed_label
    except ImportError:
        component = np.zeros(mask.shape, dtype=bool)
        if not mask[seed]:
            return component
        stack = [seed]
        component[seed] = True
        while stack:
            cz, cy, cx = stack.pop()
            for dz, dy, dx in [
                (-1, 0, 0),
                (1, 0, 0),
                (0, -1, 0),
                (0, 1, 0),
                (0, 0, -1),
                (0, 0, 1),
            ]:
                nz, ny, nx = cz + dz, cy + dy, cx + dx
                if (
                    0 <= nz < mask.shape[0]
                    and 0 <= ny < mask.shape[1]
                    and 0 <= nx < mask.shape[2]
                    and mask[nz, ny, nx]
                    and not component[nz, ny, nx]
                ):
                    component[nz, ny, nx] = True
                    stack.append((nz, ny, nx))
        return component


def _roi_indices_from_body(body: dict, ndim: int, shape: tuple[int, ...]) -> list[int]:
    raw = body.get("indices", "")
    if isinstance(raw, str):
        vals = [int(v) for v in raw.split(",") if v != ""]
    else:
        vals = [int(v) for v in raw]
    if len(vals) < ndim:
        vals.extend(s // 2 for s in shape[len(vals) :])
    return [min(max(v, 0), shape[i] - 1) for i, v in enumerate(vals[:ndim])]


def _roi_scope_indices(
    roi: dict,
    base_indices: list[int],
    shape: tuple[int, ...],
    spatial_dims: set[int],
):
    scope = roi.get("scope") or {}
    fixed = scope.get("indices") or base_indices
    if isinstance(fixed, str):
        fixed = [int(v) for v in fixed.split(",") if v != ""]
    fixed = list(fixed)
    if len(fixed) < len(shape):
        fixed.extend(base_indices[len(fixed) :])
    fixed = [min(max(int(v), 0), shape[i] - 1) for i, v in enumerate(fixed[: len(shape)])]

    raw_dims = scope.get("broadcast_dims", [])
    broadcast_dims = []
    for d in raw_dims:
        d = int(d)
        if 0 <= d < len(shape) and d not in spatial_dims:
            broadcast_dims.append(d)

    ranges = scope.get("ranges") or {}
    values = scope.get("values") or {}
    choices = []
    for d in broadcast_dims:
        key = str(d)
        if key in values:
            vals = [int(v) for v in values[key]]
        elif d in values:
            vals = [int(v) for v in values[d]]
        elif key in ranges:
            lo, hi = ranges[key]
            vals = list(range(int(lo), int(hi) + 1))
        elif d in ranges:
            lo, hi = ranges[d]
            vals = list(range(int(lo), int(hi) + 1))
        else:
            vals = list(range(shape[d]))
        vals = [v for v in vals if 0 <= v < shape[d]]
        choices.append((d, vals or [fixed[d]]))

    if not choices:
        yield fixed
        return
    for combo in itertools.product(*(vals for _d, vals in choices)):
        idx = list(fixed)
        for (d, _vals), v in zip(choices, combo):
            idx[d] = v
        yield idx


def _roi_scope_dims_string(idx: list[int], base_idx: list[int], spatial_dims: set[int]) -> str:
    bits = []
    for d, val in enumerate(idx):
        bits.append("1" if d in spatial_dims or val != base_idx[d] else "0")
    return "".join(bits)


def _roi_mask_for_shape(
    roi: dict, data_shape: tuple[int, int], indices: list[int] | None = None
) -> np.ndarray:
    h, w = data_shape
    roi_type = roi.get("type", "rect")
    if roi_type == "circle":
        cx = float(roi.get("cx", 0))
        cy = float(roi.get("cy", 0))
        r = max(0.0, float(roi.get("r", 0)))
        ys, xs = np.ogrid[:h, :w]
        return (xs - cx) ** 2 + (ys - cy) ** 2 <= r**2
    if roi_type == "freehand":
        points = roi.get("points", [])
        if len(points) < 3:
            return np.zeros((h, w), dtype=bool)
        from PIL import Image as _PILImage, ImageDraw as _PILDraw

        mask_img = _PILImage.new("L", (w, h), 0)
        _PILDraw.Draw(mask_img).polygon([(p[0], p[1]) for p in points], fill=255)
        return np.array(mask_img) > 0
    if roi_type == "floodfill":
        return _floodfill_roi_mask(roi, data_shape, indices)

    x0 = int(round(float(roi.get("x0", 0))))
    y0 = int(round(float(roi.get("y0", 0))))
    x1 = int(round(float(roi.get("x1", x0))))
    y1 = int(round(float(roi.get("y1", y0))))
    xa = max(0, min(x0, x1, w - 1))
    xb = min(w, max(x0, x1) + 1)
    ya = max(0, min(y0, y1, h - 1))
    yb = min(h, max(y0, y1) + 1)
    mask = np.zeros((h, w), dtype=bool)
    if xa < xb and ya < yb:
        mask[ya:yb, xa:xb] = True
    return mask


def _floodfill_roi_mask(
    roi: dict, data_shape: tuple[int, int], indices: list[int] | None = None
) -> np.ndarray:
    h, w = data_shape
    mask = np.zeros((h, w), dtype=bool)
    if indices is not None and roi.get("slices"):
        scope_dim = roi.get("scope_dim")
        if scope_dim is None:
            scope = roi.get("scope") or {}
            raw_dims = scope.get("broadcast_dims") or []
            scope_dim = raw_dims[0] if len(raw_dims) == 1 else None
        if scope_dim is None:
            return mask
        scope_dim = int(scope_dim)
        if not (0 <= scope_dim < len(indices)):
            return mask
        target = int(indices[scope_dim])
        for entry in roi.get("slices") or []:
            if int(entry.get("index", -1)) == target:
                roi = entry
                break
        else:
            return mask
    bbox = roi.get("bbox") or {}
    encoded = roi.get("mask_b64")
    if not encoded:
        return mask
    try:
        x0 = int(bbox.get("x0", 0))
        y0 = int(bbox.get("y0", 0))
        x1 = int(bbox.get("x1", x0))
        y1 = int(bbox.get("y1", y0))
        bw = x1 - x0 + 1
        bh = y1 - y0 + 1
        if bw <= 0 or bh <= 0:
            return mask
        sub = _decode_mask_bytes(encoded, bw * bh).reshape((bh, bw))
    except Exception:
        return mask
    xa0, ya0 = max(0, x0), max(0, y0)
    xa1, ya1 = min(w, x1 + 1), min(h, y1 + 1)
    if xa0 >= xa1 or ya0 >= ya1:
        return mask
    sx0, sy0 = xa0 - x0, ya0 - y0
    sx1, sy1 = sx0 + (xa1 - xa0), sy0 + (ya1 - ya0)
    mask[ya0:ya1, xa0:xa1] = sub[sy0:sy1, sx0:sx1]
    return mask


def _roi_finite_values(data: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if mask.shape != data.shape:
        return np.array([])
    roi = data[mask]
    return roi[np.isfinite(roi)]


_ROI_HIST_BINS = 24


def _roi_value_histogram(values: list[np.ndarray], lo, hi) -> list[int]:
    """Pixel counts in equal bins between the ROI's own min and max.

    The viewer places each ROI's bins on one value axis shared by all ROIs,
    so only the counts travel; the edges follow from min/max.
    """
    if not values or lo is None or hi is None:
        return []
    counts, _ = np.histogram(
        np.concatenate(values), bins=_ROI_HIST_BINS, range=(float(lo), float(hi) if hi > lo else float(lo) + 1.0)
    )
    return [int(c) for c in counts]


def _combine_roi_rows(rows: list[dict]) -> dict | None:
    if not rows:
        return None
    n = sum(int(r["n"]) for r in rows)
    if n <= 0:
        return None
    means = np.array([float(r["mean"]) for r in rows], dtype=np.float64)
    stds = np.array([float(r["std"]) for r in rows], dtype=np.float64)
    counts = np.array([int(r["n"]) for r in rows], dtype=np.float64)
    mean = float(np.sum(means * counts) / np.sum(counts))
    variance = float(np.sum((stds**2 + (means - mean) ** 2) * counts) / np.sum(counts))
    return {
        "min": min(r["min"] for r in rows),
        "max": max(r["max"] for r in rows),
        "mean": _safe_float(mean),
        "std": _safe_float(math.sqrt(max(0.0, variance))),
        "n": int(n),
    }


def _write_roi_mask(
    labels: np.ndarray,
    mask_2d: np.ndarray,
    indices: list[int],
    dim_x: int,
    dim_y: int,
    label: int,
) -> None:
    ys, xs = np.nonzero(mask_2d)
    for y, x in zip(ys, xs):
        idx = list(indices)
        idx[dim_y] = int(y)
        idx[dim_x] = int(x)
        labels[tuple(idx)] = label
