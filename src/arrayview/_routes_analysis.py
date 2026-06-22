import base64
import io
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
        session=Depends(get_session_or_404),
    ):
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
            )
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
            for idx in _roi_scope_indices(roi, indices, session.shape, {dim_x, dim_y}):
                raw = extract_slice(session, dim_x, dim_y, idx)
                data = apply_complex_mode(raw, complex_mode)
                mask = _roi_mask_for_shape(roi, data.shape, idx)
                finite = _roi_finite_values(data, mask)
                if finite.size:
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
    sub = mask[y0 : y1 + 1, x0 : x1 + 1].astype(np.uint8)
    return base64.b64encode(sub.tobytes()).decode("ascii")


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
    planes = []
    for value in values:
        idx = list(indices)
        idx[scope_dim] = value
        planes.append(
            apply_complex_mode(
                extract_slice(session, dim_x, dim_y, idx), complex_mode
            )
        )
    volume = np.stack(planes, axis=0)
    depth, h, w = volume.shape
    if not (0 <= py < h and 0 <= px < w):
        return {"error": "seed out of bounds"}

    seed_z = values.index(indices[scope_dim])
    seed_val = float(volume[seed_z, py, px])
    abs_tol = tolerance * (
        np.nanmax(np.abs(volume)) - np.nanmin(np.abs(volume)) + 1e-10
    )
    mask = np.abs(volume - seed_val) <= abs_tol
    component = _connected_component(mask, (seed_z, py, px))
    roi_values = volume[component]
    finite = roi_values[np.isfinite(roi_values)]

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
        "mask3d_b64": base64.b64encode(component.astype(np.uint8).tobytes()).decode("ascii"),
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
        sub = np.frombuffer(base64.b64decode(encoded), dtype=np.uint8)
        sub = sub.reshape((bh, bw)).astype(bool)
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
