import asyncio
import io

import numpy as np
from fastapi import Body, Depends, Response

from arrayview._session import Session, SESSIONS


_seg_overlay_sid: str | None = None  # current segmentation overlay session ID
_seg_label_mask: np.ndarray | None = None  # cumulative multi-label mask (3D)
_seg_current_label: int = 0  # label counter for accept
_seg_vol_axes: tuple[int, ...] | None = None  # which N-D axes map to 3D volume
_seg_fixed_indices: tuple[int, ...] | None = None  # fixed indices for non-volume dims
_seg_full_shape: tuple[int, ...] | None = None  # full N-D shape of source session


def _seg_coord_3d(
    session, dim_x: int, dim_y: int, idx_tuple: tuple[int, ...], px: int, py: int
) -> tuple[int, ...]:
    """Map a canvas click (px, py) to a 3D coordinate in the nnInteractive volume."""
    idx = list(idx_tuple)
    idx[dim_x] = px
    idx[dim_y] = py
    if _seg_vol_axes is not None:
        return tuple(idx[a] for a in _seg_vol_axes)
    return tuple(idx)


def _seg_rasterize_drawing(session, body: dict, fill: bool = False) -> np.ndarray | None:
    """Rasterize 2D drawing points into a 3D mask on the current slice."""
    from PIL import Image, ImageDraw

    points = body.get("points", [])
    if len(points) < (3 if fill else 1):
        return None

    dim_x = int(body["dim_x"])
    dim_y = int(body["dim_y"])
    idx_tuple = tuple(int(x) for x in body["indices"].split(","))

    vol_shape = _seg_label_mask.shape
    mask_3d = np.zeros(vol_shape, dtype=np.uint8)

    if _seg_vol_axes is not None:
        vol_ax_list = list(_seg_vol_axes)
        slice_axis_vol = [
            i for i, a in enumerate(vol_ax_list) if a != dim_x and a != dim_y
        ]
        if not slice_axis_vol:
            return None
        slice_ax = slice_axis_vol[0]
        slice_idx = idx_tuple[vol_ax_list[slice_ax]]

        vx = vol_ax_list.index(dim_x)
        vy = vol_ax_list.index(dim_y)
        w = vol_shape[vx]
        h = vol_shape[vy]
    else:
        others = [a for a in range(3) if a != dim_x and a != dim_y]
        slice_ax = others[0]
        slice_idx = idx_tuple[slice_ax]
        vx, vy = dim_x, dim_y
        w = vol_shape[vx]
        h = vol_shape[vy]

    img = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(img)
    coords = [(int(p[0]), int(p[1])) for p in points]

    if fill and len(coords) >= 3:
        draw.polygon(coords, fill=1)
    else:
        if len(coords) >= 2:
            draw.line(coords, fill=1, width=2)
        else:
            x, y = coords[0]
            draw.ellipse([x - 1, y - 1, x + 1, y + 1], fill=1)

    mask_2d = np.array(img, dtype=np.uint8)

    slc = [slice(None)] * 3
    slc[slice_ax] = slice_idx
    remaining = [i for i in range(3) if i != slice_ax]
    if remaining.index(vx) < remaining.index(vy):
        mask_2d = mask_2d.T
    mask_3d[tuple(slc)] = mask_2d

    return mask_3d


def _seg_expand_to_full(mask_3d: np.ndarray) -> np.ndarray:
    """Expand a 3D mask back to the full N-D shape for overlay display."""
    if _seg_full_shape is None or _seg_vol_axes is None:
        return mask_3d
    if len(_seg_full_shape) == 3:
        return mask_3d
    full = np.zeros(_seg_full_shape, dtype=mask_3d.dtype)
    ndim = len(_seg_full_shape)
    idx = _seg_fixed_indices if _seg_fixed_indices else (0,) * ndim
    slc = [slice(None) if ax in _seg_vol_axes else idx[ax] for ax in range(ndim)]
    full[tuple(slc)] = mask_3d
    return full


def _seg_apply_mask(mask: np.ndarray) -> dict:
    """Store prediction mask (+ cumulative labels) as overlay session."""
    global _seg_overlay_sid

    if _seg_label_mask is not None:
        combined = _seg_label_mask.copy()
        combined[mask > 0] = _seg_current_label + 1
    else:
        combined = mask

    overlay_data = _seg_expand_to_full(combined)
    _seg_set_overlay(overlay_data)
    return {"status": "ok", "overlay_sid": _seg_overlay_sid}


def _seg_update_overlay() -> dict:
    """Update overlay to show cumulative label mask only (after reset/accept)."""
    if _seg_label_mask is None:
        return {"status": "ok", "overlay_sid": None}

    overlay_data = _seg_expand_to_full(_seg_label_mask.copy())
    _seg_set_overlay(overlay_data)
    return {
        "status": "ok",
        "overlay_sid": _seg_overlay_sid,
        "labels": _seg_get_label_info(),
    }


def _seg_set_overlay(data: np.ndarray) -> None:
    """Create or update the segmentation overlay session."""
    global _seg_overlay_sid
    if _seg_overlay_sid and _seg_overlay_sid in SESSIONS:
        ov = SESSIONS[_seg_overlay_sid]
        ov.data = data
        ov.shape = data.shape
        ov.reset_caches()
        ov.data_version += 1
    else:
        ov = Session(data, name="nnInteractive segmentation")
        SESSIONS[ov.sid] = ov
        _seg_overlay_sid = ov.sid


_LABEL_HEX = [
    "#ff5050",
    "#50a0ff",
    "#50d250",
    "#ffaf32",
    "#b950ff",
    "#ff64be",
    "#3cd2c3",
    "#f0dc32",
    "#a06e3c",
    "#b4b4b4",
]

_seg_label_names: dict[int, str] = {}


def _seg_get_label_info() -> list[dict]:
    """Return info about all accepted labels."""
    if _seg_label_mask is None:
        return []
    labels = []
    for lbl in range(1, _seg_current_label + 1):
        voxels = int(np.sum(_seg_label_mask == lbl))
        if voxels == 0:
            continue
        labels.append(
            {
                "label": lbl,
                "name": _seg_label_names.get(lbl, f"segment {lbl}"),
                "color": _LABEL_HEX[(lbl - 1) % len(_LABEL_HEX)],
                "voxels": voxels,
            }
        )
    return labels


def register_segmentation_routes(app, get_session_or_404) -> None:
    @app.post("/seg/activate/{sid}")
    async def seg_activate(
        sid: str,
        dim_x: int = 0,
        dim_y: int = 1,
        scroll_dim: int = -1,
        indices: str = "",
        session=Depends(get_session_or_404),
    ):
        """Connect to nnInteractive server (auto-launch if needed) and upload volume."""
        from arrayview import _segmentation as seg

        shape = session.spatial_shape if session.rgb_axis is not None else session.shape
        ndim = len(shape)
        if ndim < 3:
            return {
                "status": "error",
                "message": f"nnInteractive requires 3D+ data (got {ndim}D)",
            }

        global _seg_overlay_sid, _seg_label_mask, _seg_current_label
        global _seg_vol_axes, _seg_fixed_indices, _seg_full_shape

        if not seg.is_connected():
            from arrayview._config import get_nninteractive_url

            configured_url = get_nninteractive_url()
            if configured_url:
                if not seg.try_connect(url=configured_url):
                    return {
                        "status": "error",
                        "message": f"cannot reach nnInteractive at {configured_url}",
                    }
            elif not seg.try_connect():
                err = await asyncio.to_thread(seg.try_launch)
                if err:
                    return {"status": "error", "message": err}

        if _seg_label_mask is not None and seg.is_connected():
            return {
                "status": "ok",
                "message": "resumed",
                "ndim": ndim,
                "overlay_sid": _seg_overlay_sid,
                "labels": _seg_get_label_info(),
            }

        data = np.asarray(session.data)
        if session.rgb_axis is not None:
            slc = [slice(None)] * data.ndim
            slc[session.rgb_axis] = 0
            data = data[tuple(slc)]

        _seg_full_shape = data.shape

        if ndim == 3:
            vol = data
            _seg_vol_axes = (0, 1, 2)
            _seg_fixed_indices = ()
        else:
            if scroll_dim < 0:
                for ax in range(ndim):
                    if ax != dim_x and ax != dim_y:
                        scroll_dim = ax
                        break
            vol_axes = sorted({dim_x, dim_y, scroll_dim})
            if len(vol_axes) != 3:
                return {
                    "status": "error",
                    "message": "need 3 distinct axes for segmentation",
                }
            _seg_vol_axes = tuple(vol_axes)

            idx_list = [int(x) for x in indices.split(",")] if indices else [0] * ndim
            slc = [slice(None) if ax in vol_axes else idx_list[ax] for ax in range(ndim)]
            _seg_fixed_indices = tuple(idx_list)
            vol = data[tuple(slc)]

        try:
            await asyncio.to_thread(seg.upload_volume, vol)
        except Exception as exc:
            return {"status": "error", "message": f"upload failed: {exc}"}

        _seg_label_mask = np.zeros(vol.shape, dtype=np.uint8)
        _seg_current_label = 0
        _seg_overlay_sid = None

        return {"status": "ok", "message": "connected", "ndim": ndim}

    @app.post("/seg/click/{sid}")
    async def seg_click(
        sid: str,
        dim_x: int,
        dim_y: int,
        indices: str,
        px: int,
        py: int,
        positive: bool = True,
        session=Depends(get_session_or_404),
    ):
        """Send a point interaction to nnInteractive and return overlay session ID."""
        from arrayview import _segmentation as seg

        if not seg.is_connected():
            return {"status": "error", "message": "not connected"}

        idx_tuple = tuple(int(x) for x in indices.split(","))
        coord = _seg_coord_3d(session, dim_x, dim_y, idx_tuple, px, py)

        try:
            mask = await asyncio.to_thread(seg.add_point, coord, positive)
        except Exception as exc:
            return {"status": "error", "message": str(exc)}

        return _seg_apply_mask(mask)

    @app.post("/seg/bbox/{sid}")
    async def seg_bbox(sid: str, body: dict = Body(...), session=Depends(get_session_or_404)):
        """Send a bounding box interaction to nnInteractive."""
        from arrayview import _segmentation as seg

        if not seg.is_connected():
            return {"status": "error", "message": "not connected"}

        dim_x = int(body["dim_x"])
        dim_y = int(body["dim_y"])
        idx_tuple = tuple(int(x) for x in body["indices"].split(","))
        x0, y0 = int(body["x0"]), int(body["y0"])
        x1, y1 = int(body["x1"]), int(body["y1"])

        coord1 = _seg_coord_3d(session, dim_x, dim_y, idx_tuple, x0, y0)
        coord2 = _seg_coord_3d(session, dim_x, dim_y, idx_tuple, x1, y1)

        try:
            mask = await asyncio.to_thread(seg.add_bbox, coord1, coord2, True)
        except Exception as exc:
            return {"status": "error", "message": str(exc)}

        return _seg_apply_mask(mask)

    @app.post("/seg/scribble/{sid}")
    async def seg_scribble(
        sid: str,
        body: dict = Body(...),
        session=Depends(get_session_or_404),
    ):
        """Send a scribble interaction (freehand drawn points on current slice)."""
        from arrayview import _segmentation as seg

        if not seg.is_connected() or _seg_label_mask is None:
            return {"status": "error", "message": "not connected"}

        positive = bool(body.get("positive", True))
        mask_3d = _seg_rasterize_drawing(session, body)
        if mask_3d is None:
            return {"status": "error", "message": "no points provided"}

        try:
            mask = await asyncio.to_thread(seg.add_scribble, mask_3d, positive)
        except Exception as exc:
            return {"status": "error", "message": str(exc)}

        return _seg_apply_mask(mask)

    @app.post("/seg/lasso/{sid}")
    async def seg_lasso(
        sid: str,
        body: dict = Body(...),
        session=Depends(get_session_or_404),
    ):
        """Send a lasso interaction (filled closed contour on current slice)."""
        from arrayview import _segmentation as seg

        if not seg.is_connected() or _seg_label_mask is None:
            return {"status": "error", "message": "not connected"}

        positive = bool(body.get("positive", True))
        mask_3d = _seg_rasterize_drawing(session, body, fill=True)
        if mask_3d is None:
            return {"status": "error", "message": "need at least 3 points"}

        try:
            mask = await asyncio.to_thread(seg.add_lasso, mask_3d, positive)
        except Exception as exc:
            return {"status": "error", "message": str(exc)}

        return _seg_apply_mask(mask)

    @app.post("/seg/reset/{sid}")
    async def seg_reset(sid: str):
        """Reset interactions for next object."""
        from arrayview import _segmentation as seg

        if not seg.is_connected():
            return {"status": "error", "message": "not connected"}
        try:
            await asyncio.to_thread(seg.reset_interactions)
        except Exception as exc:
            return {"status": "error", "message": str(exc)}

        return _seg_update_overlay()

    @app.post("/seg/accept/{sid}")
    async def seg_accept(sid: str):
        """Commit current prediction to cumulative label mask, increment label."""
        from arrayview import _segmentation as seg

        global _seg_current_label
        if _seg_overlay_sid is None or _seg_label_mask is None:
            return {"status": "error", "message": "no active segmentation"}

        ov_session = SESSIONS.get(_seg_overlay_sid)
        if ov_session is None:
            return {"status": "error", "message": "overlay session lost"}

        _seg_current_label += 1
        current_mask = np.asarray(ov_session.data)
        _seg_label_mask[current_mask > 0] = _seg_current_label

        try:
            await asyncio.to_thread(seg.reset_interactions)
        except Exception:
            pass

        return _seg_update_overlay()

    @app.post("/seg/disconnect")
    async def seg_disconnect():
        """Disconnect from nnInteractive server."""
        from arrayview import _segmentation as seg

        global _seg_overlay_sid, _seg_label_mask, _seg_current_label
        global _seg_vol_axes, _seg_fixed_indices, _seg_full_shape
        seg.disconnect()
        _seg_overlay_sid = None
        _seg_label_mask = None
        _seg_current_label = 0
        _seg_vol_axes = None
        _seg_fixed_indices = None
        _seg_full_shape = None
        return {"status": "ok"}

    @app.get("/seg/labels/{sid}")
    def seg_labels(sid: str):
        """Return info about all accepted segmentation labels."""
        return {"labels": _seg_get_label_info(), "overlay_sid": _seg_overlay_sid}

    @app.post("/seg/rename/{sid}")
    def seg_rename(sid: str, label: int, name: str):
        """Rename a segmentation label."""
        _seg_label_names[label] = name
        return {"status": "ok"}

    @app.post("/seg/delete_label/{sid}")
    def seg_delete_label(sid: str, label: int):
        """Delete a segmentation label from the cumulative mask."""
        if _seg_label_mask is None:
            return {"status": "error", "message": "no segmentation active"}
        _seg_label_mask[_seg_label_mask == label] = 0
        _seg_label_names.pop(label, None)
        overlay_data = _seg_expand_to_full(_seg_label_mask.copy())
        _seg_set_overlay(overlay_data)
        return {
            "status": "ok",
            "labels": _seg_get_label_info(),
            "overlay_sid": _seg_overlay_sid,
        }

    @app.get("/seg/export/{sid}")
    def seg_export(sid: str):
        """Export cumulative label mask as downloadable .npy file."""
        if _seg_label_mask is None:
            return Response(status_code=404)
        buf = io.BytesIO()
        export_data = _seg_expand_to_full(_seg_label_mask)
        np.save(buf, export_data)
        return Response(
            content=buf.getvalue(),
            media_type="application/octet-stream",
            headers={"Content-Disposition": "attachment; filename=segmentation.npy"},
        )