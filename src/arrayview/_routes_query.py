import io

import numpy as np
from fastapi import Depends, Response

from arrayview._analysis import _build_metadata
from arrayview._render import render_rgb_rgba, render_rgba
from arrayview._session import SESSIONS, wait_for_session_ready
import arrayview._session as _session_mod


def register_query_routes(app, *, get_session_or_404, pil_image, pil_imageops) -> None:
    @app.get("/autocrop/{sid}")
    def get_autocrop_bounds(
        sid: str,
        indices: str,
        margin_mm: float = 12.0,
        session=Depends(get_session_or_404),
    ):
        """Return a cached nonzero spatial bounding box for the selected item.

        This is intentionally a view-only operation.  It materializes a
        collection item only on first use, then keeps six small bounds values
        in the session cache; the source array and files are never changed.
        """
        idx = tuple(int(x) for x in indices.split(","))
        spatial_ndim = getattr(session, "collection_spatial_ndim", None)
        if spatial_ndim is None:
            meta = getattr(session, "spatial_meta", None)
            spatial_ndim = len(meta["voxel_sizes"]) if meta else min(3, len(session.shape))
        spatial_ndim = max(1, min(int(spatial_ndim), 3, len(session.shape)))
        if len(idx) != len(session.shape):
            return Response(status_code=422, content="indices rank does not match data")

        item_key = tuple(idx[spatial_ndim:])
        cache = getattr(session, "_autocrop_bounds", None)
        if cache is None:
            cache = session._autocrop_bounds = {}
        cache_key = (item_key, float(margin_mm))
        bounds = cache.get(cache_key)
        if bounds is None:
            slicer = tuple(
                slice(None) if axis < spatial_ndim else idx[axis]
                for axis in range(len(session.shape))
            )
            volume = np.asarray(session.data[slicer])
            if volume.ndim != spatial_ndim:
                volume = np.squeeze(volume)
            finite_nonzero = np.isfinite(volume) & (volume != 0)
            occupied = [np.any(finite_nonzero, axis=tuple(j for j in range(spatial_ndim) if j != axis))
                        for axis in range(spatial_ndim)]
            meta = getattr(session, "spatial_meta", None)
            spacing = tuple(float(v) for v in meta["voxel_sizes"][:spatial_ndim]) if meta else (1.0,) * spatial_ndim
            bounds = []
            for axis, line in enumerate(occupied):
                hits = np.flatnonzero(line)
                if hits.size:
                    margin = max(0, int(round(float(margin_mm) / spacing[axis])))
                    lo = max(0, int(hits[0]) - margin)
                    hi = min(int(volume.shape[axis]), int(hits[-1]) + 1 + margin)
                else:
                    lo, hi = 0, int(volume.shape[axis])
                bounds.append((lo, hi))
            cache[cache_key] = bounds
        return {"spatial_ndim": spatial_ndim, "bounds": bounds}

    @app.get("/metadata/{sid}")
    async def get_metadata(sid: str):
        # Metadata polling is frequent while a large file loads. Return
        # immediately instead of tying up one worker thread per timed-out HTTP
        # request; the opener retries until the pending Session is published.
        if sid in _session_mod.PENDING_SESSIONS:
            return Response(status_code=404, headers={"Retry-After": "1"})
        session = await wait_for_session_ready(sid)
        if not session:
            return Response(status_code=404)
        try:
            return _build_metadata(session)
        except Exception as e:
            import traceback

            traceback.print_exc()
            return Response(
                status_code=500, content=str(e).encode(), media_type="text/plain"
            )

    @app.get("/info/{sid}")
    def get_info(sid: str, session=Depends(get_session_or_404)):
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
        try:
            reason = _session_mod._recommend_colormap_reason(session.data)
            info["recommended_colormap"] = reason.split(" ")[0]
            info["recommended_colormap_reason"] = reason
        except Exception:
            info["recommended_colormap"] = None
            info["recommended_colormap_reason"] = None
        if session.fft_axes is not None:
            info["fft_axes"] = list(session.fft_axes)
        if getattr(session, "spatial_meta", None) is not None:
            spatial_meta = session.spatial_meta
            voxel_sizes = tuple(float(v) for v in spatial_meta["voxel_sizes"])
            spatial_shape = tuple(
                int(v)
                for v in spatial_meta.get(
                    "canonical_shape", session.shape[: len(voxel_sizes)]
                )
            )
            info["spatial_meta"] = {
                "voxel_sizes": list(voxel_sizes),
                "axis_labels": list(spatial_meta["axis_labels"]),
                "is_oblique": bool(spatial_meta["is_oblique"]),
                "field_of_view": [
                    float(size * spacing)
                    for size, spacing in zip(spatial_shape, voxel_sizes)
                ],
                "spatial_shape": list(spatial_shape),
            }
            info["ras_resample_active"] = bool(
                getattr(session, "ras_resample_active", False)
            )
        return info

    @app.get("/thumbnail/{sid}")
    async def get_thumbnail(
        sid: str,
        w: int = 96,
        h: int = 72,
        session=Depends(get_session_or_404),
    ):
        """Return a JPEG preview for the session's startup view."""
        target_shape = session.spatial_shape if session.rgb_axis is not None else session.shape
        ndim = len(target_shape)
        if ndim < 2:
            rgba = np.full((1, 1, 4), 128, dtype=np.uint8)
        else:
            default_dims = _session_mod._startup_dims_for_data(session.data, target_shape)
            if default_dims is None:
                dim_x = ndim - 1
                dim_y = ndim - 2
            else:
                dim_x = int(default_dims[0])
                dim_y = int(default_dims[1])
            idx_list = [int(size // 2) for size in target_shape]
            try:
                if session.rgb_axis is not None:
                    rgba = await asyncio.to_thread(
                        render_rgb_rgba, session, dim_x, dim_y, list(idx_list)
                    )
                else:
                    rgba = await asyncio.to_thread(
                        render_rgba,
                        session,
                        dim_x,
                        dim_y,
                        tuple(idx_list),
                        "gray",
                        1,
                        0,
                        False,
                        None,
                        None,
                    )
            except Exception:
                rgba = np.full((1, 1, 4), 128, dtype=np.uint8)

        image_cls = pil_image()
        image_ops = pil_imageops()
        img = image_cls.fromarray(rgba[:, :, :3])
        img = image_ops.contain(
            img,
            (max(1, int(w)), max(1, int(h))),
            method=image_cls.NEAREST if max(rgba.shape[:2]) < h else image_cls.LANCZOS,
        )
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=60)
        return Response(
            content=buf.getvalue(),
            media_type="image/jpeg",
            headers={
                "Cache-Control": "max-age=30",
                "X-ArrayView-Width": str(img.width),
                "X-ArrayView-Height": str(img.height),
            },
        )
