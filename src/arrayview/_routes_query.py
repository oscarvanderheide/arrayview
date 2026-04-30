import io

import numpy as np
from fastapi import Depends, Response

from arrayview._analysis import _build_metadata
from arrayview._render import render_rgb_rgba, render_rgba
from arrayview._session import SESSIONS, wait_for_session_ready
import arrayview._session as _session_mod


def register_query_routes(app, *, get_session_or_404, pil_image, pil_imageops) -> None:
    @app.get("/metadata/{sid}")
    async def get_metadata(sid: str):
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
            info["spatial_meta"] = {
                "voxel_sizes": list(spatial_meta["voxel_sizes"]),
                "axis_labels": list(spatial_meta["axis_labels"]),
                "is_oblique": bool(spatial_meta["is_oblique"]),
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
