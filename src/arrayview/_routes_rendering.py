import io

import numpy as np
from fastapi import Depends, Response
from fastapi.responses import JSONResponse

from arrayview._diff import _compute_diff, _diff_histogram, _render_diff_rgba
from arrayview._overlays import _composite_overlays
from arrayview._render import (
    LUTS,
    _build_mosaic_grid,
    _compute_vmin_vmax,
    _ensure_lut,
    _init_luts,
    _prepare_display,
    apply_colormap_rgba,
    apply_complex_mode,
    extract_projection,
    extract_slice,
    render_mosaic,
    render_projection_rgba,
    render_rgb_rgba,
    render_rgba,
)
from arrayview._session import HEAVY_OP_LIMIT_BYTES, SESSIONS
from arrayview._synthetic_mri import (
    qmri_display_slice,
    render_qmri_mosaic_rgba,
    synthetic_qmri_slice,
    synthetic_window,
)


from arrayview._imaging import ensure_image as _pil_image


def render_rgba_from_prepared(
    session,
    raw,
    colormap,
    dr,
    complex_mode=0,
    log_scale=False,
    vmin_override=None,
    vmax_override=None,
):
    return apply_colormap_rgba(
        session,
        raw,
        colormap,
        dr,
        complex_mode,
        log_scale,
        vmin_override=vmin_override,
        vmax_override=vmax_override,
    )


def render_rgba_from_data(data, colormap, vmin, vmax):
    if vmax > vmin:
        normalized = np.clip((data - vmin) / (vmax - vmin), 0, 1)
    else:
        normalized = np.zeros_like(data)
    _ensure_lut(colormap)
    lut = LUTS.get(colormap, LUTS["gray"])
    return lut[(normalized * 255).astype(np.uint8)]


def register_rendering_routes(app, *, get_session_or_404) -> None:
    @app.get("/slice/{sid}")
    def get_slice(
        sid: str,
        dim_x: int,
        dim_y: int,
        indices: str,
        colormap: str = "gray",
        dr: int = 1,
        slice_dim: int = -1,
        dim_z: int = -1,
        complex_mode: int = 0,
        log_scale: bool = False,
        vmin_override: float | None = None,
        vmax_override: float | None = None,
        overlay_sid: str | None = None,
        overlay_colors: str | None = None,
        overlay_alpha: float = 0.45,
        mosaic_cols: int | None = None,
        projection_mode: int = 0,
        projection_dim: int = -1,
        qmri_dim: int = -1,
        qmri_role: str = "",
        synthetic_mri: str = "",
        te: float | None = None,
        tr: float | None = None,
        ti: float | None = None,
        session=Depends(get_session_or_404),
    ):
        idx_tuple = tuple(int(x) for x in indices.split(","))
        if synthetic_mri:
            try:
                raw = synthetic_qmri_slice(
                    session,
                    dim_x,
                    dim_y,
                    idx_tuple,
                    qmri_dim,
                    synthetic_mri,
                    te=te,
                    tr=tr,
                    ti=ti,
                )
            except ValueError as exc:
                return Response(status_code=422, content=str(exc))
            vmin, vmax = synthetic_window(raw, synthetic_mri)
            rgba = render_rgba_from_data(raw, "gray", vmin, vmax)
        elif projection_mode > 0 and projection_dim >= 0:
            rgba = render_projection_rgba(
                session,
                dim_x,
                dim_y,
                idx_tuple,
                projection_dim,
                projection_mode,
                colormap,
                dr,
                complex_mode,
                log_scale,
                vmin_override,
                vmax_override,
            )
            raw = extract_projection(
                session, dim_x, dim_y, list(idx_tuple), projection_dim, projection_mode
            )
            _, vmin, vmax = _prepare_display(
                session,
                raw,
                complex_mode,
                dr,
                log_scale,
                vmin_override=vmin_override,
                vmax_override=vmax_override,
            )
        elif dim_z >= 0:
            if qmri_role:
                rgba, vmin, vmax = render_qmri_mosaic_rgba(
                    session,
                    dim_x,
                    dim_y,
                    dim_z,
                    idx_tuple,
                    colormap,
                    qmri_role,
                    complex_mode,
                    log_scale,
                    mosaic_cols=mosaic_cols,
                    vmin_override=vmin_override,
                    vmax_override=vmax_override,
                )
            else:
                rgba = render_mosaic(
                    session,
                    dim_x,
                    dim_y,
                    dim_z,
                    idx_tuple,
                    colormap,
                    dr,
                    complex_mode,
                    log_scale,
                    mosaic_cols=mosaic_cols,
                    vmin_override=vmin_override,
                    vmax_override=vmax_override,
                )
                if vmin_override is not None and vmax_override is not None:
                    vmin, vmax = float(vmin_override), float(vmax_override)
                else:
                    frames_raw = [
                        extract_slice(
                            session,
                            dim_x,
                            dim_y,
                            [
                                i if j == dim_z else idx_tuple[j]
                                for j in range(len(session.shape))
                            ],
                        )
                        for i in range(session.shape[dim_z])
                    ]
                    frames = [apply_complex_mode(frame, complex_mode) for frame in frames_raw]
                    if log_scale:
                        frames = [
                            np.log1p(np.abs(frame)).astype(np.float32) for frame in frames
                        ]
                        all_data = np.stack(frames)
                        vmin = float(np.percentile(all_data, 1))
                        vmax = float(np.percentile(all_data, 99))
                    else:
                        vmin, vmax = _compute_vmin_vmax(
                            session, np.stack(frames), dr, complex_mode
                        )
        else:
            if session.rgb_axis is not None:
                rgba = render_rgb_rgba(session, dim_x, dim_y, list(idx_tuple))
                vmin, vmax = 0.0, 255.0
            else:
                raw = (
                    qmri_display_slice(session, dim_x, dim_y, list(idx_tuple), qmri_role)
                    if qmri_role
                    else extract_slice(session, dim_x, dim_y, list(idx_tuple))
                )
                rgba = (
                    render_rgba_from_prepared(
                        session,
                        raw,
                        colormap,
                        dr,
                        complex_mode,
                        log_scale,
                        vmin_override,
                        vmax_override,
                    )
                    if qmri_role
                    else render_rgba(
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
                    )
                )
                rgba = _composite_overlays(
                    rgba,
                    overlay_sid,
                    overlay_colors,
                    overlay_alpha,
                    dim_x,
                    dim_y,
                    idx_tuple,
                    rgba.shape[:2],
                )
                _, vmin, vmax = _prepare_display(
                    session,
                    raw,
                    complex_mode,
                    dr,
                    log_scale,
                    vmin_override=vmin_override,
                    vmax_override=vmax_override,
                )
        img = _pil_image().fromarray(rgba[:, :, :3], mode="RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        return Response(
            content=buf.getvalue(),
            media_type="image/jpeg",
            headers={
                "Cache-Control": "max-age=300",
                "X-ArrayView-Vmin": str(vmin),
                "X-ArrayView-Vmax": str(vmax),
            },
        )

    @app.get("/diff/{sid_a}/{sid_b}")
    def get_diff(
        sid_a: str,
        sid_b: str,
        dim_x: int,
        dim_y: int,
        indices: str,
        dim_z: int = -1,
        dr: int = 1,
        complex_mode: int = 0,
        log_scale: bool = False,
        diff_mode: int = 1,
        diff_colormap: str = "",
        vmin_override: float = None,
        vmax_override: float = None,
    ):
        session_a = SESSIONS.get(sid_a)
        session_b = SESSIONS.get(sid_b)
        if not session_a or not session_b:
            return Response(status_code=404)
        try:
            raw, vmin, vmax, colormap, nan_mask = _compute_diff(
                session_a,
                session_b,
                dim_x,
                dim_y,
                indices,
                dim_z,
                dr,
                complex_mode,
                log_scale,
                diff_mode,
            )
        except Exception:
            return Response(status_code=422)
        if vmin_override is not None:
            vmin = vmin_override
        if vmax_override is not None:
            vmax = vmax_override
        if diff_colormap and _ensure_lut(diff_colormap):
            colormap = diff_colormap
        rgba = _render_diff_rgba(raw, vmin, vmax, colormap, nan_mask)
        img = _pil_image().fromarray(rgba[:, :, :3], mode="RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        return Response(
            content=buf.getvalue(),
            media_type="image/jpeg",
            headers={
                "Cache-Control": "no-cache",
                "X-ArrayView-Vmin": str(vmin),
                "X-ArrayView-Vmax": str(vmax),
                "X-ArrayView-Colormap": colormap,
            },
        )

    @app.get("/diff-histogram/{sid_a}/{sid_b}")
    def get_diff_histogram(
        sid_a: str,
        sid_b: str,
        dim_x: int,
        dim_y: int,
        indices: str,
        dim_z: int = -1,
        dr: int = 1,
        complex_mode: int = 0,
        log_scale: bool = False,
        diff_mode: int = 1,
        bins: int = 64,
    ):
        session_a = SESSIONS.get(sid_a)
        session_b = SESSIONS.get(sid_b)
        if not session_a or not session_b:
            return Response(status_code=404)
        try:
            raw, _, _, _, _ = _compute_diff(
                session_a,
                session_b,
                dim_x,
                dim_y,
                indices,
                dim_z,
                dr,
                complex_mode,
                log_scale,
                diff_mode,
            )
        except Exception:
            return Response(status_code=422)
        return _diff_histogram(raw, bins)

    @app.get("/oblique/{sid}")
    def get_oblique(
        sid: str,
        center: str,
        basis_h: str,
        basis_v: str,
        mv_dims: str,
        size_w: int,
        size_h: int,
        colormap: str = "gray",
        dr: int = 1,
        complex_mode: int = 0,
        log_scale: bool = False,
        vmin_override: float | None = None,
        vmax_override: float | None = None,
        quality: str = "full",
        session=Depends(get_session_or_404),
    ):
        """Render an oblique (arbitrarily-oriented) slice through a 3-D volume."""
        from scipy.ndimage import map_coordinates

        ctr = [float(x) for x in center.split(",")]
        bh = [float(x) for x in basis_h.split(",")]
        bv = [float(x) for x in basis_v.split(",")]
        dims = [int(x) for x in mv_dims.split(",")]

        draft = quality == "draft"
        grid_w = size_w // 2 if draft else size_w
        grid_h = size_h // 2 if draft else size_h

        ndim = len(session.shape)
        hw, hh = size_w / 2.0, size_h / 2.0
        s_arr = np.linspace(-hw, hw, grid_w, dtype=np.float64)
        t_arr = np.linspace(-hh, hh, grid_h, dtype=np.float64)
        ss, tt = np.meshgrid(s_arr, t_arr)

        coords = np.empty((ndim, grid_h, grid_w), dtype=np.float64)
        for ai in range(ndim):
            if ai in dims:
                ji = dims.index(ai)
                coords[ai] = ctr[ai] + ss * bh[ji] + tt * bv[ji]
            else:
                coords[ai] = ctr[ai]

        data = session.data
        if np.iscomplexobj(data):
            if complex_mode == 1:
                data_f = np.angle(data).astype(np.float32)
            elif complex_mode == 2:
                data_f = data.real.astype(np.float32)
            elif complex_mode == 3:
                data_f = data.imag.astype(np.float32)
            else:
                data_f = np.abs(data).astype(np.float32)
        else:
            data_f = np.nan_to_num(np.asarray(data, dtype=np.float32))

        sampled = map_coordinates(
            data_f, coords, order=0 if draft else 1, mode="constant", cval=0.0
        ).astype(np.float32)

        if log_scale:
            sampled = np.log1p(np.abs(sampled)).astype(np.float32)

        if vmin_override is not None and vmax_override is not None:
            vmin, vmax = float(vmin_override), float(vmax_override)
        else:
            vmin, vmax = _compute_vmin_vmax(session, sampled, dr, complex_mode)

        _ensure_lut(colormap)
        lut = LUTS.get(colormap, LUTS["gray"])
        if vmax > vmin:
            normalized = np.clip((sampled - vmin) / (vmax - vmin), 0, 1)
        else:
            normalized = np.zeros_like(sampled)
        rgba = lut[(normalized * 255).astype(np.uint8)]

        img = _pil_image().fromarray(rgba[:, :, :3], mode="RGB")
        if draft and (grid_w != size_w or grid_h != size_h):
            img = img.resize((size_w, size_h), _pil_image().NEAREST)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        return Response(
            content=buf.getvalue(),
            media_type="image/jpeg",
            headers={
                "Cache-Control": "no-store",
                "X-ArrayView-Vmin": str(vmin),
                "X-ArrayView-Vmax": str(vmax),
            },
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
        session=Depends(get_session_or_404),
    ):
        if session.rgb_axis is not None:
            return JSONResponse(
                status_code=400, content={"error": "not supported for RGB sessions"}
            )
        idx_list = [int(x) for x in indices.split(",")]
        n = session.shape[slice_dim]

        try:
            itemsize = np.dtype(session.data.dtype).itemsize
        except Exception:
            itemsize = 4
        frame_bytes = session.shape[dim_y] * session.shape[dim_x] * itemsize
        if frame_bytes * n > HEAVY_OP_LIMIT_BYTES:
            limit_mb = HEAVY_OP_LIMIT_BYTES // (1024 * 1024)
            est_mb = frame_bytes * n // (1024 * 1024)
            return JSONResponse(
                status_code=400,
                content={
                    "error": (
                        f"Grid blocked: would stack ~{est_mb} MB (limit {limit_mb} MB). "
                        "Increase ARRAYVIEW_HEAVY_OP_LIMIT_MB to override."
                    ),
                    "too_large": True,
                },
            )

        frames = []
        for i in range(n):
            idx_list[slice_dim] = i
            frames.append(extract_slice(session, dim_x, dim_y, idx_list))

        all_data = np.stack(frames)
        vmin = float(np.percentile(all_data, 1))
        vmax = float(np.percentile(all_data, 99))

        grid, _, _ = _build_mosaic_grid(all_data, n)

        nan_mask = np.isnan(grid)
        filled = np.where(nan_mask, vmin, grid)
        if vmax > vmin:
            normalized = np.clip((filled - vmin) / (vmax - vmin), 0, 1)
        else:
            normalized = np.zeros_like(filled)

        _init_luts()
        lut = LUTS.get(colormap if colormap in LUTS else "gray", LUTS["gray"])
        rgba = lut[(normalized * 255).astype(np.uint8)]
        rgba[nan_mask] = [22, 22, 22, 255]
        img = _pil_image().fromarray(rgba[:, :, :3], mode="RGB")
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
        session=Depends(get_session_or_404),
    ):
        if session.rgb_axis is not None:
            return JSONResponse(
                status_code=400, content={"error": "not supported for RGB sessions"}
            )
        idx_list = [int(x) for x in indices.split(",")]
        n = session.shape[slice_dim]

        try:
            itemsize = np.dtype(session.data.dtype).itemsize
        except Exception:
            itemsize = 4
        frame_bytes = session.shape[dim_y] * session.shape[dim_x] * itemsize
        if frame_bytes * n > HEAVY_OP_LIMIT_BYTES:
            limit_mb = HEAVY_OP_LIMIT_BYTES // (1024 * 1024)
            est_mb = frame_bytes * n // (1024 * 1024)
            return JSONResponse(
                status_code=400,
                content={
                    "error": (
                        f"GIF blocked: would stack ~{est_mb} MB (limit {limit_mb} MB). "
                        "Increase ARRAYVIEW_HEAVY_OP_LIMIT_MB to override."
                    ),
                    "too_large": True,
                },
            )

        frames = []
        for i in range(n):
            idx_list[slice_dim] = i
            frames.append(extract_slice(session, dim_x, dim_y, idx_list))

        all_data = np.stack(frames)
        vmin = float(np.percentile(all_data, 1))
        vmax = float(np.percentile(all_data, 99))

        _init_luts()
        lut = LUTS.get(colormap if colormap in LUTS else "gray", LUTS["gray"])
        gif_frames = []
        for frame in frames:
            if vmax > vmin:
                normalized = np.clip((frame - vmin) / (vmax - vmin), 0, 1)
            else:
                normalized = np.zeros_like(frame)
            rgba = lut[(normalized * 255).astype(np.uint8)]
            gif_frames.append(_pil_image().fromarray(rgba[:, :, :3], mode="RGB"))

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
