import asyncio
import io

import numpy as np
from fastapi import Body, Depends, Request, Response
from fastapi.responses import JSONResponse

from arrayview._render import render_rgba


def register_export_routes(app, *, get_session_or_404, pil_image) -> None:
    @app.get("/export_array/{sid}")
    def export_array(
        sid: str,
        save_to_downloads: int = 0,
        session=Depends(get_session_or_404),
    ):
        """Return the full N-D array as a downloadable .npy file."""
        data = np.asarray(session.data)
        buf = io.BytesIO()
        np.save(buf, data)
        buf.seek(0)
        name_stem = (session.name or "array").replace(" ", "_").replace("/", "_")
        filename = f"{name_stem}.npy"
        if save_to_downloads:
            import pathlib

            downloads = pathlib.Path.home() / "Downloads"
            dest = downloads / filename if downloads.is_dir() else pathlib.Path(filename)
            dest.write_bytes(buf.read())
            return JSONResponse({"path": str(dest)})
        return Response(
            content=buf.read(),
            media_type="application/octet-stream",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    @app.post("/save_file")
    async def save_file(request: Request):
        """Save a client-generated file to the Downloads folder."""
        body = await request.json()
        filename = body.get("filename", "arrayview_export")
        data_b64 = body.get("data", "")
        if "," in data_b64:
            data_b64 = data_b64.split(",", 1)[1]
        import base64
        import pathlib

        raw = base64.b64decode(data_b64)
        downloads = pathlib.Path.home() / "Downloads"
        dest = downloads / filename if downloads.is_dir() else pathlib.Path(filename)
        dest.write_bytes(raw)
        return JSONResponse({"path": str(dest)})

    @app.post("/exploded/{sid}")
    async def get_exploded_slices(
        sid: str,
        dim_x: int = Body(...),
        dim_y: int = Body(...),
        scroll_dim: int = Body(...),
        indices: list[int] = Body(...),
        width: int = Body(256),
        colormap: str = Body("gray"),
        dr: int = Body(1),
        complex_mode: int = Body(0),
        log_scale: bool = Body(False),
        vmin_override: float | None = Body(None),
        vmax_override: float | None = Body(None),
        session=Depends(get_session_or_404),
    ):
        """Return JPEG thumbnails for multiple slices along scroll_dim."""
        import base64

        ndim = len(session.shape)
        if ndim < 3:
            return JSONResponse({"error": "need >= 3D array"}, status_code=400)

        image_cls = pil_image()
        results = []
        base_indices = [size // 2 for size in session.shape]

        for slice_idx in indices:
            idx_list = list(base_indices)
            idx_list[scroll_dim] = min(max(0, slice_idx), session.shape[scroll_dim] - 1)

            rgba = await asyncio.to_thread(
                render_rgba,
                session,
                dim_x,
                dim_y,
                tuple(idx_list),
                colormap,
                dr,
                complex_mode,
                log_scale,
                vmin_override,
                vmax_override,
            )

            img = image_cls.fromarray(rgba[:, :, :3])
            aspect = img.height / img.width
            target_h = max(1, int(width * aspect))
            resample = image_cls.NEAREST if img.width <= width else image_cls.LANCZOS
            img = img.resize((width, target_h), resample)

            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=70)
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            results.append({"index": slice_idx, "image": f"data:image/jpeg;base64,{b64}"})

        return JSONResponse({"slices": results})