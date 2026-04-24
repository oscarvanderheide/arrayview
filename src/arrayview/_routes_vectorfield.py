import numpy as np
from fastapi import Request, Response

from arrayview._io import load_data
from arrayview._session import SESSIONS
from arrayview._vectorfield import (
    _MAX_VFIELD_ARROWS,
    _compute_vfield_arrows,
    _configure_vectorfield,
    _get_vfield_layout,
    _vfield_counts_for_level,
)


def register_vectorfield_routes(app) -> None:
    @app.get("/vectorfield/{sid}")
    def get_vectorfield(
        sid: str,
        dim_x: int,
        dim_y: int,
        indices: str,
        t_index: int = 0,
        density_offset: int = 0,
    ):
        """Return downsampled deformation vector field arrows for the current 2-D view."""
        session = SESSIONS.get(sid)
        if not session or session.vfield is None:
            return Response(status_code=404)
        try:
            idx_tuple = tuple(int(x) for x in indices.split(","))
            result = _compute_vfield_arrows(
                session, dim_x, dim_y, idx_tuple, t_index, density_offset
            )
            if result is None:
                return Response(status_code=404)
            return {
                "arrows": result["arrows"].tolist(),
                "scale": result["scale"],
                "stride": result["stride"],
            }
        except Exception as e:
            import traceback

            traceback.print_exc()
            return Response(
                status_code=500, content=str(e).encode(), media_type="text/plain"
            )

    @app.post("/attach_vectorfield")
    async def attach_vectorfield(request: Request):
        """Attach a vector field to an existing session."""
        body = await request.json()
        sid = str(body["sid"])
        filepath = str(body["filepath"])
        components_dim = body.get("components_dim")
        session = SESSIONS.get(sid)
        if not session:
            return {"error": f"session {sid} not found"}
        try:
            vf_data = load_data(filepath)
            layout = _configure_vectorfield(session, vf_data, components_dim)
            return {"ok": True, "components_dim": layout["components_dim"]}
        except Exception as e:
            return {"error": str(e)}

    @app.get("/oblique_vectorfield/{sid}")
    def get_oblique_vectorfield(
        sid: str,
        center: str,
        basis_h: str,
        basis_v: str,
        mv_dims: str,
        size_w: int,
        size_h: int,
        density_offset: int = 0,
        t_index: int = 0,
    ):
        """Return vector field arrows projected onto an oblique slice plane."""
        session = SESSIONS.get(sid)
        if not session or session.vfield is None:
            return Response(status_code=404)

        layout = _get_vfield_layout(session)
        if layout is None:
            return Response(status_code=404)

        try:
            from scipy.ndimage import map_coordinates

            ctr = [float(x) for x in center.split(",")]
            bh = np.array([float(x) for x in basis_h.split(",")], dtype=np.float64)
            bv = np.array([float(x) for x in basis_v.split(",")], dtype=np.float64)
            dims = [int(x) for x in mv_dims.split(",")]

            vf = session.vfield
            spatial_axes = tuple(int(ax) for ax in layout["spatial_axes"])
            comp_dim = int(layout["components_dim"])
            time_dim = layout["time_dim"]

            slices = [slice(None)] * vf.ndim
            if time_dim is not None:
                t = max(0, min(int(layout["n_times"]) - 1, t_index))
                slices[int(time_dim)] = t

            vf_t = np.asarray(vf[tuple(slices)], dtype=np.float32)

            height, width = size_h, size_w
            base_stride = max(1, max(height, width) // 32)
            n_arrows_target, effective_stride, use_grid = _vfield_counts_for_level(
                density_offset, height, width, base_stride
            )
            if use_grid:
                ys = np.arange(0, height, dtype=int)
                xs = np.arange(0, width, dtype=int)
                gy_grid, gx_grid = np.meshgrid(ys, xs, indexing="ij")
                gy = gy_grid.ravel()
                gx = gx_grid.ravel()
                if gy.size > _MAX_VFIELD_ARROWS:
                    keep = np.linspace(0, gy.size - 1, _MAX_VFIELD_ARROWS).astype(int)
                    gy = gy[keep]
                    gx = gx[keep]
            else:
                n_arrows = min(n_arrows_target, _MAX_VFIELD_ARROWS)
                rng = np.random.default_rng(int(height) * 10007 + int(width))
                gy = rng.integers(0, height, n_arrows).astype(int)
                gx = rng.integers(0, width, n_arrows).astype(int)
            n_arrows = gy.size

            half_w, half_h = width / 2.0, height / 2.0
            sx = gx.astype(np.float64) - half_w
            sy = gy.astype(np.float64) - half_h

            remaining_axes = [ax for ax, sl in enumerate(slices) if isinstance(sl, slice)]
            axis_map = {ax: i for i, ax in enumerate(remaining_axes)}

            n_comp = vf_t.shape[axis_map[comp_dim]]
            n_spatial = len(spatial_axes)
            comp_offset = n_spatial - n_comp

            ndim = len(ctr)
            coords_3d = np.empty((ndim, n_arrows), dtype=np.float64)
            for ai in range(ndim):
                if ai in dims:
                    ji = dims.index(ai)
                    coords_3d[ai] = ctr[ai] + sx * bh[ji] + sy * bv[ji]
                else:
                    coords_3d[ai] = ctr[ai]

            spatial_remaining = [ax for ax in remaining_axes if ax != comp_dim]
            sp_map = {ax: i for i, ax in enumerate(spatial_remaining)}
            mc_coords = np.empty((len(spatial_remaining), n_arrows), dtype=np.float64)
            for ax in spatial_remaining:
                mc_coords[sp_map[ax]] = coords_3d[ax] if ax < ndim else 0.0

            vecs_h = np.zeros(n_arrows, dtype=np.float64)
            vecs_v = np.zeros(n_arrows, dtype=np.float64)

            for ci in range(n_comp):
                comp_slices = [slice(None)] * vf_t.ndim
                comp_slices[axis_map[comp_dim]] = ci
                vf_comp = vf_t[tuple(comp_slices)]

                sampled = map_coordinates(
                    vf_comp, mc_coords, order=1, mode="constant", cval=0.0
                )

                spatial_idx = ci + comp_offset
                if spatial_idx < n_spatial:
                    sa = spatial_axes[spatial_idx]
                    if sa in dims:
                        ji = dims.index(sa)
                        vecs_h += sampled * bh[ji]
                        vecs_v += sampled * bv[ji]

            vx_s = vecs_h.astype(np.float32)
            vy_s = vecs_v.astype(np.float32)
            mags = np.sqrt(vx_s**2 + vy_s**2)
            nonzero = mags[mags > 0]
            p95 = float(np.percentile(nonzero, 95)) if nonzero.size else 1.0
            scale = float(effective_stride * 0.75 / max(p95, 1e-9))

            arrows = np.column_stack([gx, gy, vx_s, vy_s]).astype(np.float32)
            return {
                "arrows": arrows.tolist(),
                "scale": scale,
                "stride": int(round(effective_stride)),
            }
        except Exception as e:
            import traceback

            traceback.print_exc()
            return Response(
                status_code=500, content=str(e).encode(), media_type="text/plain"
            )