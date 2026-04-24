import asyncio
import io
import os

import numpy as np
from fastapi import Depends, HTTPException, Request, Response

from arrayview._io import load_data
from arrayview._session import HEAVY_OP_LIMIT_BYTES, SESSIONS, _estimate_array_bytes


def register_state_routes(app, get_session_or_404) -> None:
    @app.get("/clearcache/{sid}")
    def clear_cache(sid: str):
        session = SESSIONS.get(sid)
        if session:
            session.reset_caches()
        return {"status": "ok"}

    @app.get("/data_version/{sid}")
    def get_data_version(sid: str, session=Depends(get_session_or_404)):
        """Return the current data version for a session (incremented on reload)."""
        return {"version": getattr(session, "data_version", 0)}

    @app.post("/reload/{sid}")
    async def reload_session(sid: str, session=Depends(get_session_or_404)):
        """Reload session data from its source file (used by --watch mode)."""
        filepath = session.filepath
        if not filepath or not os.path.isfile(filepath):
            return {"error": "session has no reloadable filepath"}
        try:
            data = await asyncio.to_thread(load_data, filepath)
        except Exception as e:
            return {"error": str(e)}
        session.data = data
        session.shape = data.shape
        session.spatial_shape = data.shape
        session.rgb_axis = None
        session.fft_original_data = None
        session.fft_axes = None
        session.reset_caches()
        session.data_version = getattr(session, "data_version", 0) + 1
        return {"version": session.data_version}

    @app.post("/update/{sid}")
    async def update_session(sid: str, request: Request, session=Depends(get_session_or_404)):
        """Replace session data with a new numpy array sent as raw .npy bytes."""
        raw = await request.body()
        if not raw:
            return Response(status_code=400, content="empty body")
        try:
            arr = np.load(io.BytesIO(raw))
        except Exception as e:
            return Response(status_code=400, content=f"failed to decode array: {e}")
        session.data = arr
        session.shape = arr.shape
        session.spatial_shape = arr.shape
        session.rgb_axis = None
        session.fft_original_data = None
        session.fft_axes = None
        session.reset_caches()
        session.data_version = getattr(session, "data_version", 0) + 1
        return {"version": session.data_version}

    @app.get("/cache_info/{sid}")
    def cache_info(sid: str, session=Depends(get_session_or_404)):
        """Phase 5: debug endpoint — returns per-session cache usage and budgets."""
        return {
            "raw_cache": {
                "entries": len(session.raw_cache),
                "used_bytes": session._raw_bytes,
                "budget_bytes": session.RAW_CACHE_BYTES,
                "used_mb": round(session._raw_bytes / 1e6, 2),
                "budget_mb": round(session.RAW_CACHE_BYTES / 1e6, 2),
            },
            "rgba_cache": {
                "entries": len(session.rgba_cache),
                "used_bytes": session._rgba_bytes,
                "budget_bytes": session.RGBA_CACHE_BYTES,
                "used_mb": round(session._rgba_bytes / 1e6, 2),
                "budget_mb": round(session.RGBA_CACHE_BYTES / 1e6, 2),
            },
            "mosaic_cache": {
                "entries": len(session.mosaic_cache),
                "used_bytes": session._mosaic_bytes,
                "budget_bytes": session.MOSAIC_CACHE_BYTES,
                "used_mb": round(session._mosaic_bytes / 1e6, 2),
                "budget_mb": round(session.MOSAIC_CACHE_BYTES / 1e6, 2),
            },
            "heavy_op_limit_mb": round(HEAVY_OP_LIMIT_BYTES / 1e6, 1),
        }

    @app.post("/alpha/{sid}")
    async def set_alpha(sid: str, request: Request):
        """Toggle alpha (0=off, 1=transparent below vmin)."""
        session = SESSIONS.get(sid)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        body = await request.json()
        level = 1 if int(body.get("level", 0)) else 0
        session.alpha_level = level
        session.rgba_cache.clear()
        session._rgba_bytes = 0
        return {"level": level}

    @app.post("/resample_ras/{sid}")
    async def resample_ras(sid: str, request: Request):
        """Toggle RAS resample for a NIfTI session."""
        import time

        session = SESSIONS.get(sid)
        if session is None:
            return {"error": "session not found"}
        body = await request.json()
        enabled = bool(body.get("enabled", False))
        sm = getattr(session, "spatial_meta", None)
        if sm is None or session.original_volume is None:
            return {"skipped": True, "reason": "not_nifti"}

        if enabled:
            if not sm["is_oblique"]:
                return {"skipped": True, "reason": "axis_aligned"}
            t0 = time.time()
            if session.resampled_volume is None:
                try:
                    from scipy.ndimage import affine_transform
                except ImportError:
                    return {"error": "scipy not available"}
                vol = session.original_volume
                if vol.ndim != 3:
                    return {"skipped": True, "reason": "ndim_not_3"}
                affine_canonical = sm["affine_canonical"]
                rot = np.asarray(affine_canonical[:3, :3], dtype=np.float64)
                iso = float(min(sm["voxel_sizes"]))
                shp = vol.shape
                corners_idx = np.array(
                    [
                        [i, j, k]
                        for i in (0, shp[0] - 1)
                        for j in (0, shp[1] - 1)
                        for k in (0, shp[2] - 1)
                    ],
                    dtype=np.float64,
                )
                origin = np.asarray(affine_canonical[:3, 3], dtype=np.float64)
                ras = corners_idx @ rot.T + origin
                ras_min = ras.min(axis=0)
                ras_max = ras.max(axis=0)
                out_shape = tuple(
                    int(np.ceil((ras_max[i] - ras_min[i]) / iso)) + 1
                    for i in range(3)
                )
                inv_rot = np.linalg.inv(rot)
                matrix = inv_rot * iso
                offset = inv_rot @ (ras_min - origin)
                try:
                    resampled = affine_transform(
                        np.asarray(vol),
                        matrix=matrix,
                        offset=offset,
                        output_shape=out_shape,
                        order=1,
                        cval=0.0,
                        prefilter=False,
                    )
                except Exception as exc:
                    return {"error": f"resample failed: {exc}"}
                session.resampled_volume = resampled
            session.data = session.resampled_volume
            session.shape = session.resampled_volume.shape
            if session.rgb_axis is None:
                session.spatial_shape = session.shape
            session.ras_resample_active = True
            session.raw_cache.clear()
            session.rgba_cache.clear()
            session.mosaic_cache.clear()
            session._raw_bytes = session._rgba_bytes = session._mosaic_bytes = 0
            return {
                "ok": True,
                "shape": list(session.shape),
                "elapsed_ms": int((time.time() - t0) * 1000),
            }

        session.data = session.original_volume
        session.shape = session.original_volume.shape
        if session.rgb_axis is None:
            session.spatial_shape = session.shape
        session.ras_resample_active = False
        session.raw_cache.clear()
        session.rgba_cache.clear()
        session.mosaic_cache.clear()
        session._raw_bytes = session._rgba_bytes = session._mosaic_bytes = 0
        return {"ok": True, "shape": list(session.shape)}

    @app.post("/fft/{sid}")
    async def toggle_fft(sid: str, request: Request):
        session = SESSIONS.get(sid)
        if not session:
            return {"error": "Invalid session"}

        body = await request.json()
        axes_str = str(body.get("axes", "")).strip()

        if session.fft_original_data is not None:
            session.data = session.fft_original_data
            session.shape = session.data.shape
            session.fft_original_data = None
            session.fft_axes = None
            session.reset_caches()
            return {
                "status": "restored",
                "is_complex": bool(np.iscomplexobj(session.data)),
            }

        try:
            axes = tuple(int(a.strip()) for a in axes_str.split(",") if a.strip())
            if not axes:
                raise ValueError("No axes specified")
        except Exception as e:
            return {"error": str(e)}

        est = _estimate_array_bytes(session)
        if est > HEAVY_OP_LIMIT_BYTES:
            limit_mb = HEAVY_OP_LIMIT_BYTES // (1024 * 1024)
            est_mb = est // (1024 * 1024)
            return {
                "error": (
                    f"FFT blocked: array is ~{est_mb} MB (limit {limit_mb} MB). "
                    "Convert to a smaller sub-volume or increase "
                    "ARRAYVIEW_HEAVY_OP_LIMIT_MB."
                ),
                "too_large": True,
            }

        session.fft_original_data = session.data
        full = np.array(session.data)
        session.data = np.fft.fftshift(np.fft.fftn(full, axes=axes), axes=axes)
        session.shape = session.data.shape
        session.fft_axes = axes
        session.reset_caches()
        return {
            "status": "fft_applied",
            "axes": list(axes),
            "is_complex": bool(np.iscomplexobj(session.data)),
        }

    @app.post("/set_rgb/{sid}")
    async def set_rgb_endpoint(sid: str, request: Request):
        """Toggle RGB rendering for a session."""
        session = SESSIONS.get(sid)
        if not session:
            return {"error": "session not found"}
        body = await request.json()
        axis = body.get("axis")
        if axis is None:
            session.rgb_axis = None
            session.spatial_shape = session.data.shape
        else:
            axis = int(axis)
            if not (0 <= axis < len(session.data.shape)):
                return {
                    "error": f"axis {axis} out of range for shape {list(session.data.shape)}"
                }
            if session.data.shape[axis] not in (3, 4):
                return {
                    "error": f"dim {axis} has size {session.data.shape[axis]}, need 3 or 4 for RGB/RGBA"
                }
            session.rgb_axis = axis
            session.spatial_shape = tuple(
                s for i, s in enumerate(session.data.shape) if i != axis
            )
        session.rgba_cache.clear()
        session._rgba_bytes = 0
        return {
            "ok": True,
            "is_rgb": session.rgb_axis is not None,
            "spatial_shape": list(session.spatial_shape),
        }