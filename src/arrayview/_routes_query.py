import io

import numpy as np
from fastapi import Depends, HTTPException, Request, Response

from arrayview._analysis import _build_metadata
from arrayview._render import render_rgb_rgba, render_rgba
from arrayview._session import SESSIONS, wait_for_session_ready
import arrayview._session as _session_mod


def _viewer_related_sids(body: dict, primary_sid: str) -> list[str]:
    """Return validated, deduplicated related sessions from a viewer URL."""
    related_sids: list[str] = []
    for field in ("compare_sid", "compare_sids", "overlay_sid"):
        raw_value = body.get(field)
        if raw_value is None:
            continue
        if not isinstance(raw_value, str) or len(raw_value) > 8192:
            raise HTTPException(status_code=400, detail="Invalid related session IDs")
        for value in raw_value.split(","):
            related_sid = value.strip()
            if not related_sid or related_sid == primary_sid:
                continue
            if len(related_sid) > 128 or related_sid not in SESSIONS:
                raise HTTPException(status_code=409, detail="Related session changed")
            if related_sid not in related_sids:
                related_sids.append(related_sid)
            if len(related_sids) > 128:
                raise HTTPException(status_code=400, detail="Too many related sessions")
    return related_sids


def register_query_routes(app, *, get_session_or_404, pil_image, pil_imageops) -> None:
    @app.post("/viewer-phase/{sid}/{request_id}")
    async def record_viewer_phase(sid: str, request_id: str, request: Request):
        """Record browser-observed launch phases for remote display readiness."""
        if not request_id or len(request_id) > 128:
            raise HTTPException(status_code=400, detail="Invalid launch request ID")
        body = await request.json()
        expected_server_id = body.get("server_id")
        if expected_server_id != _session_mod.SERVER_RUNTIME.instance_id:
            raise HTTPException(status_code=409, detail="ArrayView server generation changed")
        phase = str(body.get("phase") or "")
        token = str(body.get("token") or "")
        window_id = str(body.get("window_id") or "")
        viewer_instance_id = str(body.get("viewer_instance_id") or "")
        if not token or len(token) > 256 or not window_id:
            raise HTTPException(status_code=400, detail="Incomplete viewer phase identity")
        allowed = {
            "script-loaded",
            "ws-open",
            "metadata-loaded",
            "frame-rendered",
        }
        if phase != "launch-prepared" and phase not in allowed:
            raise HTTPException(status_code=400, detail="Invalid viewer phase")
        if phase != "launch-prepared" and not viewer_instance_id:
            raise HTTPException(status_code=400, detail="Incomplete viewer phase identity")
        session = await wait_for_session_ready(sid)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")
        journals = getattr(session, "viewer_phase_journals", None)
        if journals is None:
            journals = session.viewer_phase_journals = {}
        if phase == "launch-prepared":
            previous_journal = journals.get(request_id)
            previous_release_task = (
                previous_journal.get("release_task")
                if previous_journal is not None
                else None
            )
            if (
                previous_release_task is not None
                and not previous_release_task.done()
            ):
                previous_release_task.cancel()
            journal = journals[request_id] = {
                "token": token,
                "window_id": window_id,
                "viewer_instance_ids": [],
                "phases": [],
                "related_sids": None,
                "connection_count": 0,
                "release_task": None,
                "disconnect_release_grace_seconds": 30.0,
            }
            return {
                "sid": sid,
                "request_id": request_id,
                "window_id": window_id,
                "server_id": _session_mod.SERVER_RUNTIME.instance_id,
                "token": token,
                "phases": [],
                "viewer_instance_ids": [],
                "related_sids": [],
            }
        journal = journals.get(request_id)
        if (
            journal is None
            or journal["token"] != token
            or journal["window_id"] != window_id
        ):
            raise HTTPException(status_code=409, detail="Viewer phase owner changed")
        if body.get("sid") != sid:
            raise HTTPException(status_code=409, detail="Viewer primary session changed")
        related_sids = _viewer_related_sids(body, sid)
        if journal["related_sids"] is None:
            journal["related_sids"] = related_sids
        elif journal["related_sids"] != related_sids:
            raise HTTPException(status_code=409, detail="Viewer related sessions changed")
        if viewer_instance_id not in journal["viewer_instance_ids"]:
            journal["viewer_instance_ids"].append(viewer_instance_id)
        phases = journal["phases"]
        if phase not in phases:
            phases.append(phase)
        if bool(body.get("release_on_disconnect")):
            journal["disconnect_release_grace_seconds"] = 30.0
        return {
            "sid": sid,
            "request_id": request_id,
            "window_id": window_id,
            "server_id": _session_mod.SERVER_RUNTIME.instance_id,
            "token": token,
            "phases": list(phases),
            "viewer_instance_ids": list(journal["viewer_instance_ids"]),
            "related_sids": list(journal["related_sids"]),
        }

    @app.get("/viewer-phase/{sid}/{request_id}")
    async def get_viewer_phases(sid: str, request_id: str, token: str):
        """Return phases correlated to one browser launch transaction."""
        session = SESSIONS.get(sid)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")
        journals = getattr(session, "viewer_phase_journals", {})
        journal = journals.get(request_id, {})
        if not journal or journal.get("token") != token:
            raise HTTPException(status_code=409, detail="Viewer phase owner changed")
        return {
            "sid": sid,
            "request_id": request_id,
            "window_id": journal.get("window_id"),
            "server_id": _session_mod.SERVER_RUNTIME.instance_id,
            "token": journal.get("token"),
            "phases": list(journal.get("phases", [])),
            "viewer_instance_ids": list(
                journal.get("viewer_instance_ids", [])
            ),
            "related_sids": list(journal.get("related_sids") or []),
        }

    @app.get("/autocrop/{sid}")
    def get_autocrop_bounds(
        sid: str,
        indices: str,
        margin_mm: float = 3.0,
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
    def get_info(
        sid: str,
        include_overlay_labels: bool = False,
        session=Depends(get_session_or_404),
    ):
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
        if include_overlay_labels:
            info["overlay_labels"] = []
            try:
                if np.issubdtype(session.data.dtype, np.integer):
                    labels = np.unique(session.data)
                    labels = labels[labels > 0]
                    if labels.size <= 16:
                        info["overlay_labels"] = [int(value) for value in labels]
            except (AttributeError, TypeError, ValueError):
                pass
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
            if spatial_meta.get("dicom_meta") is not None:
                info["dicom_meta"] = spatial_meta["dicom_meta"]
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
