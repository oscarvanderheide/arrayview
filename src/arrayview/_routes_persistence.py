import json
import os
import threading
from datetime import datetime, timezone

from fastapi import Request
from fastapi.responses import JSONResponse

from arrayview._session import SESSIONS


_OBLIQUE_RECENT_FILE = os.path.expanduser("~/.arrayview/oblique_recent.json")
_OBLIQUE_LOCK = threading.Lock()

_CROP_RECENT_FILE = os.path.expanduser("~/.arrayview/crop_recent.json")
_CROP_LOCK = threading.Lock()
_CROP_STATE: dict[str, dict] = {}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clamp_int(value, lo: int, hi: int, fallback: int) -> int:
    try:
        v = int(value)
    except Exception:
        v = int(fallback)
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


def _normalize_oblique_preset(
    preset: dict | None,
    *,
    ndim: int,
    shape: tuple[int, ...],
) -> dict | None:
    if not isinstance(preset, dict):
        return None
    try:
        shape_out = [int(v) for v in preset.get("shape", list(shape))]
    except Exception:
        return None
    if len(shape_out) != ndim:
        return None
    if any(int(v) <= 0 for v in shape_out):
        return None

    try:
        mv_dims = [int(v) for v in preset.get("mv_dims", [])]
    except Exception:
        return None
    if len(mv_dims) != 3 or len(set(mv_dims)) != 3:
        return None
    if any((d < 0 or d >= ndim) for d in mv_dims):
        return None

    try:
        indices = [int(v) for v in preset.get("indices", [])]
    except Exception:
        return None
    if len(indices) != ndim:
        return None
    for d in range(ndim):
        n = max(1, int(shape_out[d]))
        indices[d] = _clamp_int(indices[d], 0, n - 1, n // 2)

    vecs_raw = preset.get("oblique_vecs")
    if not isinstance(vecs_raw, list) or len(vecs_raw) != 3:
        return None
    vecs_out: list[dict[str, list[float]]] = []
    for item in vecs_raw:
        if not isinstance(item, dict):
            return None
        try:
            bh = [float(v) for v in item.get("bh", [])]
            bv = [float(v) for v in item.get("bv", [])]
            nn = [float(v) for v in item.get("n", [])]
        except Exception:
            return None
        if len(bh) != 3 or len(bv) != 3 or len(nn) != 3:
            return None
        vecs_out.append({"bh": bh, "bv": bv, "n": nn})

    pane_labels = preset.get("pane_labels")
    if not isinstance(pane_labels, list) or len(pane_labels) != 3:
        pane_labels = ["Oblique A", "Oblique B", "Oblique C"]
    pane_labels = [str(v) for v in pane_labels]

    pane_defs_raw = preset.get("pane_defs")
    pane_defs_out = None
    if isinstance(pane_defs_raw, list) and len(pane_defs_raw) == 3:
        tmp_defs: list[dict[str, int]] = []
        ok_defs = True
        for pane_def in pane_defs_raw:
            if not isinstance(pane_def, dict):
                ok_defs = False
                break
            try:
                dx = int(pane_def.get("dim_x"))
                dy = int(pane_def.get("dim_y"))
                sd = int(pane_def.get("slice_dir"))
            except Exception:
                ok_defs = False
                break
            if any((d < 0 or d >= ndim) for d in (dx, dy, sd)):
                ok_defs = False
                break
            if len({dx, dy, sd}) != 3:
                ok_defs = False
                break
            tmp_defs.append({"dim_x": dx, "dim_y": dy, "slice_dir": sd})
        if ok_defs:
            pane_defs_out = tmp_defs

    out = {
        "version": int(preset.get("version", 1)),
        "shape": shape_out,
        "mv_dims": mv_dims,
        "indices": indices,
        "oblique_vecs": vecs_out,
        "pane_labels": pane_labels,
    }
    if pane_defs_out is not None:
        out["pane_defs"] = pane_defs_out
    lock_raw = preset.get("oblique_ortho_lock")
    if isinstance(lock_raw, bool):
        out["oblique_ortho_lock"] = lock_raw
    return out


def _load_recent_oblique_file(path: str, *, ndim: int, shape: tuple[int, ...]) -> dict | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None
    return _normalize_oblique_preset(data, ndim=ndim, shape=shape)


def _write_recent_oblique_file(path: str, session, preset: dict) -> tuple[bool, str | None]:
    try:
        folder = os.path.dirname(path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        payload = dict(preset)
        payload["saved_at"] = _utc_now_iso()
        payload["sid"] = getattr(session, "sid", None)
        payload["name"] = getattr(session, "name", "") or ""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return True, None
    except Exception as exc:
        return False, str(exc)


def _clamp_crop_range(x_start, x_end, nx: int) -> tuple[int, int]:
    nx = max(1, int(nx))
    default_start = max(0, nx // 4)
    default_end = min(nx, (3 * nx) // 4)
    if default_end <= default_start:
        default_end = min(nx, default_start + 1)
    xs = _clamp_int(x_start, 0, nx - 1, default_start)
    xe = _clamp_int(x_end, 1, nx, default_end)
    if xe <= xs:
        xe = min(nx, xs + 1)
    if xs >= xe:
        xs = max(0, xe - 1)
    return int(xs), int(xe)


def _load_recent_crop_file(path: str, nx: int) -> dict | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return None
        xs, xe = _clamp_crop_range(data.get("x_start"), data.get("x_end"), nx)
        out: dict = {"x_start": xs, "x_end": xe}
        if "viz_z" in data:
            try:
                out["viz_z"] = int(data["viz_z"])
            except Exception:
                pass
        if "viz_y" in data:
            try:
                out["viz_y"] = int(data["viz_y"])
            except Exception:
                pass
        return out
    except Exception:
        return None


def _write_recent_crop_file(path: str, state: dict) -> tuple[bool, str | None]:
    try:
        folder = os.path.dirname(path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        payload = {
            "x_start": int(state["x_start"]),
            "x_end": int(state["x_end"]),
            "nx": int(state["nx"]),
            "readout_dim": int(state.get("readout_dim", -1)),
            "viz_z": int(state.get("viz_z", -1)),
            "viz_y": int(state.get("viz_y", -1)),
            "saved_at": _utc_now_iso(),
            "sid": state.get("sid"),
            "name": state.get("name", ""),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return True, None
    except Exception as exc:
        return False, str(exc)


def register_persistence_routes(app) -> None:
    @app.post("/oblique/save")
    async def oblique_save(request: Request):
        body = await request.json()
        sid = str(body.get("sid") or "").strip()
        if not sid:
            return JSONResponse({"error": "missing_sid"}, status_code=400)
        session = SESSIONS.get(sid)
        if session is None:
            return JSONResponse({"error": "session_not_found"}, status_code=404)
        shape = tuple(int(v) for v in getattr(session, "shape", ()) or ())
        ndim = len(shape)
        if ndim <= 0:
            return JSONResponse({"error": "invalid_session_shape"}, status_code=500)
        preset = _normalize_oblique_preset(body.get("preset"), ndim=ndim, shape=shape)
        if preset is None:
            return JSONResponse({"error": "invalid_oblique_preset"}, status_code=400)
        with _OBLIQUE_LOCK:
            ok, err = _write_recent_oblique_file(_OBLIQUE_RECENT_FILE, session, preset)
        if not ok:
            return JSONResponse(
                {"error": "oblique_save_failed", "detail": str(err)},
                status_code=500,
            )
        return JSONResponse(
            {
                "ok": True,
                "sid": sid,
                "saved_path": _OBLIQUE_RECENT_FILE,
                "preset": preset,
            }
        )

    @app.post("/oblique/load_recent")
    async def oblique_load_recent(request: Request):
        body = await request.json()
        sid = str(body.get("sid") or "").strip()
        if not sid:
            return JSONResponse({"error": "missing_sid"}, status_code=400)
        session = SESSIONS.get(sid)
        if session is None:
            return JSONResponse({"error": "session_not_found"}, status_code=404)
        shape = tuple(int(v) for v in getattr(session, "shape", ()) or ())
        ndim = len(shape)
        if ndim <= 0:
            return JSONResponse({"error": "invalid_session_shape"}, status_code=500)
        with _OBLIQUE_LOCK:
            preset = _load_recent_oblique_file(
                _OBLIQUE_RECENT_FILE,
                ndim=ndim,
                shape=shape,
            )
        if preset is None:
            return JSONResponse(
                {"error": "oblique_recent_file_missing_or_invalid"},
                status_code=404,
            )
        return JSONResponse(
            {
                "ok": True,
                "sid": sid,
                "path": _OBLIQUE_RECENT_FILE,
                "preset": preset,
            }
        )

    @app.post("/crop/register")
    async def crop_register(request: Request):
        body = await request.json()
        sid = str(body.get("sid") or "").strip()
        if not sid:
            return JSONResponse({"error": "missing_sid"}, status_code=400)
        session = SESSIONS.get(sid)
        if session is None:
            return JSONResponse({"error": "session_not_found"}, status_code=404)

        shape = tuple(int(s) for s in getattr(session, "spatial_shape", ()) or ())
        if not shape:
            shape = tuple(int(s) for s in getattr(session, "shape", ()) or ())
        ndim = len(shape)
        if ndim <= 0:
            return JSONResponse({"error": "invalid_session_shape"}, status_code=500)

        readout_dim = _clamp_int(
            body.get("readout_dim", ndim - 1),
            0,
            max(0, ndim - 1),
            ndim - 1,
        )
        nx = int(shape[readout_dim])
        xs, xe = _clamp_crop_range(body.get("x_start"), body.get("x_end"), nx)

        non_readout_dims = [d for d in range(ndim) if d != readout_dim]
        default_viz_z_dim = non_readout_dims[0] if non_readout_dims else -1
        default_viz_y_dim = non_readout_dims[1] if len(non_readout_dims) > 1 else -1
        viz_z = int(
            body.get(
                "viz_z",
                shape[default_viz_z_dim] // 2 if default_viz_z_dim >= 0 else -1,
            )
        )
        viz_y = int(
            body.get(
                "viz_y",
                shape[default_viz_y_dim] // 2 if default_viz_y_dim >= 0 else -1,
            )
        )

        raw_recent = body.get("recent_file")
        recent_file = str(raw_recent).strip() if raw_recent is not None else ""
        if not recent_file:
            recent_file = _CROP_RECENT_FILE
        loaded_recent = False
        if recent_file:
            recent = _load_recent_crop_file(recent_file, nx)
            if recent is not None:
                xs, xe = int(recent["x_start"]), int(recent["x_end"])
                if "viz_z" in recent:
                    viz_z = int(recent["viz_z"])
                if "viz_y" in recent:
                    viz_y = int(recent["viz_y"])
                loaded_recent = True

        state = {
            "sid": sid,
            "name": str(body.get("name") or getattr(session, "name", "") or ""),
            "shape": list(shape),
            "readout_dim": int(readout_dim),
            "nx": int(nx),
            "x_start": int(xs),
            "x_end": int(xe),
            "viz_z": int(viz_z),
            "viz_y": int(viz_y),
            "confirmed": False,
            "save_requested": False,
            "saved_recent": False,
            "saved_recent_path": None,
            "recent_file": recent_file,
            "loaded_recent": bool(loaded_recent),
            "updated_at": _utc_now_iso(),
        }
        with _CROP_LOCK:
            _CROP_STATE[sid] = state
        return JSONResponse(state)

    @app.get("/crop/state/{sid}")
    def crop_state(sid: str):
        with _CROP_LOCK:
            state = _CROP_STATE.get(sid)
            if state is None:
                return JSONResponse({"error": "crop_session_not_found"}, status_code=404)
            return JSONResponse(dict(state))

    @app.post("/crop/update")
    async def crop_update(request: Request):
        body = await request.json()
        sid = str(body.get("sid") or "").strip()
        if not sid:
            return JSONResponse({"error": "missing_sid"}, status_code=400)
        with _CROP_LOCK:
            state = _CROP_STATE.get(sid)
            if state is None:
                return JSONResponse({"error": "crop_session_not_found"}, status_code=404)
            if "readout_dim" in body:
                shape = tuple(int(s) for s in state.get("shape", ()) or ())
                ndim = len(shape)
                if ndim > 0:
                    new_rd = _clamp_int(
                        body.get("readout_dim"),
                        0,
                        max(0, ndim - 1),
                        int(state.get("readout_dim", ndim - 1)),
                    )
                    if new_rd != int(state.get("readout_dim", -1)):
                        new_nx = int(shape[new_rd])
                        state["readout_dim"] = int(new_rd)
                        state["nx"] = int(new_nx)
                        if "x_start" not in body and "x_end" not in body:
                            state["x_start"] = 0
                            state["x_end"] = int(new_nx)
                        state["confirmed"] = False
                        state["loaded_recent"] = False
            xs, xe = _clamp_crop_range(
                body.get("x_start", state.get("x_start")),
                body.get("x_end", state.get("x_end")),
                int(state["nx"]),
            )
            state["x_start"] = int(xs)
            state["x_end"] = int(xe)
            if "viz_z" in body:
                state["viz_z"] = int(body.get("viz_z", state.get("viz_z", -1)))
            if "viz_y" in body:
                state["viz_y"] = int(body.get("viz_y", state.get("viz_y", -1)))
            state["updated_at"] = _utc_now_iso()
            return JSONResponse(dict(state))

    @app.post("/crop/load_recent")
    async def crop_load_recent(request: Request):
        body = await request.json()
        sid = str(body.get("sid") or "").strip()
        if not sid:
            return JSONResponse({"error": "missing_sid"}, status_code=400)
        with _CROP_LOCK:
            state = _CROP_STATE.get(sid)
            if state is None:
                return JSONResponse({"error": "crop_session_not_found"}, status_code=404)
            recent_file = state.get("recent_file") or _CROP_RECENT_FILE
            recent = _load_recent_crop_file(str(recent_file), int(state["nx"]))
            if recent is None:
                return JSONResponse(
                    {"error": "recent_file_missing_or_invalid"},
                    status_code=404,
                )
            state["x_start"] = int(recent["x_start"])
            state["x_end"] = int(recent["x_end"])
            if "viz_z" in recent:
                state["viz_z"] = int(recent["viz_z"])
            if "viz_y" in recent:
                state["viz_y"] = int(recent["viz_y"])
            state["loaded_recent"] = True
            state["updated_at"] = _utc_now_iso()
            return JSONResponse(dict(state))

    @app.post("/crop/confirm")
    async def crop_confirm(request: Request):
        body = await request.json()
        sid = str(body.get("sid") or "").strip()
        if not sid:
            return JSONResponse({"error": "missing_sid"}, status_code=400)
        with _CROP_LOCK:
            state = _CROP_STATE.get(sid)
            if state is None:
                return JSONResponse({"error": "crop_session_not_found"}, status_code=404)
            state["confirmed"] = True
            state["save_requested"] = True
            state["saved_recent"] = False
            state["saved_recent_path"] = None
            state["updated_at"] = _utc_now_iso()
            target = state.get("recent_file") or _CROP_RECENT_FILE
            ok, err = _write_recent_crop_file(str(target), state)
            if ok:
                state["saved_recent"] = True
                state["saved_recent_path"] = str(target)
            else:
                state["save_error"] = err
            return JSONResponse(dict(state))

    @app.post("/crop/clear")
    async def crop_clear(request: Request):
        body = await request.json()
        sid = str(body.get("sid") or "").strip()
        if not sid:
            return JSONResponse({"error": "missing_sid"}, status_code=400)
        with _CROP_LOCK:
            _CROP_STATE.pop(sid, None)
        return JSONResponse({"ok": True, "sid": sid})
