"""Routes for reading and updating persistent user preferences."""

from __future__ import annotations

from copy import deepcopy
import os

from fastapi import Request
from fastapi.responses import JSONResponse

from arrayview._config import (
    BUILTIN_PREFERENCES,
    InvalidPreferenceError,
    MalformedConfigError,
    PREFERENCE_SCHEMA,
    get_preferences,
    update_preferences,
)
import arrayview._session as _session_mod
from arrayview._session import SESSIONS


def _response_payload() -> dict:
    defaults = deepcopy(BUILTIN_PREFERENCES)
    schema = deepcopy(PREFERENCE_SCHEMA)
    for key, options in schema["window"].items():
        schema["window"][key] = [value for value in options if value != "none"]
    schema["window"].pop("default", None)
    try:
        from arrayview._platform import _is_vscode_remote, _native_window_gui

        native_available = _native_window_gui() is not None
        vscode_remote = _is_vscode_remote()
    except Exception:
        native_available = False
        vscode_remote = False
    if not native_available:
        defaults["window"]["default"] = "browser"
        defaults["window"]["terminal"] = "browser"
        for key in ("default", "terminal", "vscode", "jupyter", "julia"):
            schema["window"][key] = [
                value for value in schema["window"][key] if value != "native"
            ]
    if vscode_remote:
        schema["window"]["vscode"] = ["vscode", "none"]

    overrides = {}
    if os.environ.get("ARRAYVIEW_WINDOW", "").strip():
        overrides["window"] = "ARRAYVIEW_WINDOW"
    if os.environ.get("ARRAYVIEW_NNINTERACTIVE_URL", "").strip():
        overrides["nninteractive.url"] = "ARRAYVIEW_NNINTERACTIVE_URL"
    return {
        "preferences": get_preferences(),
        "defaults": defaults,
        "schema": schema,
        "overrides": overrides,
        "server_instance_id": _session_mod.SERVER_RUNTIME.instance_id,
    }


def register_preferences_routes(app) -> None:
    @app.get("/preferences/{sid}")
    async def preferences_get(sid: str):
        if SESSIONS.get(sid) is None:
            return JSONResponse({"error": "session_not_found"}, status_code=404)
        return _response_payload()

    @app.patch("/preferences/{sid}")
    async def preferences_patch(sid: str, request: Request):
        if SESSIONS.get(sid) is None:
            return JSONResponse({"error": "session_not_found"}, status_code=404)
        content_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
        if content_type != "application/json":
            return JSONResponse({"error": "json_content_type_required"}, status_code=415)
        if request.headers.get("x-arrayview-preferences") != "1":
            return JSONResponse({"error": "preferences_header_required"}, status_code=403)
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "invalid_json"}, status_code=400)
        if not isinstance(body, dict):
            return JSONResponse({"error": "invalid_json"}, status_code=400)
        if body.get("server_instance_id") != _session_mod.SERVER_RUNTIME.instance_id:
            return JSONResponse({"error": "stale_server_instance"}, status_code=409)
        changes = body.get("changes")
        try:
            viewer_changes = changes.get("viewer", {}) if isinstance(changes, dict) else {}
            colormaps = viewer_changes.get("colormaps")
            if colormaps is not None:
                from arrayview._render import _ensure_lut

                invalid = [
                    name for name in colormaps
                    if not isinstance(name, str) or not _ensure_lut(name)
                ]
                if invalid:
                    raise InvalidPreferenceError(f"Unknown colormap: {invalid[0]}")
            update_preferences(changes)
        except InvalidPreferenceError as exc:
            return JSONResponse(
                {"error": "invalid_preference", "detail": str(exc)},
                status_code=422,
            )
        except MalformedConfigError as exc:
            return JSONResponse(
                {"error": "malformed_config", "detail": str(exc)},
                status_code=409,
            )
        except OSError as exc:
            return JSONResponse(
                {"error": "config_write_failed", "detail": str(exc)},
                status_code=500,
            )
        return _response_payload()
