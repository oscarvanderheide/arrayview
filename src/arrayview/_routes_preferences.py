"""Routes for reading and updating persistent user preferences."""

from __future__ import annotations

import os

from fastapi import Request
from fastapi.responses import JSONResponse

from arrayview._config import (
    InvalidPreferenceError,
    MalformedConfigError,
    PREFERENCE_SCHEMA,
    get_preferences,
    update_preferences,
)
from arrayview._session import SESSIONS


def _response_payload() -> dict:
    overrides = {}
    if os.environ.get("ARRAYVIEW_WINDOW", "").strip():
        overrides["window"] = "ARRAYVIEW_WINDOW"
    if os.environ.get("ARRAYVIEW_NNINTERACTIVE_URL", "").strip():
        overrides["nninteractive.url"] = "ARRAYVIEW_NNINTERACTIVE_URL"
    return {
        "preferences": get_preferences(),
        "schema": PREFERENCE_SCHEMA,
        "overrides": overrides,
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
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "invalid_json"}, status_code=400)
        changes = body.get("changes") if isinstance(body, dict) else None
        try:
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
