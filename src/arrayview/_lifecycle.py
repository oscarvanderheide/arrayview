"""Session lifecycle helpers shared by REST and WebSocket close paths."""

from __future__ import annotations

from arrayview._session import PENDING_SESSION_EVENTS, PENDING_SESSIONS, SESSIONS


def release_session(sid: str) -> bool:
    """Drop a loaded or pending session and release its cached array memory."""
    PENDING_SESSIONS.discard(sid)
    event = PENDING_SESSION_EVENTS.pop(sid, None)
    if event is not None:
        try:
            event.set()
        except Exception:
            pass

    session = SESSIONS.pop(sid, None)
    if session is None:
        return False

    try:
        session.reset_caches()
    except Exception:
        pass
    try:
        session.data = None
    except Exception:
        pass

    try:
        from arrayview._routes_persistence import _CROP_LOCK, _CROP_STATE

        with _CROP_LOCK:
            _CROP_STATE.pop(sid, None)
    except Exception:
        pass

    return True
