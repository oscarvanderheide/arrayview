"""Session lifecycle helpers shared by REST and WebSocket close paths."""

from __future__ import annotations

import shutil
import threading

from arrayview._session import PENDING_SESSION_EVENTS, PENDING_SESSIONS, SESSIONS

_SESSION_LEASE_LOCK = threading.Lock()


def acquire_session_leases(sids: list[str]) -> bool:
    """Atomically record another viewer tab using related sessions."""
    with _SESSION_LEASE_LOCK:
        sessions = [SESSIONS.get(sid) for sid in sids]
        if any(session is None for session in sessions):
            return False
        for session in sessions:
            session.viewer_leases = (
                max(1, int(getattr(session, "viewer_leases", 1))) + 1
            )
        return True


def release_session(sid: str) -> bool:
    """Release one viewer lease and drop the session after the final lease."""
    PENDING_SESSIONS.discard(sid)
    event = PENDING_SESSION_EVENTS.pop(sid, None)
    if event is not None:
        try:
            event.set()
        except Exception:
            pass

    with _SESSION_LEASE_LOCK:
        session = SESSIONS.get(sid)
        if session is None:
            return False

        leases = max(1, int(getattr(session, "viewer_leases", 1)))
        if leases > 1:
            session.viewer_leases = leases - 1
            return True

        SESSIONS.pop(sid, None)

    try:
        session.reset_caches()
    except Exception:
        pass
    try:
        session.data = None
    except Exception:
        pass
    staging_dir = getattr(session, "_drop_staging_dir", None)
    if staging_dir:
        shutil.rmtree(staging_dir, ignore_errors=True)

    try:
        from arrayview._routes_persistence import _CROP_LOCK, _CROP_STATE

        with _CROP_LOCK:
            _CROP_STATE.pop(sid, None)
    except Exception:
        pass

    return True
