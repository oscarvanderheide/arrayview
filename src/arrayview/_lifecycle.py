"""Session lifecycle helpers shared by REST and WebSocket close paths."""

from __future__ import annotations

import shutil
import threading

from arrayview._session import (
    CANCELLED_PENDING_SESSIONS,
    NATIVE_READY_REQUESTS,
    PENDING_SESSION_EVENTS,
    PENDING_SESSIONS,
    SESSIONS,
    VIEWER_CONNECTION_EPOCHS,
    VIEWER_RELEASE_TASKS,
)

_SESSION_LEASE_LOCK = threading.Lock()


def commit_pending_session(sid: str, session) -> bool:
    """Commit a background load unless its owning request was released."""
    with _SESSION_LEASE_LOCK:
        if sid in CANCELLED_PENDING_SESSIONS:
            return False
        SESSIONS[sid] = session
        return True


def commit_session_group_unless_cancelled(
    request_sid: str,
    sessions: list,
) -> bool:
    """Atomically commit all sessions produced by one registration request."""
    with _SESSION_LEASE_LOCK:
        if request_sid in CANCELLED_PENDING_SESSIONS:
            return False
        for session in sessions:
            SESSIONS[session.sid] = session
        return True


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


def release_session(sid: str, *, cancel_if_missing: bool = False) -> bool:
    """Release one viewer lease and drop the session after the final lease."""
    with _SESSION_LEASE_LOCK:
        was_pending = sid in PENDING_SESSIONS
        if was_pending or (cancel_if_missing and sid not in SESSIONS):
            CANCELLED_PENDING_SESSIONS.add(sid)
        PENDING_SESSIONS.discard(sid)
        event = PENDING_SESSION_EVENTS.pop(sid, None)
        session = SESSIONS.get(sid)
        if session is None:
            released = was_pending
        else:
            leases = max(1, int(getattr(session, "viewer_leases", 1)))
            if leases > 1:
                session.viewer_leases = leases - 1
                released = True
                session = None
            else:
                related_sids = [
                    *getattr(session, "related_release_sids", []),
                    *getattr(session, "collection_overlay_sids", []),
                ]
                SESSIONS.pop(sid, None)
                NATIVE_READY_REQUESTS.difference_update(
                    {item for item in NATIVE_READY_REQUESTS if item[0] == sid}
                )
                released = True

    release_task = VIEWER_RELEASE_TASKS.pop(sid, None)
    VIEWER_CONNECTION_EPOCHS.pop(sid, None)
    if release_task is not None and not release_task.done():
        try:
            release_task.get_loop().call_soon_threadsafe(release_task.cancel)
        except (RuntimeError, AttributeError):
            pass

    if event is not None:
        try:
            event.set()
        except Exception:
            pass

    if session is None:
        return released

    for related_sid in dict.fromkeys(str(value) for value in related_sids):
        if related_sid != sid:
            release_session(related_sid)

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
