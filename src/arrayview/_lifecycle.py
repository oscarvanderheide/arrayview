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
    VIEWER_PHASE_JOURNALS,
    VIEWER_LAUNCH_ROUTES,
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
        group_sids = {request_sid, *(session.sid for session in sessions)}
        if group_sids.intersection(CANCELLED_PENDING_SESSIONS):
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
    retire_journals = False
    with _SESSION_LEASE_LOCK:
        was_pending = sid in PENDING_SESSIONS
        if was_pending or (cancel_if_missing and sid not in SESSIONS):
            CANCELLED_PENDING_SESSIONS.add(sid)
        PENDING_SESSIONS.discard(sid)
        event = PENDING_SESSION_EVENTS.pop(sid, None)
        session = SESSIONS.get(sid)
        if session is None:
            released = was_pending
            retire_journals = was_pending or cancel_if_missing
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
                retire_journals = True
                released = True

        if retire_journals:
            journals = VIEWER_PHASE_JOURNALS.pop(sid, {})
            for journal in journals.values():
                navigation_key = journal.get("navigation_key")
                if navigation_key:
                    VIEWER_LAUNCH_ROUTES.pop(navigation_key, None)
                journal_task = journal.get("release_task")
                if journal_task is not None and not journal_task.done():
                    try:
                        journal_task.get_loop().call_soon_threadsafe(
                            journal_task.cancel
                        )
                    except (RuntimeError, AttributeError):
                        pass

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
    watch_stop = getattr(session, "_source_watch_stop", None)
    watch_thread = getattr(session, "_source_watch_thread", None)
    if watch_stop is not None:
        watch_stop.set()
    if watch_thread is not None and watch_thread is not threading.current_thread():
        watch_thread.join(timeout=2.0)
    drop_staging_dir = getattr(session, "_drop_staging_dir", None)
    if drop_staging_dir:
        shutil.rmtree(drop_staging_dir, ignore_errors=True)
    if getattr(session, "_source_staging_dirs", []):
        from arrayview._source_safety import cleanup_staging_directory

        for staging_dir in dict.fromkeys(session._source_staging_dirs):
            cleanup_staging_directory(staging_dir)

    try:
        from arrayview._routes_persistence import _CROP_LOCK, _CROP_STATE

        with _CROP_LOCK:
            _CROP_STATE.pop(sid, None)
    except Exception:
        pass

    return True
