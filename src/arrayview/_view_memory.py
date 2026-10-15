"""Remember the last view of each file, for Preferences → Reopen where you left off.

Stored in ~/.arrayview/view_memory.json, keyed by the file's real path (plus the
array name inside .npz/.mat files). A remembered view is only handed back for
an array of the same shape, so a file rewritten with new dimensions starts fresh.
A file rewritten with the same shape keeps its slice, view and colormap, but not
its display range, which no longer fits the new values.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time

VIEW_MEMORY_PATH = os.path.expanduser("~/.arrayview/view_memory.json")
_MAX_ENTRIES = 200
_RANGE_KEYS = ("manualVmin", "manualVmax")
_LOCK = threading.Lock()


def view_memory_key(session) -> str | None:
    filepath = getattr(session, "filepath", None)
    if not filepath:
        return None
    try:
        key = os.path.realpath(str(filepath))
    except (OSError, ValueError):
        return None
    array_key = getattr(session, "array_key", None)
    return f"{key}#{array_key}" if array_key else key


def _file_signature(session) -> list | None:
    from arrayview._session import file_signature

    sig = file_signature(str(session.filepath))
    return list(sig) if sig else None


def _load() -> dict:
    try:
        with open(VIEW_MEMORY_PATH, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _save(entries: dict) -> None:
    folder = os.path.dirname(VIEW_MEMORY_PATH)
    os.makedirs(folder, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix=".view_memory.", suffix=".tmp", dir=folder)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(entries, f)
        os.replace(temp_path, VIEW_MEMORY_PATH)
    except Exception:
        try:
            os.unlink(temp_path)
        except FileNotFoundError:
            pass
        raise


def remembered_view(session) -> dict | None:
    key = view_memory_key(session)
    if key is None:
        return None
    with _LOCK:
        entry = _load().get(key)
    if not isinstance(entry, dict) or not isinstance(entry.get("view"), dict):
        return None
    if entry.get("shape") != [int(s) for s in session.shape]:
        return None
    view = dict(entry["view"])
    if entry.get("signature") is None or entry.get("signature") != _file_signature(session):
        for k in _RANGE_KEYS:
            view.pop(k, None)
    return view


def remember_view(session, view: dict) -> bool:
    key = view_memory_key(session)
    if key is None or not isinstance(view, dict):
        return False
    with _LOCK:
        entries = _load()
        entries[key] = {
            "shape": [int(s) for s in session.shape],
            "view": view,
            "signature": _file_signature(session),
            "updated": time.time(),
        }
        if len(entries) > _MAX_ENTRIES:
            oldest = sorted(entries, key=lambda k: entries[k].get("updated", 0) if isinstance(entries[k], dict) else 0)
            for stale in oldest[: len(entries) - _MAX_ENTRIES]:
                entries.pop(stale, None)
        _save(entries)
    return True
