"""Session-owned source-file polling for explicit local ``--watch`` launches."""

from __future__ import annotations

import json
import os
import threading
import urllib.request


def _mtimes(filepaths: list[str]) -> tuple[float, ...]:
    return tuple(os.stat(path).st_mtime for path in filepaths)


def start_file_watch(
    filepath: str | list[str], sid: str, port: int
) -> tuple[threading.Event, threading.Thread]:
    """Reload ``sid`` when its file changes; a list watches every file of a stack."""
    filepaths = [filepath] if isinstance(filepath, str) else list(filepath)
    stop_event = threading.Event()

    def _watch() -> None:
        try:
            last_mtime = _mtimes(filepaths)
        except OSError:
            return
        while not stop_event.wait(1.0):
            try:
                mtime = _mtimes(filepaths)
            except OSError:
                continue
            if mtime == last_mtime:
                continue
            last_mtime = mtime
            try:
                request = urllib.request.Request(
                    f"http://localhost:{port}/reload/{sid}",
                    data=b"",
                    method="POST",
                )
                with urllib.request.urlopen(request, timeout=10) as response:
                    result = json.loads(response.read())
                version = result.get("version", "?")
                print(
                    f"[ArrayView] File changed — reloaded (version {version})",
                    flush=True,
                )
            except Exception:
                # The session/server may be closing. The release path owns the
                # stop event; a transient HTTP failure is not a reason to leak.
                continue

    thread = threading.Thread(
        target=_watch,
        daemon=True,
        name=f"arrayview-watch-{sid[:8]}",
    )
    thread.start()
    return stop_event, thread


def attach_file_watch(session, filepath: str, port: int) -> None:
    # Several files opened as one stack: watch each of them.
    files = getattr(session.data, "_file_matrix", None)
    if files is not None:
        filepath = [path for row in files for path in row if path]
    stop_event, thread = start_file_watch(filepath, session.sid, port)
    session._source_watch_stop = stop_event
    session._source_watch_thread = thread
