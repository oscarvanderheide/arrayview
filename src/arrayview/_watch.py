"""Session-owned source-file polling for explicit local ``--watch`` launches."""

from __future__ import annotations

import json
import os
import threading
import urllib.request


def start_file_watch(filepath: str, sid: str, port: int) -> tuple[threading.Event, threading.Thread]:
    stop_event = threading.Event()

    def _watch() -> None:
        try:
            last_mtime = os.stat(filepath).st_mtime
        except OSError:
            return
        while not stop_event.wait(1.0):
            try:
                mtime = os.stat(filepath).st_mtime
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
    stop_event, thread = start_file_watch(filepath, session.sid, port)
    session._source_watch_stop = stop_event
    session._source_watch_thread = thread
