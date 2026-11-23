"""Open ArrayView launched in a remote Rex terminal on the Mac running Rex.

Two halves:

* Remote side (``announce``): ArrayView running in a Rex terminal whose Rex app
  is on another machine reports its viewer URL as an OSC 7501 program-status
  record (``id=arrayview``). That is an ordinary terminal escape; Rex stores it
  and emits a ``program_status_changed`` event on the remote Rex server.

* Mac side (``watch``): a small helper on the user's Mac follows the events of
  every remote Rex server the Rex app is connected to right now (it never
  connects to a host the app is not already using, so sleeping machines stay
  asleep). On an ArrayView report it forwards a free local port to the remote
  viewer with ``rex forward`` and opens a native window (browser fallback).

``arrayview rex setup`` installs the helper as a login agent; ``arrayview rex
remove`` uninstalls it; ``arrayview rex watch`` runs it in the foreground.
"""

from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request

STATUS_ID = "arrayview"
_AGENT_LABEL = "com.arrayview.rex-watch"
_HOST_POLL_SECONDS = 3.0


# ---------------------------------------------------------------------------
# Remote side
# ---------------------------------------------------------------------------


def announce(url: str, view: str = "native") -> bool:
    """Tell the user's Rex app (via the remote Rex server) where the viewer is.

    *view* is ``"browser"`` when the user asked for a browser tab
    (``--window browser`` or the config default); the Mac helper honours it.
    """
    # Rex keeps only the protocol's own keys, so everything rides in msg.
    view = "browser" if view == "browser" else "native"
    payload = json.dumps(
        {"url": url, "view": view, "nonce": time.time_ns()}, separators=(",", ":")
    )
    msg = base64.b64encode(payload.encode()).decode().rstrip("=")
    seq = (
        f"\033]7501;state=idle:app=arrayview:id={STATUS_ID}"
        f":msg={msg}\033\\"
    )
    try:
        with open("/dev/tty", "w") as tty:
            tty.write(seq)
            tty.flush()
        return True
    except OSError:
        pass
    try:
        if sys.stdout.isatty():
            sys.stdout.write(seq)
            sys.stdout.flush()
            return True
    except Exception:
        pass
    return False


# ---------------------------------------------------------------------------
# Mac side
# ---------------------------------------------------------------------------


def _rex_bin() -> str | None:
    found = shutil.which("rex")
    if found:
        return found
    for candidate in (
        "/opt/homebrew/bin/rex",
        "/usr/local/bin/rex",
        os.path.expanduser("~/.local/bin/rex"),
    ):
        if os.path.isfile(candidate):
            return candidate
    return None


def _log(message: str) -> None:
    print(f"[ArrayView rex] {time.strftime('%H:%M:%S')} {message}", flush=True)


def _connected_endpoints() -> set[str]:
    """Remote Rex servers the local Rex app is connected to right now."""
    try:
        out = subprocess.run(
            ["ps", "-axo", "args="], capture_output=True, text=True, timeout=5
        ).stdout
    except Exception:
        return set()
    endpoints = set()
    for line in out.splitlines():
        parts = line.split()
        if "__internal" in parts and "ssh-tunnel" in parts:
            idx = parts.index("ssh-tunnel")
            if idx + 1 < len(parts) and parts[idx + 1].startswith("ssh://"):
                endpoints.add(parts[idx + 1])
    return endpoints


def _report_from_event(event: dict) -> tuple[str, str] | None:
    """(viewer URL, "native" | "browser") from an ArrayView status report."""
    if event.get("type") != "session_event":
        return None
    inner = (event.get("payload") or {}).get("event") or {}
    if inner.get("type") != "block_event":
        return None
    block = inner.get("payload") or {}
    if block.get("name") != "program_status_changed":
        return None
    record = (block.get("payload") or {}).get("record") or {}
    if record.get("id") != STATUS_ID or record.get("app") != "arrayview":
        return None
    text = record.get("msg") or ""
    view = "native"
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            text = str(data.get("url") or "")
            view = "browser" if data.get("view") == "browser" else "native"
    except ValueError:
        pass  # a bare URL
    parsed = urllib.parse.urlsplit(text)
    if parsed.scheme != "http" or parsed.hostname not in {"localhost", "127.0.0.1"}:
        return None
    if not parsed.port:
        return None
    return text, view


def _viewer_url_from_event(event: dict) -> str | None:
    report = _report_from_event(event)
    return report[0] if report else None


class _Opened:
    """One remote ArrayView server shown on this Mac: its forward and windows."""

    def __init__(self, forward: subprocess.Popen, local_port: int):
        self.forward = forward
        self.local_port = local_port
        self.windows: list[subprocess.Popen] = []
        self.browser = False
        self._last_ping = time.monotonic()

    def live_windows(self) -> list[subprocess.Popen]:
        self.windows = [w for w in self.windows if w.poll() is None]
        return self.windows

    def show(self, remote_url: str) -> bool:
        """Ask an open window to show the array in *remote_url* as a new tab."""
        query = urllib.parse.parse_qs(urllib.parse.urlsplit(remote_url).query)
        sid = (query.get("sid") or [""])[0]
        if not sid or not self.live_windows():
            return False
        req = urllib.request.Request(
            f"http://localhost:{self.local_port}/notify/{urllib.parse.quote(sid)}",
            data=json.dumps({"wait": True}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                return bool(json.loads(resp.read() or b"{}").get("notified"))
        except Exception:
            return False

    def finished(self) -> bool:
        """True once nothing on this Mac still needs the forward."""
        if self.forward.poll() is not None:
            return True
        if self.live_windows():
            return False
        if not self.browser:
            return True
        # Browser tabs can't be watched; keep the forward while the remote
        # server still answers.
        if time.monotonic() - self._last_ping < 30:
            return False
        self._last_ping = time.monotonic()
        return not _wait_http(f"http://localhost:{self.local_port}/ping", timeout=5)

    def close(self) -> None:
        for proc in [*self.windows, self.forward]:
            if proc.poll() is None:
                try:
                    proc.terminate()
                except Exception:
                    pass


class Watcher:
    def __init__(self, rex: str, window: str = "native"):
        self.rex = rex
        self.window = window
        self._lock = threading.Lock()
        self._opened: dict[tuple[str, int], _Opened] = {}
        self._streams: dict[str, threading.Thread] = {}

    # -- forwarding and display ------------------------------------------

    def _start_forward(self, endpoint: str, remote_port: int) -> tuple[subprocess.Popen, int] | None:
        proc = subprocess.Popen(
            [
                self.rex, "-S", endpoint, "forward", "open", "--no-open",
                "--port", "0", str(remote_port),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        line = proc.stdout.readline() if proc.stdout else ""
        local = urllib.parse.urlsplit(line.strip())
        if not local.port:
            proc.terminate()
            return None
        # Keep draining stdout so the forward never blocks on a full pipe.
        threading.Thread(target=lambda: proc.stdout.read(), daemon=True).start()
        return proc, local.port

    def _open_display(self, opened: _Opened, url: str, view: str = "native") -> None:
        if self.window == "native" and view == "native":
            try:
                from arrayview._launcher import _open_webview_cli_tracked, _shell_url

                # The tabbed window, as a local launch opens it, so later
                # launches can add their arrays to it as new tabs.
                port = urllib.parse.urlsplit(url).port
                sid = (urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)
                       .get("sid") or [""])[0]
                shell = _shell_url(
                    port, sid, _session_name(port, sid), viewer_url=url
                ) if sid else url
                ok, proc = _open_webview_cli_tracked(
                    shell, 1400, 900, shell_port=port if sid else None
                )
                if ok and proc is not None:
                    opened.windows.append(proc)
                    return
            except Exception as exc:
                _log(f"native window failed ({exc}); using the browser")
        subprocess.run(["open", url], check=False)
        opened.browser = True

    def handle(self, endpoint: str, remote_url: str, view: str = "native") -> None:
        parsed = urllib.parse.urlsplit(remote_url)
        key = (endpoint, parsed.port)
        with self._lock:
            current = self._opened.get(key)
            if current is not None and not current.finished():
                if view == "native" and current.show(remote_url):
                    _log(f"showed new array from {endpoint} in the open window")
                    return
                # No window can take it (e.g. only a browser tab is open):
                # open another one through the same forward.
                self._open_display(
                    current, _local_url(remote_url, current.local_port), view
                )
                _log(f"opened another view for {endpoint} port {parsed.port}")
                return
            if current is not None:
                current.close()
            started = self._start_forward(endpoint, parsed.port)
            if started is None:
                _log(f"could not forward port {parsed.port} from {endpoint}")
                return
            forward, local_port = started
            opened = _Opened(forward, local_port)
            local_url = _local_url(remote_url, local_port)
            _wait_http(local_url, timeout=10.0)
            self._open_display(opened, local_url, view)
            self._opened[key] = opened
            _log(f"opened {endpoint} port {parsed.port} at {local_url}")

    def reap(self) -> None:
        """Close forwards once their windows closed or their server ended."""
        with self._lock:
            for key, opened in list(self._opened.items()):
                if opened.finished():
                    opened.close()
                    del self._opened[key]
                    _log(f"closed forward for {key[0]} port {key[1]}")

    # -- event streams ------------------------------------------------------

    def _stream(self, endpoint: str) -> None:
        _log(f"watching {endpoint}")
        while endpoint in _connected_endpoints():
            try:
                proc = subprocess.Popen(
                    [self.rex, "-S", endpoint, "events", "--all", "--json"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True,
                )
            except OSError:
                break
            assert proc.stdout is not None
            for line in proc.stdout:
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                report = _report_from_event(event)
                if report:
                    threading.Thread(
                        target=self.handle, args=(endpoint, *report), daemon=True
                    ).start()
            proc.wait()
            time.sleep(5)
        _log(f"stopped watching {endpoint}")
        with self._lock:
            self._streams.pop(endpoint, None)

    def close_all(self) -> None:
        with self._lock:
            for opened in self._opened.values():
                opened.close()
            self._opened.clear()

    def run(self) -> None:
        import atexit
        import signal

        # Windows are started in their own process group, so they would
        # outlive the helper (e.g. on a restart) unless closed here.
        atexit.register(self.close_all)
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
        _log("waiting for ArrayView launches in remote Rex terminals")
        while True:
            for endpoint in _connected_endpoints():
                with self._lock:
                    if endpoint in self._streams:
                        continue
                    thread = threading.Thread(
                        target=self._stream, args=(endpoint,), daemon=True
                    )
                    self._streams[endpoint] = thread
                thread.start()
            self.reap()
            time.sleep(_HOST_POLL_SECONDS)


def _session_name(port: int, sid: str) -> str:
    try:
        with urllib.request.urlopen(f"http://localhost:{port}/sessions", timeout=3) as r:
            for row in json.loads(r.read()):
                if row.get("sid") == sid and row.get("name"):
                    return str(row["name"])
    except Exception:
        pass
    return "Array"


def _local_url(remote_url: str, local_port: int) -> str:
    parsed = urllib.parse.urlsplit(remote_url)
    return urllib.parse.urlunsplit(
        ("http", f"localhost:{local_port}", parsed.path or "/",
         parsed.query, parsed.fragment)
    )


def _wait_http(url: str, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status < 500:
                    return True
        except Exception:
            time.sleep(0.3)
    return False


# ---------------------------------------------------------------------------
# Login agent
# ---------------------------------------------------------------------------


def _agent_path() -> str:
    return os.path.expanduser(f"~/Library/LaunchAgents/{_AGENT_LABEL}.plist")


def _log_path() -> str:
    return os.path.expanduser("~/Library/Logs/arrayview-rex-watch.log")


def _watch_command() -> list[str]:
    """A command that keeps working after this process exits.

    ``uvx`` runs from a cache environment that can be cleaned up, so a uvx
    install re-enters through ``uvx`` instead of this interpreter.
    """
    uvx = shutil.which("uvx")
    if "archive-v" in sys.prefix and uvx:
        return [uvx, "arrayview", "rex", "watch"]
    return [sys.executable, "-m", "arrayview", "rex", "watch"]


def _plist(command: list[str], rex: str) -> str:
    from xml.sax.saxutils import escape

    args = "".join(f"\n        <string>{escape(a)}</string>" for a in command)
    path = os.pathsep.join(
        dict.fromkeys(
            [os.path.dirname(rex), os.path.dirname(command[0]),
             "/opt/homebrew/bin", "/usr/local/bin", "/usr/bin", "/bin"]
        )
    )
    log = escape(_log_path())
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{_AGENT_LABEL}</string>
    <key>ProgramArguments</key>
    <array>{args}
    </array>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>{escape(path)}</string>
    </dict>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>ProcessType</key>
    <string>Interactive</string>
    <key>StandardOutPath</key>
    <string>{log}</string>
    <key>StandardErrorPath</key>
    <string>{log}</string>
</dict>
</plist>
"""


def _launchctl(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["launchctl", *args], capture_output=True, text=True)


def setup() -> int:
    if sys.platform != "darwin":
        print("Run this on the Mac where the Rex app is, not on the remote machine.")
        return 1
    rex = _rex_bin()
    if rex is None:
        print("Rex's command-line tool wasn't found. In Rex, run 'Install Rex for Agents'.")
        return 1
    path = _agent_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    domain = f"gui/{os.getuid()}"
    _launchctl("bootout", f"{domain}/{_AGENT_LABEL}")
    with open(path, "w") as fh:
        fh.write(_plist(_watch_command(), rex))
    result = _launchctl("bootstrap", domain, path)
    if result.returncode != 0:
        print(f"Couldn't start the helper: {result.stderr.strip()}")
        return 1
    print(
        "Done. ArrayView started in a remote Rex terminal now opens here by itself.\n"
        "Undo with: arrayview rex remove"
    )
    return 0


def remove() -> int:
    path = _agent_path()
    _launchctl("bootout", f"gui/{os.getuid()}/{_AGENT_LABEL}")
    if os.path.exists(path):
        os.unlink(path)
    print("Removed. Remote Rex launches will print a link to Cmd-click again.")
    return 0


def is_installed() -> bool:
    return os.path.exists(_agent_path())


def main(argv: list[str]) -> int:
    command = argv[0] if argv else ""
    if command == "setup":
        return setup()
    if command == "remove":
        return remove()
    if command == "watch":
        rex = _rex_bin()
        if rex is None:
            print("Rex's command-line tool wasn't found.")
            return 1
        window = "browser" if "--browser" in argv else "native"
        Watcher(rex, window=window).run()
        return 0
    print(
        "usage: arrayview rex setup | remove | watch [--browser]\n"
        "  setup   open remote Rex launches on this Mac automatically\n"
        "  remove  undo setup"
    )
    return 0 if command in {"", "-h", "--help"} else 1
