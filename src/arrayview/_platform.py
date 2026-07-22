"""Environment detection: Jupyter, VS Code, SSH, tunnel, Julia."""

from __future__ import annotations

import importlib.util
import hashlib
import json
import os
import subprocess
import sys


def get_ppid(pid: int) -> int:
    """Return parent PID of *pid*, or -1 if unavailable.

    Cross-platform: /proc on Linux, ps on macOS, wmic on Windows.
    """
    if sys.platform == "win32":
        try:
            r = subprocess.run(
                ["wmic", "process", "where", f"processid={pid}",
                 "get", "parentprocessid"],
                capture_output=True, text=True, timeout=5,
            )
            for line in r.stdout.strip().splitlines():
                val = line.strip()
                if val.isdigit():
                    return int(val)
        except Exception:
            pass
        return -1
    try:
        with open(f"/proc/{pid}/status") as fh:
            for line in fh:
                if line.startswith("PPid:"):
                    return int(line.split()[1])
    except Exception:
        pass
    try:
        r = subprocess.run(
            ["ps", "-p", str(pid), "-o", "ppid="],
            capture_output=True, text=True, timeout=2,
        )
        return int(r.stdout.strip())
    except Exception:
        pass
    return -1


def _find_ancestor_environment_value(name: str, depth: int = 12) -> str | None:
    """Recover one environment value from this process or its ancestors."""
    value = os.environ.get(name)
    if value:
        return value

    pid = os.getpid()
    encoded_prefix = f"{name}=".encode()
    text_prefix = f"{name}="
    for _ in range(depth):
        pid = get_ppid(pid)
        if pid <= 1:
            break
        try:
            with open(f"/proc/{pid}/environ", "rb") as handle:
                for entry in handle.read().split(b"\0"):
                    if entry.startswith(encoded_prefix):
                        value = entry[len(encoded_prefix) :].decode()
                        if value:
                            return value
        except Exception:
            pass
        if sys.platform == "darwin":
            try:
                result = subprocess.run(
                    ["ps", "ewwww", "-p", str(pid)],
                    capture_output=True,
                    text=True,
                    timeout=3,
                )
                for token in result.stdout.split():
                    if token.startswith(text_prefix):
                        value = token[len(text_prefix) :]
                        if value:
                            return value
            except Exception:
                pass
    return None


# ---------------------------------------------------------------------------
# Jupyter
# ---------------------------------------------------------------------------

_jupyter_server_port: int | None = None

_JUPYTER_CACHE: bool | None = None  # None = not yet computed
_MATLAB_CACHE: bool | None = None


def _in_jupyter() -> bool:
    global _JUPYTER_CACHE
    if _JUPYTER_CACHE is not None:
        return _JUPYTER_CACHE
    try:
        from IPython import get_ipython

        shell = get_ipython()
        result = shell is not None and "ipykernel" in type(shell).__module__
    except ImportError:
        result = False
    _JUPYTER_CACHE = result
    return result


def _in_matlab() -> bool:
    """True when Python is running inside or directly under MATLAB.

    MATLAB desktop launches can inherit VS Code terminal environment variables
    when MATLAB itself was started from an integrated terminal. For local
    desktop MATLAB sessions we still want ArrayView to prefer the native
    PyWebView window, so we treat MATLAB as its own local desktop host.
    Remote/tunnel routing is still governed separately by _is_vscode_remote()
    and SSH detection.
    """
    global _MATLAB_CACHE
    if _MATLAB_CACHE is not None:
        return _MATLAB_CACHE

    if any(key.startswith("MATLAB") for key in os.environ):
        _MATLAB_CACHE = True
        return True

    exe = sys.executable.lower()
    if "matlab" in exe:
        _MATLAB_CACHE = True
        return True

    if any(modname.lower().startswith("matlab") for modname in sys.modules):
        _MATLAB_CACHE = True
        return True

    def _command_from_pid(pid: int) -> str:
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as fh:
                raw = fh.read().replace(b"\0", b" ").decode(errors="ignore")
            if raw:
                return raw.lower()
        except Exception:
            pass
        try:
            r = subprocess.run(
                ["ps", "-p", str(pid), "-o", "command="],
                capture_output=True,
                text=True,
                timeout=2,
            )
            return r.stdout.strip().lower()
        except Exception:
            return ""

    pid = os.getpid()
    for _ in range(8):
        cmd = _command_from_pid(pid)
        if "matlab" in cmd:
            _MATLAB_CACHE = True
            return True
        pid = get_ppid(pid)
        if pid <= 1:
            break

    _MATLAB_CACHE = False
    return False


# ---------------------------------------------------------------------------
# VS Code IPC hook recovery  (needed by detection functions below)
# ---------------------------------------------------------------------------

_VSCODE_IPC_HOOK_CACHE: str | None = "__unset__"  # sentinel; None means not found


def _find_vscode_ipc_hook() -> str | None:
    """Return the value of VSCODE_IPC_HOOK_CLI, searching ancestor processes.

    uv run (and similar launchers) strip environment variables before executing
    Python.  Walking up the process tree lets us recover VSCODE_IPC_HOOK_CLI
    from the shell that originally invoked the command.  Result is cached after
    the first call so repeated calls don't re-walk the process tree.
    """
    global _VSCODE_IPC_HOOK_CACHE
    if _VSCODE_IPC_HOOK_CACHE != "__unset__":
        return _VSCODE_IPC_HOOK_CACHE

    def _ipc_from_pid(pid: int) -> str:
        # Linux: /proc/<pid>/environ (null-separated KEY=VALUE pairs)
        try:
            with open(f"/proc/{pid}/environ", "rb") as fh:
                for entry in fh.read().split(b"\0"):
                    if entry.startswith(b"VSCODE_IPC_HOOK_CLI="):
                        return entry[len(b"VSCODE_IPC_HOOK_CLI=") :].decode()
        except Exception:
            pass
        # macOS: `ps ewwww` appends the environment after the argument list.
        # Each env var is a whitespace-separated TOKEN of the form KEY=VALUE.
        try:
            r = subprocess.run(
                ["ps", "ewwww", "-p", str(pid)],
                capture_output=True,
                text=True,
                timeout=3,
            )
            for token in r.stdout.split():
                if token.startswith("VSCODE_IPC_HOOK_CLI="):
                    return token[len("VSCODE_IPC_HOOK_CLI=") :]
        except Exception:
            pass
        return ""

    # Own environment first
    val = os.environ.get("VSCODE_IPC_HOOK_CLI", "")
    if val and os.path.exists(val):
        _VSCODE_IPC_HOOK_CACHE = val
        return val

    # Walk up to 12 ancestor processes
    pid = os.getpid()
    for _ in range(12):
        pid = get_ppid(pid)
        if pid <= 1:
            break
        val = _ipc_from_pid(pid)
        if val and os.path.exists(val):
            _VSCODE_IPC_HOOK_CACHE = val
            return val

    # tmux detaches the process tree from the VS Code terminal that launched it,
    # so ancestor-walk fails.  Two fallback strategies when TERM_PROGRAM=tmux:
    #
    # Strategy 1: tmux show-environment.  Works only if VSCODE_IPC_HOOK_CLI was
    # explicitly listed in tmux's `update-environment` option (non-default).
    if os.environ.get("TERM_PROGRAM") == "tmux":
        try:
            r = subprocess.run(
                ["tmux", "show-environment", "VSCODE_IPC_HOOK_CLI"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            # Output is "VSCODE_IPC_HOOK_CLI=/path/to/socket" or "-VSCODE_IPC_HOOK_CLI"
            line = r.stdout.strip()
            if line.startswith("VSCODE_IPC_HOOK_CLI="):
                val = line[len("VSCODE_IPC_HOOK_CLI=") :]
                if val and os.path.exists(val):
                    _VSCODE_IPC_HOOK_CACHE = val
                    return val
        except Exception:
            pass

        # Strategy 2: enumerate ALL clients attached to the current tmux session
        # and read VSCODE_IPC_HOOK_CLI from each one's environment.
        #
        # Why: tmux pane processes are children of tmux-server, so the ancestor-
        # walk never reaches the VS Code terminal shell.  But each tmux CLIENT
        # process (the `tmux` command that attached to the session from a VS Code
        # terminal) DID inherit VSCODE_IPC_HOOK_CLI directly.  By checking every
        # client for the current session we handle:
        #   - single client (most common)
        #   - multiple clients attached (e.g. shared pairing session)
        #   - session created outside VS Code then attached from VS Code terminal
        try:
            # Get current session ID so we only probe clients for THIS session.
            r_sid = subprocess.run(
                ["tmux", "display-message", "-p", "#{session_id}"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            session_id = r_sid.stdout.strip()
            if session_id:
                r_clients = subprocess.run(
                    ["tmux", "list-clients", "-t", session_id, "-F", "#{client_pid}"],
                    capture_output=True,
                    text=True,
                    timeout=2,
                )
                for line in r_clients.stdout.strip().splitlines():
                    try:
                        client_pid = int(line.strip())
                    except ValueError:
                        continue
                    if client_pid > 1:
                        val = _ipc_from_pid(client_pid)
                        if val and os.path.exists(val):
                            _VSCODE_IPC_HOOK_CACHE = val
                            return val
        except Exception:
            pass

    _VSCODE_IPC_HOOK_CACHE = None
    return None


# ---------------------------------------------------------------------------
# VS Code CLI detection
# ---------------------------------------------------------------------------


def _current_vscode_remote_cli() -> str | None:
    """Return the remote CLI belonging to this exact VS Code server.

    ``VSCODE_NLS_CONFIG.defaultMessagesFile`` lives below the active server's
    ``server/out`` directory.  Deriving the CLI from that path avoids guessing
    between co-located Tunnel and Remote-SSH installations under the same home
    directory.
    """
    # uv may omit this variable from the launched Python process even though
    # the integrated terminal shell still has it. Recovering it from the
    # ancestor chain keeps placement bound to this exact VS Code server.
    raw_config = _find_ancestor_environment_value("VSCODE_NLS_CONFIG")
    if not raw_config:
        return None
    try:
        messages_file = json.loads(raw_config).get("defaultMessagesFile")
    except (AttributeError, TypeError, ValueError):
        return None
    if not isinstance(messages_file, str) or not messages_file:
        return None
    out_dir = os.path.dirname(messages_file)
    server_dir = os.path.dirname(out_dir)
    if os.path.basename(out_dir) != "out" or os.path.basename(server_dir) != "server":
        return None
    candidate = os.path.join(server_dir, "bin", "remote-cli", "code")
    if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
        return candidate
    return None


def _find_code_cli(*, is_remote: bool | None = None) -> str | None:
    """Return path to the VS Code CLI ('code'), or None if not found.

    In a VS Code tunnel/remote, the tunnel server provides its own ``code``
    helper at ``~/.vscode-server/.../remote-cli/code``.  We prefer that over
    a desktop ``code`` when ``VSCODE_IPC_HOOK_CLI`` is set, because the
    desktop binary would open a *new* VS Code window instead of routing
    through the tunnel.
    """
    import glob
    import shutil

    if is_remote is None:
        is_remote = _is_vscode_remote()

    found = shutil.which("code")

    # In a VS Code remote/tunnel, bind installation to this exact window's
    # server.  A shared home can contain both Tunnel and Remote-SSH helpers;
    # choosing the newest glob can update/reload the wrong extension host.
    if is_remote:
        exact_remote_cli = _current_vscode_remote_cli()
        if exact_remote_cli:
            return exact_remote_cli

        # uv run and similar launchers may strip VSCODE_IPC_HOOK_CLI.  When an
        # exact hook can still be recovered, the normal `code` wrapper is safe:
        # _ensure_vscode_extension injects that hook into the installer env.
        if _find_vscode_ipc_hook() and found:
            return found

        # The tunnel helper is typically at one of:
        #   ~/.vscode-server/bin/<commit>/bin/remote-cli/code
        #   ~/.vscode-server/cli/servers/.../server/bin/remote-cli/code
        #   ~/.vscode/cli/servers/.../server/bin/remote-cli/code  (newer tunnels)
        candidates: list[str] = []
        for pattern in [
            os.path.expanduser("~/.vscode-server/bin/*/bin/remote-cli/code"),
            os.path.expanduser(
                "~/.vscode-server/cli/servers/*/server/bin/remote-cli/code"
            ),
            os.path.expanduser("~/.vscode/cli/servers/*/server/bin/remote-cli/code"),
        ]:
            for candidate in glob.glob(pattern):
                if os.access(candidate, os.X_OK) and candidate not in candidates:
                    candidates.append(candidate)
        if len(candidates) == 1:
            return candidates[0]
        # Multiple uncorrelated helpers are unsafe: fail closed instead of
        # installing into whichever Tunnel/Remote-SSH profile is newest.
        return None

    if found and (is_remote or "remote-cli" not in found):
        return found
    candidates: list[str] = []
    if sys.platform == "darwin":
        candidates = [
            "/opt/homebrew/bin/code",
            "/usr/local/bin/code",
            os.path.expanduser(
                "~/Applications/Visual Studio Code.app/Contents/Resources/app/bin/code"
            ),
            "/Applications/Visual Studio Code.app/Contents/Resources/app/bin/code",
        ]
    elif sys.platform.startswith("linux"):
        candidates = [
            "/usr/bin/code",
            "/usr/local/bin/code",
            "/snap/bin/code",
            os.path.expanduser("~/.local/bin/code"),
        ]
    for path in candidates:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    return None


# ---------------------------------------------------------------------------
# VS Code terminal / remote / tunnel detection
# ---------------------------------------------------------------------------


def _in_vscode_terminal() -> bool:
    """True when running inside any VS Code integrated terminal (local or remote)."""
    if _in_matlab() and not _is_vscode_remote():
        return False
    if os.environ.get("TERM_PROGRAM") == "vscode":
        return True
    if os.environ.get("VSCODE_IPC_HOOK_CLI"):
        return True
    # uv run and similar launchers strip env vars; walk ancestor processes.
    return _find_vscode_ipc_hook() is not None


def _has_vscode_window_registration() -> bool:
    signal_dir = os.path.expanduser("~/.arrayview")
    if not os.path.isdir(signal_dir):
        return False
    try:
        return any(
            filename.startswith("window-") and filename.endswith(".json")
            for filename in os.listdir(signal_dir)
        )
    except Exception:
        return False


def _process_is_alive(pid: object) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, TypeError, ValueError):
        return False


def _exact_vscode_window_registration(
    ipc: str | None = None,
) -> tuple[str, dict] | None:
    """Return the live registration belonging to this exact VS Code server.

    A revived integrated terminal can retain ``VSCODE_NLS_CONFIG`` while losing
    both its IPC hook and the window id injected by the extension.  Tunnel and
    Remote-SSH windows may also share ``~/.arrayview``, so selecting an arbitrary
    remote registration is unsafe.  When direct window evidence is unavailable,
    match the live extension-host executable to the exact server root named by
    the terminal's NLS configuration, and only accept a unique match.
    """
    signal_dir = os.path.expanduser("~/.arrayview")

    def _read(window_id: str) -> tuple[str, dict] | None:
        try:
            with open(
                os.path.join(signal_dir, f"window-{window_id}.json"),
                encoding="utf-8",
            ) as handle:
                registration = json.load(handle)
        except (OSError, ValueError, TypeError):
            return None
        if not isinstance(registration, dict) or not _process_is_alive(
            registration.get("pid")
        ):
            return None
        return window_id, registration

    if ipc:
        window_id = hashlib.sha256(ipc.encode()).hexdigest()[:16]
        if registration := _read(window_id):
            return registration
    elif window_id := os.environ.get("ARRAYVIEW_WINDOW_ID"):
        if registration := _read(window_id):
            return registration

    exact_cli = _current_vscode_remote_cli()
    if exact_cli is None:
        return None
    server_dir = os.path.realpath(
        os.path.dirname(os.path.dirname(os.path.dirname(exact_cli)))
    )
    try:
        filenames = os.listdir(signal_dir)
    except OSError:
        return None

    matches: list[tuple[str, dict]] = []
    for filename in filenames:
        if not filename.startswith("window-") or not filename.endswith(".json"):
            continue
        window_id = filename[len("window-") : -len(".json")]
        candidate = _read(window_id)
        if candidate is None:
            continue
        registration = candidate[1]
        if registration.get("remoteName") is None:
            continue
        try:
            executable = os.path.realpath(
                os.readlink(f"/proc/{int(registration['pid'])}/exe")
            )
        except (KeyError, OSError, TypeError, ValueError):
            continue
        if os.path.dirname(executable) == server_dir:
            matches.append(candidate)

    return matches[0] if len(matches) == 1 else None


def _exact_vscode_registration_remote(ipc: str) -> bool | None:
    """Return the owning extension host's remote flag when registered.

    The registration is the display host's own statement of placement. Generic
    terminal variables and installed ``remote-cli`` helpers are not authority:
    both can exist in a local desktop window.
    """
    exact = _exact_vscode_window_registration(ipc)
    if exact is not None and "remoteName" in exact[1]:
        return exact[1].get("remoteName") is not None
    return None


def _has_live_vscode_remote_registration() -> bool:
    signal_dir = os.path.expanduser("~/.arrayview")
    try:
        filenames = os.listdir(signal_dir)
    except OSError:
        return False
    for filename in filenames:
        if not filename.startswith("window-") or not filename.endswith(".json"):
            continue
        try:
            with open(os.path.join(signal_dir, filename), encoding="utf-8") as handle:
                registration = json.load(handle)
        except (OSError, ValueError, TypeError):
            continue
        if registration.get("remoteName") is not None and _process_is_alive(
            registration.get("pid")
        ):
            return True
    return False


def _is_vscode_remote() -> bool:
    """True when running inside a VS Code remote/tunnel session.

    A VS Code IPC hook proves that the process is in an integrated terminal;
    it does not prove that the extension host is remote.  In particular, local
    Linux terminals have the same hook.  Require an SSH or VS Code server marker
    before selecting the remote signal/forwarding path.
    """
    # Try env var first, then walk process tree (handles uv run env stripping).
    ipc = os.environ.get("VSCODE_IPC_HOOK_CLI") or _find_vscode_ipc_hook()
    registered_remote = _exact_vscode_registration_remote(ipc)
    if registered_remote is not None:
        return registered_remote
    # VSCODE_NLS_CONFIG points inside the exact active VS Code server. Unlike
    # the presence of unrelated remote-cli installations on disk, a validated
    # server/out -> bin/remote-cli/code path is authoritative remote evidence.
    if _current_vscode_remote_cli() is not None:
        return True
    ssh = bool(os.environ.get("SSH_CONNECTION") or os.environ.get("SSH_CLIENT"))
    vscode_server = bool(os.environ.get("VSCODE_AGENT_FOLDER"))
    if ipc and (ssh or vscode_server):
        return True
    # TERM_PROGRAM=vscode + SSH_CONNECTION = VS Code SSH remote (belt-and-suspenders)
    if os.environ.get("TERM_PROGRAM") == "vscode" and (
        os.environ.get("SSH_CONNECTION") or os.environ.get("SSH_CLIENT")
    ):
        return True
    # Notebook kernels do not inherit the integrated terminal's IPC env vars,
    # so remote VS Code notebooks need a second detection path. The remote
    # opener extension keeps a live remote window registration while the client
    # window is active.
    if _in_jupyter():
        return _has_live_vscode_remote_registration()
    return False


def _in_vscode_tunnel() -> bool:
    """True for a VS Code tunnel remote, excluding Remote-SSH."""
    if os.environ.get("SSH_CLIENT") or os.environ.get("SSH_CONNECTION"):
        return False
    return _is_vscode_remote()


def _can_native_window() -> bool:
    """True if a pywebview native window can be opened.

    Returns False whenever a VS Code terminal is detectable, meaning we're
    running inside a VS Code terminal (local or remote/tunnel).  In that case
    we always prefer the VS Code tab route over a native window, because on
    a tunnel-server machine the user isn't looking at that screen.
    """
    if _in_vscode_terminal():
        return False
    if _is_vscode_remote():
        return False
    # Plain SSH (no VS Code): the display is on the client machine, not here.
    if os.environ.get("SSH_CLIENT") or os.environ.get("SSH_CONNECTION"):
        return False
    return _native_window_gui() is not None


def _native_window_gui() -> str | None:
    """Return the pywebview GUI backend name to use, or None if unavailable.

    On macOS/Windows and Linux we return ``""`` so pywebview uses its own
    backend auto-detection, which is more robust than forcing a specific
    backend: it probes each installed binding and falls back across them.
    Forcing ``"qt"`` because ``find_spec("qtpy")`` succeeds is a bad probe —
    qtpy being importable does not mean QtWebEngine can actually initialise
    (missing libnss/libxkbcommon/QtWebEngineProcess is common on Linux), and
    a failed forced backend hangs for ~10s before the watchdog falls back to a
    browser. Letting pywebview pick avoids that. We still require a display
    server on Linux and the webview package itself everywhere.
    """
    if importlib.util.find_spec("webview") is None:
        return None
    if sys.platform in ("darwin", "win32"):
        return ""
    # Linux/BSD: need a display server. pywebview probes Qt/GTK/Tk backends
    # itself, so we don't pin one — we only confirm a display is reachable.
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        return None
    return ""


# ---------------------------------------------------------------------------
# Julia environment detection
# ---------------------------------------------------------------------------


def _is_julia_env() -> bool:
    """True when running inside Julia via PythonCall/PyCall.
    In this environment the GIL is not released between Julia statements, so
    daemon threads (uvicorn) cannot serve HTTP requests once view() returns.
    We detect it by checking for juliacall/julia markers in loaded modules or
    the executable path.
    """
    if any("juliacall" in k.lower() or "julia" in k.lower() for k in sys.modules):
        return True
    exe = sys.executable.lower()
    return "julia" in exe


_julia_jupyter_cache: bool | None = None


def _in_julia_jupyter() -> bool:
    """True when running in Julia via PythonCall inside an IJulia Jupyter kernel.

    In IJulia, Julia's stdout is redirected to an IJulia stream object; its type
    name contains "IJulia". In a plain terminal, stdout is a Base.TTY.
    Fallback: try accessing Main.IJulia (present when `using IJulia` was called).
    """
    global _julia_jupyter_cache
    if _julia_jupyter_cache is not None:
        return _julia_jupyter_cache
    try:
        import juliacall as _jl

        r = _jl.Main.seval('occursin("IJulia", string(typeof(stdout)))')
        if str(r).strip().lower() == "true":
            _julia_jupyter_cache = True
            return True
        r2 = _jl.Main.seval("try; Main.IJulia; true; catch; false; end")
        _julia_jupyter_cache = str(r2).strip().lower() == "true"
    except Exception:
        _julia_jupyter_cache = False
    return _julia_jupyter_cache


# ---------------------------------------------------------------------------
# Unified environment detection (used by config system)
# ---------------------------------------------------------------------------


def detect_environment() -> str:
    """Return the current environment name for config lookup.

    Returns one of: 'jupyter', 'vscode', 'julia', 'ssh', 'terminal'.
    Checked in priority order — first match wins.
    """
    if _in_jupyter():
        return "jupyter"
    if _in_vscode_terminal():
        return "vscode"
    if _is_julia_env():
        return "julia"
    if os.environ.get("SSH_CONNECTION") or os.environ.get("SSH_CLIENT"):
        return "ssh"
    return "terminal"
