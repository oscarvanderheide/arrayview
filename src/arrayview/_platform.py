"""Environment detection: Jupyter, VS Code, SSH, tunnel, Julia."""

from __future__ import annotations

import os
import subprocess
import sys

# ---------------------------------------------------------------------------
# Jupyter
# ---------------------------------------------------------------------------

_jupyter_server_port: int | None = None

_JUPYTER_CACHE: bool | None = None  # None = not yet computed


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

    def _ppid(pid: int) -> int:
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
                capture_output=True,
                text=True,
                timeout=2,
            )
            return int(r.stdout.strip())
        except Exception:
            pass
        return -1

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
        pid = _ppid(pid)
        if pid <= 1:
            break
        val = _ipc_from_pid(pid)
        if val and os.path.exists(val):
            _VSCODE_IPC_HOOK_CACHE = val
            return val

    _VSCODE_IPC_HOOK_CACHE = None
    return None


# ---------------------------------------------------------------------------
# VS Code CLI detection
# ---------------------------------------------------------------------------


def _find_code_cli() -> str | None:
    """Return path to the VS Code CLI ('code'), or None if not found.

    In a VS Code tunnel/remote, the tunnel server provides its own ``code``
    helper at ``~/.vscode-server/.../remote-cli/code``.  We prefer that over
    a desktop ``code`` when ``VSCODE_IPC_HOOK_CLI`` is set, because the
    desktop binary would open a *new* VS Code window instead of routing
    through the tunnel.
    """
    import glob
    import shutil

    # In a VS Code remote/tunnel, prefer the server's remote-cli helper.
    # uv run and similar launchers may strip VSCODE_IPC_HOOK_CLI from the
    # current process, so also consult the recovered ancestor-process value.
    if os.environ.get("VSCODE_IPC_HOOK_CLI") or _find_vscode_ipc_hook():
        # The tunnel helper is typically at one of:
        #   ~/.vscode-server/bin/<commit>/bin/remote-cli/code
        #   ~/.vscode-server/cli/servers/.../server/bin/remote-cli/code
        #   ~/.vscode/cli/servers/.../server/bin/remote-cli/code  (newer tunnels)
        for pattern in [
            os.path.expanduser("~/.vscode-server/bin/*/bin/remote-cli/code"),
            os.path.expanduser(
                "~/.vscode-server/cli/servers/*/server/bin/remote-cli/code"
            ),
            os.path.expanduser("~/.vscode/cli/servers/*/server/bin/remote-cli/code"),
        ]:
            matches = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
            for m in matches:
                if os.access(m, os.X_OK):
                    return m

    found = shutil.which("code")
    if found:
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
    if os.environ.get("TERM_PROGRAM") == "vscode":
        return True
    if os.environ.get("VSCODE_IPC_HOOK_CLI"):
        return True
    # uv run and similar launchers strip env vars; walk ancestor processes.
    return _find_vscode_ipc_hook() is not None


def _is_vscode_remote() -> bool:
    """True when running inside a VS Code remote/tunnel session.

    This covers:
    - Linux SSH remote (VSCODE_IPC_HOOK_CLI set, non-macOS/Windows)
    - macOS/Windows SSH remote (SSH_CONNECTION set)
    - macOS/Windows tunnel remote: detected by finding a remote-cli binary in
      ~/.vscode/cli/servers/ or ~/.vscode-server/ (placed there by the tunnel
      server), which is only present when this machine IS the tunnel remote.
    """
    # Try env var first, then walk process tree (handles uv run env stripping).
    ipc = os.environ.get("VSCODE_IPC_HOOK_CLI") or _find_vscode_ipc_hook()
    if ipc:
        if sys.platform not in ("darwin", "win32"):
            return True
        # SSH remote on macOS/Windows
        if os.environ.get("SSH_CONNECTION") or os.environ.get("SSH_CLIENT"):
            return True
        # Tunnel remote on macOS/Windows: the remote-cli binary is present on
        # this machine only when it is acting as the tunnel server.
        code = _find_code_cli()
        if code and "remote-cli" in code:
            return True
    # TERM_PROGRAM=vscode + SSH_CONNECTION = VS Code SSH remote (belt-and-suspenders)
    if os.environ.get("TERM_PROGRAM") == "vscode" and (
        os.environ.get("SSH_CONNECTION") or os.environ.get("SSH_CLIENT")
    ):
        return True
    return False


def _in_vscode_tunnel() -> bool:
    """True when running inside a VS Code tunnel or SSH remote session."""
    if os.environ.get("SSH_CLIENT") or os.environ.get("SSH_CONNECTION"):
        return True
    if os.environ.get("VSCODE_INJECTION") or os.environ.get("VSCODE_AGENT_FOLDER"):
        return True
    return False


def _can_native_window() -> bool:
    """True if a pywebview native window can be opened.

    Returns False whenever a VS Code terminal is detectable, meaning we're
    running inside a VS Code terminal (local or remote/tunnel).  In that case
    we always prefer the Simple Browser route over a native window, because on
    a tunnel-server machine the user isn't looking at that screen.
    """
    # TERM_PROGRAM=vscode is the most reliable VS Code terminal indicator and
    # is preserved across uv-run / tmux where the IPC hook may be stripped.
    if os.environ.get("TERM_PROGRAM") == "vscode":
        return False
    # If a VS Code IPC hook is findable we are inside a VS Code terminal.
    if _find_vscode_ipc_hook():
        return False
    if _is_vscode_remote():
        return False
    # Plain SSH (no VS Code): the display is on the client machine, not here.
    if os.environ.get("SSH_CLIENT") or os.environ.get("SSH_CONNECTION"):
        return False
    if sys.platform in ("darwin", "win32"):
        return True
    # Linux/BSD: need a display server AND pywebview's GUI bindings
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        return False
    import importlib.util

    return (
        importlib.util.find_spec("qtpy") is not None
        or importlib.util.find_spec("gi") is not None
    )


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
