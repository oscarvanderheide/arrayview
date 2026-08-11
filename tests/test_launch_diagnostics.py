import json
import socket
import sys

import arrayview._launcher as launcher
import arrayview._platform as _platform_mod


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("localhost", 0))
        return int(sock.getsockname()[1])


def test_diagnose_serializes_shared_snapshot_and_plan(monkeypatch, capsys):
    port = _free_port()
    monkeypatch.setattr(
        sys,
        "argv",
        ["arrayview", "--diagnose", "--port", str(port), "--window", "browser"],
    )
    monkeypatch.setattr(_platform_mod, "_is_vscode_remote", lambda: False)
    monkeypatch.setattr(_platform_mod, "_in_vscode_terminal", lambda: False)

    launcher.arrayview()

    diagnostics = json.loads(capsys.readouterr().out)
    assert diagnostics["snapshot"]["invocation"] == "cli"
    assert diagnostics["snapshot"]["requested_window"] == "browser"
    assert diagnostics["snapshot"]["server"]["port"] == port
    assert diagnostics["plan"]["invocation"] == "cli"
    assert diagnostics["plan"]["requested_port"] == port
    assert diagnostics["plan"]["display"] == "browser"


def test_diagnose_retains_host_probes(monkeypatch, capsys):
    port = _free_port()
    monkeypatch.setattr(sys, "argv", ["arrayview", "--diagnose", "--port", str(port)])

    launcher.arrayview()

    diagnostics = json.loads(capsys.readouterr().out)
    assert diagnostics["loopback"]["host"] == "localhost"
    assert diagnostics["loopback"]["getaddrinfo"]
    assert diagnostics["loopback"]["bind_probe"]["ok"] is True
    assert "webview" in diagnostics["native_dependencies"]
    assert "ipc_hook_recovered" in diagnostics["vscode"]


# ---------------------------------------------------------------------------
# Failure copy
# ---------------------------------------------------------------------------

from arrayview._diagnostics import FATAL, SETUP, diagnose, format_failure

# The exact strings the opener produces on the first run after installing.
_REMOTE_INSTALLED = (
    "[ArrayView] VS Code viewer failed to become ready: ArrayView's VS Code "
    "opener was installed; reload this exact window once, then retry."
)
_LOCAL_INSTALLED = (
    "ArrayView display handoff failed: ArrayView updated its VS Code opener; "
    "reload this VS Code window once, then retry"
)


def test_freshly_installed_opener_reads_as_an_instruction_not_a_failure():
    """Installing the opener worked. The only thing left is a reload, so the
    message must not be dressed up as an error."""
    for message in (_REMOTE_INSTALLED, _LOCAL_INSTALLED):
        severity, _what, _fix = diagnose(message)
        assert severity == SETUP, f"{message!r} should be a setup step, got {severity}"

        rendered = format_failure(message, colour=False)
        assert "failed" not in rendered.lower(), (
            f"nothing failed here, got:\n{rendered}"
        )
        assert "--trace" not in rendered and "traceback" not in rendered.lower(), (
            f"there is no bug to trace, got:\n{rendered}"
        )
        assert "reload" in rendered.lower(), f"it must say to reload, got:\n{rendered}"
        assert "run the same command again" in rendered.lower(), (
            f"it must say what to do after the reload, got:\n{rendered}"
        )


def test_freshly_installed_opener_does_not_echo_the_internal_wording():
    """The instruction is the whole message; repeating the raw text under it
    is what made this read like a crash report."""
    rendered = format_failure(_REMOTE_INSTALLED, colour=False)
    assert "reload this exact window" not in rendered, (
        f"the internal detail should not be echoed, got:\n{rendered}"
    )
    assert len(rendered.splitlines()) == 2, (
        f"two lines: what happened, what to do. got:\n{rendered}"
    )


def test_a_freshly_installed_opener_is_not_a_stale_one():
    """The local wording matches the stale-opener pattern too, so ordering in
    the table is load-bearing."""
    severity, what, _fix = diagnose(_LOCAL_INSTALLED)
    assert severity == SETUP
    assert "older" not in what.lower(), (
        f"a just-installed opener is not an out-of-date one, got {what!r}"
    )


def test_a_real_failure_still_reads_as_one():
    """The setup path must not soften anything else."""
    rendered = format_failure("could not open this file: broken.npy", colour=False)
    severity, _what, _fix = diagnose("could not open this file: broken.npy")
    assert severity == FATAL
    assert "broken.npy" in rendered, (
        f"real failures still show what they were about, got:\n{rendered}"
    )


def test_integrated_browser_navigation_failure_does_not_blame_the_array():
    message = (
        "Integrated browser did not start the viewer script before recovery "
        "timeout"
    )
    _severity, what, fix = diagnose(message)
    assert "did not navigate" in what
    assert "too large" not in what.lower()
    assert "too large" not in fix.lower()
    assert "reload" in fix.lower()
    assert "run the same command again" not in fix.lower(), (
        "the diagnosed window is stuck, so retrying before reload repeats the failure"
    )


def test_an_unrecognized_failure_still_offers_the_traceback():
    severity, _what, fix = diagnose("something nobody has seen before")
    assert severity == FATAL
    assert "--trace" in fix
