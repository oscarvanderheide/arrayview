"""Real launch checks: real CLI, real server, real kernel, headless browser.

Nothing here imitates VS Code. Each test starts ArrayView the way a user
would and only passes once the array's first frame is drawn.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time

import httpx
import numpy as np
import pytest

from arrayview._platform import _find_vscode_ipc_hook


pytestmark = pytest.mark.browser


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("localhost", 0))
        return int(sock.getsockname()[1])


def _tcp_port_open(port: int) -> bool:
    try:
        with socket.create_connection(("localhost", port), timeout=0.2):
            return True
    except OSError:
        return False


def _wait_until(predicate, *, timeout: float, message: str):
    deadline = time.monotonic() + timeout
    last_error = None
    while time.monotonic() < deadline:
        try:
            value = predicate()
            if value:
                return value
        except Exception as exc:
            last_error = exc
        time.sleep(0.05)
    suffix = f"; last error: {last_error}" if last_error is not None else ""
    raise AssertionError(message + suffix)


def _wait_json(url: str, predicate, *, timeout: float = 15.0):
    def probe():
        response = httpx.get(url, timeout=0.5)
        if response.status_code != 200:
            return None
        payload = response.json()
        return payload if predicate(payload) else None

    return _wait_until(probe, timeout=timeout, message=f"no matching response from {url}")



def test_native_frame_observation_survives_shell_websocket_reconnect(
    page,
    client,
    server_url,
    tmp_path,
):
    array_path = tmp_path / "native-ready-reconnect.npy"
    np.save(array_path, np.arange(64, dtype=np.float32).reshape(8, 8))
    response = client.post(
        "/load",
        json={"filepath": str(array_path), "name": "native-ready-reconnect"},
    )
    response.raise_for_status()
    sid = response.json()["sid"]
    request_id = "native-ready-after-reconnect"

    page.goto(
        f"{server_url}/shell?init_sid={sid}&init_name=native-ready-reconnect"
    )
    page.frame_locator("iframe").locator("#canvas-wrap").wait_for(
        state="visible",
        timeout=15_000,
    )
    page.wait_for_function("() => ws && ws.readyState === WebSocket.OPEN")

    # Reproduce the race: the viewer's one-shot frame event arrives after the
    # shell socket starts closing but before its automatic reconnect.
    page.evaluate(
        """() => {
            window.__arrayviewReconnect = connectShellWS;
            connectShellWS = () => {};
            ws.close();
        }"""
    )
    page.wait_for_function("() => ws.readyState === WebSocket.CLOSED")
    page.evaluate(
        """({ sid, requestId }) => {
            tabs[sid].iframe.src = `/?sid=${sid}&native_request_id=${requestId}`;
        }""",
        {"sid": sid, "requestId": request_id},
    )
    page.wait_for_function(
        "key => nativeReadyObservations.has(key)",
        arg=f"{sid}:{request_id}",
        timeout=15_000,
    )
    page.evaluate(
        """() => {
            connectShellWS = window.__arrayviewReconnect;
            connectShellWS();
        }"""
    )

    _wait_json(
        f"{server_url}/ping",
        lambda body: f"{sid}:{request_id}"
        in body.get("native_ready_requests", []),
        timeout=5.0,
    )


def test_real_ipykernel_inline_session_renders_and_is_kernel_owned(
    page,
    monkeypatch,
    tmp_path,
):
    from jupyter_client.manager import start_new_kernel

    port = _free_port()
    kernel_root = tmp_path / "jupyter"
    kernel_dir = kernel_root / "kernels" / "arrayview-test"
    kernel_dir.mkdir(parents=True)
    (kernel_dir / "kernel.json").write_text(
        json.dumps(
            {
                "argv": [
                    sys.executable,
                    "-m",
                    "ipykernel_launcher",
                    "-f",
                    "{connection_file}",
                ],
                "display_name": "ArrayView test kernel",
                "language": "python",
            }
        )
    )
    monkeypatch.setenv("JUPYTER_PATH", str(kernel_root))
    monkeypatch.setenv("ARRAYVIEW_JUPYTER_PROXY", "0")

    manager, kernel = start_new_kernel(
        kernel_name="arrayview-test",
        startup_timeout=30,
    )
    try:
        message_id = kernel.execute(
            "\n".join(
                [
                    "import numpy as np",
                    "from IPython.display import display",
                    "import arrayview",
                    f"artifact = arrayview.view(np.arange(64, dtype=np.float32).reshape(8, 8), name='ipykernel-gate', window='inline', port={port})",
                    "display(artifact)",
                ]
            )
        )
        html_outputs = []
        kernel_errors = []
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            message = kernel.get_iopub_msg(timeout=5.0)
            if message.get("parent_header", {}).get("msg_id") != message_id:
                continue
            message_type = message["header"]["msg_type"]
            content = message["content"]
            if message_type in {"display_data", "execute_result"}:
                html = content.get("data", {}).get("text/html")
                if html:
                    html_outputs.append(html)
            elif message_type == "error":
                kernel_errors.append("\n".join(content.get("traceback", [])))
            elif message_type == "status" and content.get("execution_state") == "idle":
                break
        else:
            raise AssertionError("ipykernel execution did not become idle")

        assert not kernel_errors, "\n".join(kernel_errors)
        assert html_outputs, "ipykernel did not publish an inline HTML artifact"
        iframe_match = re.search(r"src=['\"]([^'\"]+)['\"]", html_outputs[-1])
        assert iframe_match, html_outputs[-1]
        initial_iframe_url = iframe_match.group(1)
        assert f"{port}" in initial_iframe_url
        assert "inline=1" in initial_iframe_url

        status = _wait_json(
            f"http://localhost:{port}/ping",
            lambda body: body.get("owner_mode") == "kernel"
            and body.get("active_sessions") == 1,
        )
        assert status["pid"] == manager.provisioner.pid

        # Explicit direct-inline mode remains valid when the browser and kernel
        # share localhost. Proxy behavior is covered through a real notebook
        # server below rather than emulated from a page with no proxy route.
        page.goto(f"http://localhost:{port}/")
        page.set_content(html_outputs[-1])
        viewer_frame = page.frame_locator("iframe")
        viewer_frame.locator("#canvas-wrap").wait_for(
            state="visible",
            timeout=15_000,
        )
        viewer_frame.locator("body").wait_for(state="visible")
        viewer_page = next(frame for frame in page.frames if frame != page.main_frame)
        viewer_page.wait_for_function(
            """() => {
                const canvas = document.querySelector('canvas#viewer');
                return !!(canvas && canvas.width && canvas.height);
            }""",
            timeout=15_000,
        )
        assert viewer_page.url.startswith(f"http://localhost:{port}/")
    finally:
        try:
            manager.shutdown_kernel(now=True)
        finally:
            kernel.stop_channels()

    _wait_until(
        lambda: not _tcp_port_open(port),
        timeout=10.0,
        message="kernel-owned ArrayView server survived kernel shutdown",
    )


def test_real_jupyter_classic_slow_proxy_keeps_selected_route(
    page,
    tmp_path,
):
    """A slow notebook proxy must not send a remote browser to localhost."""
    notebook_port = _free_port()
    arrayview_port = _free_port()
    token = "arrayview-classic-test-token"
    kernel_root = tmp_path / "jupyter"
    kernel_dir = kernel_root / "kernels" / "arrayview-test"
    kernel_dir.mkdir(parents=True)
    (kernel_dir / "kernel.json").write_text(
        json.dumps(
            {
                "argv": [
                    sys.executable,
                    "-m",
                    "ipykernel_launcher",
                    "-f",
                    "{connection_file}",
                ],
                "display_name": "ArrayView test kernel",
                "language": "python",
            }
        )
    )

    cells = [
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "arrayview-0",
            "metadata": {},
            "outputs": [],
            "source": [
                "import numpy as np\n",
                "from arrayview import view\n",
                "shape = (12, 218, 170)\n",
                "base = np.linspace(0, 1, np.prod(shape), dtype=np.float32).reshape(shape)\n",
                "arr = (base + 1j * (1 - base)).astype(np.complex64)\n",
                f"view(arr.copy(), name='classic-view-0', window='inline', port={arrayview_port})\n",
            ],
        }
    ]
    for index in range(1, 6):
        cells.append(
            {
                "cell_type": "code",
                "execution_count": None,
                "id": f"arrayview-{index}",
                "metadata": {},
                "outputs": [],
                "source": [
                    f"view(arr.copy(), name='classic-view-{index}', window='inline', port={arrayview_port})\n"
                ],
            }
        )
    notebook_path = tmp_path / "repro.ipynb"
    notebook_path.write_text(
        json.dumps(
            {
                "cells": cells,
                "metadata": {
                    "kernelspec": {
                        "display_name": "ArrayView test kernel",
                        "language": "python",
                        "name": "arrayview-test",
                    },
                    "language_info": {"name": "python", "version": "3"},
                },
                "nbformat": 4,
                "nbformat_minor": 5,
            }
        )
    )

    env = dict(os.environ)
    env["JUPYTER_PATH"] = str(kernel_root)
    env["ARRAYVIEW_JUPYTER_PROXY"] = "1"
    env["ARRAYVIEW_DISCONNECT_GRACE_SECONDS"] = "0.2"
    nbclassic = Path(sys.executable).with_name("jupyter-nbclassic")
    assert nbclassic.is_file()
    log_path = tmp_path / "nbclassic.log"
    log_handle = log_path.open("w")
    server = subprocess.Popen(
        [
            str(nbclassic),
            "--no-browser",
            f"--ServerApp.port={notebook_port}",
            "--ServerApp.port_retries=0",
            "--ServerApp.ip=localhost",
            f"--ServerApp.root_dir={tmp_path}",
            f"--IdentityProvider.token={token}",
            "--ServerApp.allow_remote_access=False",
        ],
        cwd=tmp_path,
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    base_url = f"http://localhost:{notebook_port}"
    websocket_urls = []
    direct_requests = []
    request_failures = []
    console_messages = []
    page.on("websocket", lambda ws: websocket_urls.append(ws.url))
    page.on(
        "requestfailed",
        lambda request: request_failures.append(
            {"url": request.url, "failure": str(request.failure)}
        ),
    )
    page.on("console", lambda message: console_messages.append(message.text))

    def _delay_proxy_document(route, request):
        if request.resource_type == "document":
            time.sleep(2.0)
        route.continue_()

    page.route(
        f"{base_url}/proxy/{arrayview_port}/**",
        _delay_proxy_document,
    )

    def _reject_direct_request(route, request):
        direct_requests.append(request.url)
        route.abort()

    page.route(
        f"http://localhost:{arrayview_port}/**",
        _reject_direct_request,
    )
    try:
        _wait_until(
            lambda: httpx.get(
                f"{base_url}/api?token={token}", timeout=0.5
            ).status_code
            == 200,
            timeout=20.0,
            message="nbclassic server did not become ready",
        )
        page.set_viewport_size({"width": 1400, "height": 4200})
        page.goto(
            f"{base_url}/notebooks/{notebook_path.name}?token={token}",
            wait_until="domcontentloaded",
        )
        page.wait_for_function(
            """() => window.Jupyter
                && Jupyter.notebook
                && Jupyter.notebook.kernel
                && Jupyter.notebook.kernel.is_connected()""",
            timeout=30_000,
        )
        page.evaluate(
            """() => {
                window.__avRenderedFrames = new Map();
                window.__avEvents = [];
                window.addEventListener('message', event => {
                    const message = event.data;
                    if (!message || message.source !== 'arrayview-viewer') return;
                    window.__avEvents.push(message.phase);
                    if (message.phase === 'frame-rendered') {
                        window.__avRenderedFrames.set(event.source, message.detail);
                    }
                });
            }"""
        )
        page.evaluate("() => Jupyter.notebook.execute_all_cells()")
        page.wait_for_function(
            """() => document.querySelectorAll("iframe[title='ArrayView']").length === 6""",
            timeout=60_000,
        )
        try:
            page.wait_for_function(
                """() => window.__avRenderedFrames.size === 6
                    && Array.from(window.__avRenderedFrames.values()).every(
                        detail => detail && detail.width > 0 && detail.height > 0
                    )""",
                timeout=60_000,
            )
        except Exception:
            status = None
            try:
                status = httpx.get(
                    f"http://localhost:{arrayview_port}/ping", timeout=0.5
                ).json()
            except Exception:
                pass
            diagnostics = {
                "ping": status,
                "frame_urls": [
                    frame.url for frame in page.frames if "sid=" in frame.url
                ],
                "websocket_urls": websocket_urls,
                "request_failures": request_failures,
                "console": console_messages,
                "events": page.evaluate("() => window.__avEvents"),
                "nbclassic_log": log_path.read_text(),
            }
            pytest.fail(json.dumps(diagnostics, indent=2, sort_keys=True))

        status = _wait_json(
            f"http://localhost:{arrayview_port}/ping",
            lambda body: body.get("owner_mode") == "kernel"
            and body.get("active_sessions") == 6
            and body.get("active_viewer_sockets") == 6
            and body.get("viewer_connections_seen", 0) >= 6,
        )
        assert status["active_viewer_sockets"] == 6
        frame_urls = [frame.url for frame in page.frames if "sid=" in frame.url]
        assert len(frame_urls) == 6
        assert all(
            f"/proxy/{arrayview_port}/" in frame_url
            for frame_url in frame_urls
        ), frame_urls
        proxy_websockets = [
            url
            for url in websocket_urls
            if f"/proxy/{arrayview_port}/ws/" in url
        ]
        assert len(proxy_websockets) == 6, websocket_urls
        assert direct_requests == []

        original_urls = set(frame_urls)
        page.evaluate("() => Jupyter.notebook.get_cell(2).clear_output()")
        _wait_json(
            f"http://localhost:{arrayview_port}/ping",
            lambda body: body.get("active_viewer_sockets") == 5
            and body.get("active_sessions") == 5,
        )
        page.evaluate("() => Jupyter.notebook.get_cell(2).execute()")
        page.wait_for_function(
            "() => window.__avRenderedFrames.size === 7",
            timeout=60_000,
        )
        _wait_json(
            f"http://localhost:{arrayview_port}/ping",
            lambda body: body.get("active_viewer_sockets") == 6
            and body.get("active_sessions") == 6
            and body.get("viewer_connections_seen", 0) >= 7,
        )
        replacement_urls = {
            frame.url for frame in page.frames if "sid=" in frame.url
        }
        assert len(replacement_urls) == 6
        assert len(original_urls & replacement_urls) == 5
        assert direct_requests == []
    finally:
        page.close()
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(server.pid, signal.SIGKILL)
            server.wait(timeout=5)
        log_handle.close()

    _wait_until(
        lambda: not _tcp_port_open(notebook_port),
        timeout=10.0,
        message="nbclassic server survived test cleanup",
    )
    _wait_until(
        lambda: not _tcp_port_open(arrayview_port),
        timeout=10.0,
        message="kernel-owned ArrayView server survived nbclassic cleanup",
    )


_CANVAS_HAS_PIXELS = """() => {
    const canvas = document.querySelector('canvas#viewer');
    if (!canvas || !canvas.width || !canvas.height) return false;
    const pixels = canvas.getContext('2d').getImageData(
        0, 0, canvas.width, canvas.height
    ).data;
    for (let i = 0; i < pixels.length; i += 4) {
        if (pixels[i] || pixels[i + 1] || pixels[i + 2]) return true;
    }
    return false;
}"""


@pytest.mark.skipif(sys.platform == "win32", reason="opener stand-in is a shell script")
@pytest.mark.skipif(
    bool(os.environ.get("VSCODE_IPC_HOOK_CLI") or _find_vscode_ipc_hook()),
    reason="started from VS Code, which ArrayView detects and opens a tab in instead",
)
def test_cli_browser_launch_renders_and_leaves_nothing_running(browser, tmp_path):
    """`arrayview file.npy --window browser` draws the array, then cleans up."""
    array_path = tmp_path / "launch.npy"
    np.save(array_path, np.linspace(0, 1, 32 * 24, dtype=np.float32).reshape(32, 24))

    # Stand-in for the system browser: record the URL, the test opens it.
    # macOS opens URLs with `open`, Linux with `xdg-open` or $BROWSER.
    url_file = tmp_path / "opened-url.txt"
    opener_bin = tmp_path / "bin"
    opener_bin.mkdir()
    for name in ("open", "xdg-open"):
        opener = opener_bin / name
        opener.write_text(f'#!/bin/sh\necho "$1" >> "{url_file}"\n')
        opener.chmod(0o755)

    home = tmp_path / "home"
    home.mkdir()
    runtime = tmp_path / "runtime"
    env = {
        "PATH": f"{opener_bin}{os.pathsep}{os.environ['PATH']}",
        "HOME": str(home),
        "USER": os.environ.get("USER", "arrayview-test"),
        "BROWSER": str(opener),
        "ARRAYVIEW_RUNTIME_DIR": str(runtime),
    }
    port = _free_port()
    console = Path(sys.executable).with_name("arrayview")
    assert console.is_file(), f"console entry point missing: {console}"

    cli = subprocess.Popen(
        [str(console), str(array_path), "--window", "browser", "--port", str(port)],
        cwd=tmp_path,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    context = None
    try:
        url = _wait_until(
            lambda: url_file.is_file() and url_file.read_text().strip(),
            timeout=25,
            message="the CLI never asked the browser to open a URL",
        ).splitlines()[0]
        assert url.startswith(f"http://localhost:{port}/")

        context = browser.new_context()
        page = context.new_page()
        page.goto(url)
        page.wait_for_function(_CANVAS_HAS_PIXELS, timeout=15_000)

        stdout, stderr = cli.communicate(timeout=20)
        assert cli.returncode == 0, f"stdout={stdout}\nstderr={stderr}"
        assert "Traceback" not in stderr

        context.close()
        context = None
        stop = subprocess.run(
            [str(console), "stop"],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert stop.returncode == 0, f"stdout={stop.stdout}\nstderr={stop.stderr}"
        _wait_until(
            lambda: not _tcp_port_open(port),
            timeout=10,
            message="ArrayView server still running after the viewer closed",
        )
        assert not list((runtime / "instances").glob("*.json"))
    finally:
        if context is not None:
            context.close()
        if cli.poll() is None:
            cli.kill()
            cli.communicate(timeout=5)
