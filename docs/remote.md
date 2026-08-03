# Remote

## SSH

Start ArrayView on the remote host, then forward its port when connecting over
SSH:

```bash
ssh -L 8000:localhost:8000 user@remote
arrayview data.npy --port 8000
```

Then open `http://localhost:8000` in your local browser.

## VS Code

Auto-detects VS Code terminals and opens in a VS Code tab. Works automatically.

The same routing applies when `view()` is called from Python, MATLAB, or Julia.
See [MATLAB and Julia](foreign-hosts.md).

## Jupyter

Inline viewers in Jupyter use the notebook server as a bridge to ArrayView.
This is what makes remote notebooks work without forwarding a second port.
ArrayView keeps that route while it loads, even when the notebook server is
temporarily slow.

If the browser and kernel truly run on the same machine and the notebook server
does not provide its proxy route, direct localhost access can be requested
before importing ArrayView:

```python
import os
os.environ["ARRAYVIEW_JUPYTER_PROXY"] = "0"
```

Do not disable the proxy for a remote notebook: the browser's `localhost` is
then a different machine from the kernel.

## VS Code tunnel

The VS Code extension uses the normal WebSocket viewer through VS Code's
private remote-browser proxy. ArrayView starts or reuses the FastAPI server and
opens its remote `localhost` URL in VS Code's integrated browser. The port
remains private and is not exposed through a public developer-tunnel URL.

```bash
arrayview volume.nii.gz     # opens in a VS Code integrated-browser tab
```

### How it works

```
Integrated browser ←VS Code private proxy→ remote localhost FastAPI server
```

The extension reads ArrayView's signal file, verifies the exact backend and
target window, and opens a request-specific loopback URL. Each invocation gets
its own request identity so multiple ArrayView tabs can remain open at once.
Slice requests, metadata, overlays, and compare views use the same
HTTP/WebSocket routes as local browser mode. If the private proxy or exact
target cannot be verified, ArrayView fails instead of making the port public.

### Persistent server mode

Normal launches start or reuse the required server automatically. For multi-hop
setups or a shared viewer URL, run a persistent server explicitly:

```bash
arrayview --serve
```

Leave port 8000 private and load arrays normally. The server persists across
invocations. Stop it with `arrayview stop`.

## Multi-hop

When data lives on a server you SSH into from the tunnel-remote machine:

```
Local VS Code ──(devtunnel)──▶ remote ──(SSH)──▶ server
```

1. Start `arrayview --serve` on the remote machine and leave the port private.
2. SSH into the server with a reverse tunnel:

```bash
ssh -R 8000:localhost:8000 user@gpu-server
```

3. On the server:

```bash
arrayview array.npy
```

The array is sent back to the remote machine and the viewer opens in a VS Code tab locally.

If port 8000 is already taken on the GPU server:

```bash
ssh -R 8765:localhost:8000 user@gpu-server
arrayview array.npy --relay 8765
```

## Window modes

| Value | Behavior |
|-------|----------|
| `native` | Desktop window (default outside Jupyter) |
| `browser` | System browser |
| `vscode` | VS Code tab |
| `inline` | Inline IFrame (default in Jupyter) |
