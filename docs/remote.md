# Remote

## SSH

Forward port 8000 when connecting over SSH:

```bash
ssh -L 8000:localhost:8000 user@remote
```

Then open `http://localhost:8000` in your local browser.

## VS Code

Auto-detects VS Code terminals and opens in a VS Code tab. Works automatically.

## VS Code tunnel

The VS Code extension uses the normal WebSocket viewer through VS Code's
forwarded-port support. ArrayView starts or reuses the FastAPI server, asks VS
Code to expose the port, promotes it to public when the tunnel API is
available, and opens the viewer in a VS Code tab.

```bash
arrayview volume.nii.gz     # opens in a webview tab automatically
```

### How it works

```
Viewer (VS Code tab) ←WebSocket/HTTP→ FastAPI server
```

The extension reads ArrayView's signal file, resolves the localhost URL through
VS Code's tunnel API, and opens that URL in a webview panel. Slice requests,
metadata, overlays, compare views, and shell tab injection all use the same
HTTP/WebSocket routes as local browser mode.

### Persistent server mode

For multi-hop setups or when sharing the viewer URL, you can run the server
explicitly:

```bash
arrayview --serve
```

Set port 8000 to Public in the VS Code Ports tab, then load arrays normally.
The server persists across invocations. Kill it with `arrayview --kill`.

## Multi-hop

When data lives on a server you SSH into from the tunnel-remote machine:

```
Local VS Code ──(devtunnel)──▶ remote ──(SSH)──▶ server
```

1. Start `arrayview --serve` on the remote machine, set port to Public.
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
