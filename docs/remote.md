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

The VS Code extension uses a **direct webview** transport: the viewer runs
inside a VS Code webview panel and communicates with a Python subprocess via
the extension host.  No port forwarding or public ports are needed — everything
stays inside the tunnel.

```bash
arrayview volume.nii.gz     # opens in a webview tab automatically
```

### How it works

```
Viewer (webview) ←postMessage→ Extension Host ←stdin/stdout→ Python
```

Instead of a WebSocket connection to a running server, the extension spawns a
dedicated Python process per array.  Slice requests travel through VS Code's
`postMessage` IPC and the extension relays them to Python's stdin as JSON.
Binary RGBA responses flow back through stdout with a length prefix.

This avoids the main limitation of the WebSocket approach: VS Code tunnels
don't expose arbitrary ports, so the old method required manually setting
port 8000 to "Public" in the Ports tab.

### Fallback: WebSocket mode

If you prefer the traditional WebSocket server (e.g. for multi-hop setups or
when sharing the viewer URL), you can still use it:

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
