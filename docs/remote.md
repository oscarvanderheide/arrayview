# Remote

## SSH

ArrayView detects SSH sessions and prints a port-forwarding hint. Forward port 8000 when connecting:

```bash
ssh -L 8000:localhost:8000 user@remote
```

Then open `http://localhost:8000` in your local browser.

## VS Code

ArrayView auto-detects VS Code terminals and opens arrays in Simple Browser. Works automatically for local terminals — no configuration needed.

## VS Code tunnel

For remote development through a VS Code tunnel, start the server on the remote machine:

```bash
# on the remote machine
arrayview --serve
```

Set port 8000 to Public in the VS Code Ports tab. Load arrays normally — each opens in Simple Browser:

```bash
arrayview scan.nii.gz
```

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

The array is sent back to the remote machine and the viewer opens in Simple Browser locally.

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
| `vscode` | VS Code Simple Browser |
| `inline` | Inline IFrame (default in Jupyter) |
