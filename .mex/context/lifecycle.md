---
name: lifecycle
description: Ownership contract for ArrayView backends, viewer sessions, VS Code tabs, and shutdown/release behavior.
triggers:
  - "lifecycle"
  - "server ownership"
  - "startup"
  - "shutdown"
  - "orphan process"
  - "VS Code tab"
  - "backend unavailable"
  - "Jupyter iframe"
  - "SSH"
edges:
  - target: context/architecture.md
    condition: when component boundaries or display routing need broader context
  - target: context/stack.md
    condition: when VS Code, FastAPI, WebSocket, or packaging details are needed
last_updated: 2026-07-14
---

# Lifecycle

This contract describes who owns the backend, when it starts, and what closes it.

## Ownership Matrix

| Invocation | Display owner | Backend model | Shutdown/release |
|---|---|---|---|
| Local VS Code CLI `arrayview file.npy` | VS Code URL webview panel | Shared transient daemon | Panel close releases URL sessions; last viewer WebSocket close stops daemon |
| Plain Python script `view(arr)` | Browser/native/VS Code display | Non-daemon background server thread | Survives caller until viewer connects then closes |
| Jupyter `view(arr)` | Notebook kernel inline iframe | Kernel-owned daemon server thread | Iframe disappearance must not hard-kill backend |
| Julia/PythonCall | Browser/VS Code route from subprocess | Detached subprocess | Never in-process; avoid GIL deadlock |
| Remote/tunnel | VS Code integrated browser (desktop tunnel) or URL webview panel | Forwarded WebSocket server | Persistent only when `--serve` or tunnel ownership requires it |
| Plain SSH | User-forwarded localhost URL | Transient server unless `--serve` requested | Viewer close ends transient session |

## Local VS Code CLI

- `arrayview file.npy` from a local VS Code terminal should return to the prompt.
- Multiple local CLI launches may share one backend and open separate tabs.
- Closing one tab releases only that tab's arrays/sessions.
- Closing the last viewer tab should stop the transient daemon.
- The VS Code wrapper must not show "backend unavailable" based on a webview-side `fetch()`; backend health checks belong in the extension host.

## Python Script

- `view(arr)` from a script should survive the script exiting.
- The backend must outlive the caller until viewer instances close.
- When the last viewer instance closes, free arrays and shut the backend down.
- Quick viewer connect/disconnect races must count as "a viewer connected" so transient waiters do not linger until connect timeout.

## Jupyter

- Jupyter keeps the backend kernel-owned.
- An iframe disappearing should not hard-kill the backend.
- Explicit close or cleanup should free the session.
- Repeated `view()` calls should reuse the kernel-owned server when appropriate.

## Remote, Tunnel, And SSH

- Remote or tunnel launches may persist when `--serve` or tunnel display ownership requires it.
- Desktop VS Code tunnel display uses the integrated browser. With VS Code's remote browser proxy enabled it opens the backend URL directly; otherwise it automatically promotes the tunnel port to public and requires a verified non-loopback `asExternalUri` URL. Other remote hosts retain the URL webview path.
- With multiple registered tunnel windows, a missing `ARRAYVIEW_WINDOW_ID` first recovers the exact live registration from the terminal's IPC hook. Only when no exact registration exists does it use the shared focused-window broadcast fallback.
- An exact registered `ARRAYVIEW_WINDOW_ID` wins; do not redirect it to a newer same-parent registration because live tunnel windows can share ancestry.
- Protocol request claims are atomic across extension hosts. Compatibility queue copies with the same request ID must never open in a sibling window or overwrite a terminal ACK.
- A desktop tunnel may use a loopback backend URL only through VS Code's enabled integrated-browser remote proxy; otherwise desktop and web-hosted tunnels require a verified non-loopback public URL. Remote-SSH may legitimately resolve to a local forwarded URL and must not be subjected to tunnel-only public-port commands. First-frame proof from the correlated backend phase journal remains the acceptance gate.
- Plain SSH should use `localhost` forwarding guidance and stay transient unless a shared server was explicitly requested.

## Shared Rules

- Global lifecycle state lives in `_session.py`.
- `release_session()` is the session-release primitive.
- Viewer WebSocket connect/disconnect owns active viewer counts.
- URL panel disposal must release every SID encoded in the URL: `sid`, `compare_sid`, `compare_sids`, and `overlay_sid`.
- Tunnel webview-panel disposal posts release requests to the local backend immediately;
  the forwarded public URL is display-only and is not the cleanup authority.
- The desktop-tunnel integrated browser has no stable tab-disposal handle. Its
  correlated viewer marks the SID for fenced WebSocket-disconnect release with
  a short reconnect grace period.
- Reused file and collection sessions acquire one lease per tab, so closing one
  tab cannot invalidate another tab that shares the same SID.
- A VS Code readiness ACK includes the live opener version and is terminal only
  after the requested SID exists and the viewer reports its first rendered frame.
- Existing-server tunnel loads publish a pending SID before loading large files, so port resolution can overlap disk I/O. Pending metadata probes return immediately and the WebSocket waits on the shared pending-session event.
- Older ArrayView packages must not delete or downgrade a newer installed opener.
- Tunnel registration cleanup must not remove live same-tunnel sibling windows.
- Explicit cleanup wins over implicit disappearance.
- Any VS Code extension source change must rebuild `src/arrayview/arrayview-opener.vsix` and keep the packaged version in sync.

## Verification Anchors

- `tests/lifecycle_matrix.py` is the top-level lifecycle gate; it reports automated, real-process, local-state, and manual-only checks separately.
- `tests/test_lifecycle_contract.py` covers invocation ownership, release routes, transient daemon shutdown, and bundled VSIX lifecycle content.
- `tests/test_cli.py` covers CLI launch behavior.
- `tests/test_api.py` contains the affected WebSocket close and CLI helper coverage.
- `vscode-extension/test_lifecycle_helpers.js` covers URL SID collection and backend ping URL parsing.
- `vscode-extension/extension.js` must pass Node syntax checks after any extension change.
