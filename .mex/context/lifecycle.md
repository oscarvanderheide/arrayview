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
last_updated: 2026-06-19
---

# Lifecycle

This contract describes who owns the backend, when it starts, and what closes it.

## Ownership Matrix

| Invocation | Display owner | Backend model | Shutdown/release |
|---|---|---|---|
| Local VS Code CLI `arrayview file.npy` | VS Code URL webview panel | Shared transient daemon | Panel close releases URL sessions; last viewer WebSocket close stops daemon |
| VS Code file click/custom editor | VS Code extension | Extension-owned stdio subprocess | Panel close kills subprocess |
| Plain Python script `view(arr)` | Browser/native/VS Code display | Non-daemon background server thread | Survives caller until viewer connects then closes |
| Jupyter `view(arr)` | Notebook kernel inline iframe | Kernel-owned daemon server thread | Iframe disappearance must not hard-kill backend |
| Julia/PythonCall | Browser/VS Code route from subprocess | Detached subprocess | Never in-process; avoid GIL deadlock |
| Remote/tunnel | VS Code direct webview when possible | stdio direct webview or intentional persistent server | Persistence allowed only when transport requires it |
| Plain SSH | User-forwarded localhost URL | Transient server unless `--serve` requested | Viewer close ends transient session |

## Local VS Code CLI

- `arrayview file.npy` from a local VS Code terminal should return to the prompt.
- Multiple local CLI launches may share one backend and open separate tabs.
- Closing one tab releases only that tab's arrays/sessions.
- Closing the last viewer tab should stop the transient daemon.
- The VS Code wrapper must not show "backend unavailable" based on a webview-side `fetch()`; backend health checks belong in the extension host.

## VS Code File Click And Custom Editor

- Prefer direct extension-owned stdio subprocesses.
- Do not require localhost or a shared port when direct mode is available.
- Closing the tab should terminate the subprocess.
- This path is transient and owned by the extension session, not by a long-lived server.

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

- Remote or tunnel launches may persist when `--serve` or direct-server constraints require it.
- Direct VS Code tunnel display should prefer stdio direct webview to avoid forwarded-port auth and public-port setup.
- With multiple registered tunnel windows, a missing or stale `ARRAYVIEW_WINDOW_ID` must fail closed with a diagnostic rather than broadcasting to whichever window is focused.
- An exact registered `ARRAYVIEW_WINDOW_ID` wins; do not redirect it to a newer same-parent registration because live tunnel windows can share ancestry.
- Plain SSH should use `localhost` forwarding guidance and stay transient unless a shared server was explicitly requested.

## Shared Rules

- Global lifecycle state lives in `_session.py`.
- `release_session()` is the session-release primitive.
- Viewer WebSocket connect/disconnect owns active viewer counts.
- URL panel disposal must release every SID encoded in the URL: `sid`, `compare_sid`, `compare_sids`, and `overlay_sid`.
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
