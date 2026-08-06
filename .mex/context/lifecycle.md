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
last_updated: 2026-08-05
---

# Lifecycle

This contract describes who owns the backend, when it starts, and what closes it.

## Ownership Matrix

| Invocation | Display owner | Backend model | Shutdown/release |
|---|---|---|---|
| Local VS Code CLI `arrayview file.npy` | VS Code URL webview panel | Shared transient daemon | Panel close releases URL sessions; last viewer WebSocket close stops daemon |
| Plain Python script `view(arr)` | Browser/native/VS Code display | Non-daemon background server thread | Calling process remains alive until a viewer connects then closes, or the bounded connect timeout expires |
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

## Why a single-file click creates a tab at all

A click on an array is intercepted by registering as that file's editor, and
VS Code's rule is that an editor is always a visible tab. There is no pre-open
interception API left: the old `window.registerEditorOpener` hook that could
answer "handled" and prevent any tab from being created was removed from VS
Code (absent in 1.128; a 0.15.24 prototype that relied on it silently disabled
single-file interception). The only supported interception is the custom
editor, so a tab exists from the moment the file is clicked. Do not try to
remove it with an editor opener — that cannot work on current VS Code.

What that tab is used for depends on where the viewer can live:

- **Local VS Code and Remote SSH** — the viewer runs in that same webview, so
  the click tab is a placeholder that is navigated in place. One tab, no close.
- **Desktop tunnel** (and any `--window browser` request) — the viewer needs
  VS Code's built-in browser, which is always its own tab, so the click tab can
  never become the viewer; it is handed over instead. It keeps saying
  "Opening …" for the whole launch and is closed in the moment before the
  browser open command is issued, so the workbench applies the close and the
  open together. Three ordering constraints are load-bearing: return from
  `resolveCustomEditor` *without awaiting the launch* (disposing while VS Code
  still awaits it fails the click with "OverlayWebview has been disposed");
  close *before* the browser tab navigates, never during (overlapping the two
  killed 5 of 27 navigations); and close only after the route, the claim and
  the readiness journal have all succeeded, so a tab is never thrown away for a
  launch that then produces no viewer. Closing the click tab immediately
  instead was tried in v0.15.26: the tab paints anyway and the editor
  underneath shows for ~400 ms, which reads as more flicker, not less.
- **No tab at all**: the `arrayview.openFile` command (right-click → Open in
  ArrayView) opens no editor, so the viewer's tab is the only one that appears.
  Every editor route — left-click as default, Open With → ArrayView — goes
  through the custom editor and therefore through a tab.

The handover must not also be a rename, or one tab reads as two. Three labels
have to agree: the click tab's (`webviewPanel.title`, set to
`ArrayView: <name>` because VS Code would otherwise use the bare filename), the
viewer page's `<title>` (set at parse time from `av_name`), and the name
metadata later carries. The private launch route keeps its query out of the
URL, so the viewer's parse-time title script reads the injected launch query
first — reading only `location.search` there left every tunnel tab named after
the host until the array had loaded.
- **Folder launches** keep their placeholder tab: enumerating a folder can take
  minutes, and there is no click tab to reuse, so its own tab is the feedback.

Why the tunnel cannot just show the viewer in the click tab: a webview's
`portMapping` does not remap WebSocket ports, and the viewer uses one document
origin for both HTTP and WebSocket traffic. Only the built-in browser's remote
proxy carries both privately.

## Python Script

- `view(arr)` from a script keeps the calling process/server alive while the viewer is active; it does not promise that an in-process server survives process exit.
- The display call may return while the server thread remains owned by the calling process.
- When the last viewer instance closes, free arrays and shut the backend down.
- Quick viewer connect/disconnect races must count as "a viewer connected" so transient waiters do not linger until connect timeout.

## Jupyter

- Jupyter keeps the backend kernel-owned.
- An iframe disappearing should not hard-kill the backend.
- Explicit close or cleanup should free the session.
- Repeated `view()` calls should reuse the kernel-owned server when appropriate.

## Remote, Tunnel, And SSH

- Remote or tunnel launches may persist when `--serve` or tunnel display ownership requires it, but persistence must remain bounded. Current defaults use a 210-second viewer-connect timeout and a 1,800-second idle timeout unless configured otherwise.
- VS Code tunnel display uses the integrated browser and opens the verified backend loopback URL through VS Code's private remote-browser proxy. Each request has an independent browser identity; active or pending viewers must not force a later request onto a public URL.
- With multiple registered tunnel windows, a missing or stale `ARRAYVIEW_WINDOW_ID` recovers the exact live registration from the terminal IPC hook or the uniquely matching VS Code server root. If no exact registration can be recovered, delivery fails closed instead of broadcasting to a possibly wrong window.
- An exact registered `ARRAYVIEW_WINDOW_ID` wins; do not redirect it to a newer same-parent registration because live tunnel windows can share ancestry.
- Protocol request claims are atomic across extension hosts. Compatibility queue copies with the same request ID must never open in a sibling window or overwrite a terminal ACK.
- A tunnel may use a loopback backend URL only through VS Code's enabled integrated-browser remote proxy. If that private route, the exact backend, or the exact target window cannot be verified, delivery fails closed; it never promotes or falls back to a public tunnel URL. Remote-SSH remains separate and may legitimately resolve to a local forwarded URL. First-frame proof from the correlated backend phase journal remains the acceptance gate.
- Plain SSH should use `localhost` forwarding guidance and stay transient unless a shared server was explicitly requested.

## Shared Rules

- Global lifecycle state lives in `_session.py`.
- `release_session()` is the session-release primitive.
- Viewer WebSocket connect/disconnect owns active viewer counts.
- URL panel disposal must release every SID encoded in the URL: `sid`, `compare_sid`, `compare_sids`, and `overlay_sid`.
- The desktop-tunnel integrated browser fires no tab-disposal event. Its
  correlated viewer marks the SID for fenced WebSocket-disconnect release with
  a short reconnect grace period. (The `Tab` handle v0.15.18 captures is for
  blank-navigation recovery during launch only; it is not a disposal signal.)
- A desktop-tunnel request must never retry `workbench.action.browser.open`
  blindly. A tab that dropped its navigation never matches the reuse URL filter,
  so VS Code creates another visible tab instead of reusing it — real v0.15.14
  retries produced four tabs from one request.
- Retrying is allowed only after the exact tab from this request has been
  closed. v0.15.18 captures the `Tab` object its own open command created, and
  on a missing `script-loaded` closes that object with
  `vscode.window.tabGroups.close` before preparing fresh navigation state, up to
  four bounded attempts. Every ambiguity stops recovery rather than guessing:
  more than one new tab, a stale handle, a known text/diff/notebook/custom/
  terminal input, or a refused close.
- The integrated-browser tab reports `input=undefined` on the real host, so the
  reject-list is what makes closing safe. Do not convert it into a positive
  `TabInputWebview` requirement — that would disable recovery silently.
- If a desktop-tunnel window accepts the open command but never requests the
  viewer page, restarting only the extension host is insufficient. A full reload
  of that exact VS Code window clears the observed stuck browser state. Do not
  hide this boundary with a public webview fallback.
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

- Start with the exact public command in the real environment being changed. Success means the intended display opens, its first array frame renders, caller blocking/return behavior is correct, a second launch works, and closing it releases the session/process as designed.
- Classify evidence as `real host`, `real process`, `component`, or `unavailable`. Never report component or simulated coverage as real-host validation.
- `tests/lifecycle_matrix.py` records automated, real-process, local-state, and manual-only checks separately. A green exit with `MANUAL` rows does not prove those rows and is not a complete launch gate.
- `tests/test_lifecycle_contract.py` covers invocation ownership, release routes, transient daemon shutdown, and bundled VSIX lifecycle content.
- `tests/test_cli.py` covers CLI launch behavior.
- `tests/test_api.py` contains the affected WebSocket close and CLI helper coverage.
- `vscode-extension/test_lifecycle_helpers.js` covers URL SID collection and backend ping URL parsing.
- `vscode-extension/extension.js` must pass Node syntax checks after any extension change.
- GUI-affecting validation may open windows, tabs, prompts, install extensions, or reload an extension host. Warn the user first, and do not create temporary VS Code profiles or install/reload extensions without explicit permission.
