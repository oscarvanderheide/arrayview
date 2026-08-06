---
name: handoff-vscode-open-flicker
description: Handoff for the VS Code tab-flicker task (opening arrays showed a temporary tab). What was changed in v0.15.26, what is proven impossible, and what still needs real-host confirmation.
last_updated: 2026-08-05
---

# Handoff: VS Code tab-flicker when opening arrays

## The user's ask (plain language)

Click an array, get one tab with ArrayView in it. No temporary tab, no second
tab appearing, no tab closing behind it. A slow array should show a loading
message inside the viewer, not in a separate tab. The placeholder tab for
folder launches is fine. The right-click entry and the
"Open With → ArrayView → set as default" workflow must both keep working.

## The right-click entry never appeared (fixed in v0.15.29)

v0.15.25 gated it on `explorerResourceIsFile`, which is not a VS Code context
key — the clause was always false, so the menu item was never rendered and the
command was never invoked. That is why the live log showed every open still
going through the custom editor. The gate is now `!explorerResourceIsFolder`.
`tests/test_lifecycle_contract.py` asserted the broken key and was updated.

## Two routes, and only one of them can ever show a tab

- **`arrayview.openFile` command** (right-click → Open in ArrayView, or the
  command palette) opens no editor at all. The viewer's tab is the only tab
  that ever appears. This is the genuinely flicker-free route.
- **Any editor route** — left-click when ArrayView is the default, or
  Open With → ArrayView — is the custom editor. VS Code opens the editor, and
  therefore the tab, before the extension is called at all. On a desktop
  tunnel that tab cannot become the viewer, so it can only be handed over.

## What v0.15.28 changes

The handover is now a single step. On a desktop tunnel the click tab keeps
saying "Opening …" for the whole launch and is closed in the moment *before*
`workbench.action.browser.open` is issued, so the workbench applies the close
and the open together: never both tabs, never neither.

v0.15.26 tried the opposite — closing the click tab immediately after
`resolveCustomEditor` returned. It works (log: `closed click tab` 16 ms after
the click) and it is worse: the tab still paints, and then the previously
active editor shows for the ~400 ms until the viewer tab arrives. Do not go
back to it.

Touched in `vscode-extension/extension.js`:

- `_viewerOpensInBuiltInBrowser()` — the route test, answerable at click time.
- `_launchWithStatusProgress()` / `_settleLaunchProgress()` /
  `_pendingLaunchProgress` — a `ProgressLocation.Window` spinner correlated by
  `handoffPath`, running from the click until the viewer page reports
  `script-loaded`. Used by the custom editor *and* by `arrayview.openFile`,
  which previously reported nothing at all while it launched.
- `resolveCustomEditor` — on the built-in-browser route it registers the
  placeholder and returns **without awaiting the launch**. That is what makes
  the later dispose legal; the 2026-08-04 attempt awaited it, so its dispose
  always landed while VS Code was still resolving the editor.
- `openInIntegratedBrowser(..., onBeforeNavigate)` — fired once, immediately
  before the first navigation command, after everything that could still fail
  cheaply has succeeded. The click tab closes there.

Everywhere else (local VS Code, Remote SSH) is untouched: the click tab is
still the placeholder that gets navigated into the viewer.

## Measured before the change (real host, `~/.arrayview/extension.log`)

| launch | click → viewer tab | both tabs visible |
|---|---|---|
| cold, no daemon | 2.1 s | 0.5 s |
| warm daemon (FASTLOAD) | 0.34 s | 0.27 s |
| warm, page stalled twice | 0.37 s | 3.7 s |

## Proven dead-ends — do not retry

- `vscode.window.registerEditorOpener` does not exist in modern VS Code
  (verified absent in the 1.128 server `out/` and the API reference). A
  0.15.24 prototype using it silently left clicks unintercepted.
- There is no pre-open interception API. The custom editor is the only
  mechanism, and a custom editor is always a tab.
- The viewer cannot live in the click tab on a tunnel: webview `portMapping`
  does not remap WebSocket ports (see `_backendPortMapping`), and the viewer
  uses one origin for HTTP and WebSocket. Only the built-in browser's remote
  proxy carries both privately.
- Disposing the click tab *while* `resolveCustomEditor` is still awaited fails
  the click with "OverlayWebview has been disposed" (2026-08-04).
- Disposing a webview *during* the browser tab's navigation killed 5 of 27
  navigations (0 of 13 once the wait was added).

## Still to confirm on the real host

1. Click an array. Expect: one tab, the ArrayView tab. A status-bar spinner
   while it opens. No tab that appears and then closes.
2. Check the log for `CUSTOM-EDITOR: closed click tab` followed by
   `CUSTOM-EDITOR: launch spinner ended (viewer page loaded)`, and **no**
   "OverlayWebview has been disposed" error notification.
3. Confirm Remote SSH / local VS Code still navigate the click tab in place
   (`HANDOFF: navigated placeholder …`), unchanged.
4. Second, separate flicker source still present: when the viewer page does not
   load inside the pre-script budget, the viewer tab itself is closed and
   reopened (`PANEL: closed exact blank tab; retrying navigation`). Seen twice
   in one launch on 2026-08-05. Do not retune that threshold without
   measurement — see `.mex/patterns/debug-vscode-extension-python.md`.

## Rebuild / install recipe (exact active host)

- Bump `vscode-extension/package.json` **and** `_VSCODE_EXT_VERSION` in
  `src/arrayview/_vscode_extension.py` together.
- `cd vscode-extension && npx --yes @vscode/vsce@3.7.1 package
  --allow-star-activation -o ../src/arrayview/arrayview-opener.vsix`
- Install through `_ensure_vscode_extension()`; the tunnel host loads
  `~/.vscode-server/extensions`, while the plain `code --install-extension` CLI
  writes `~/.vscode`.
- The user must reload the window to activate the new version.

## How to test

- `node --check vscode-extension/extension.js`; the Node suite in
  `vscode-extension/test_*.js` all passes.
- `uv run pytest tests/test_lifecycle_contract.py -k "menu or vsix or bundled
  or version or folder"` and `uv run pytest tests/test_vscode_ack_protocol.py`.
- Two failures in `tests/test_lifecycle_contract.py`
  (`test_remote_vscode_spawned_daemon_keeps_backend_persistent`,
  `test_integrated_launch_cleanup_is_scoped_per_request_token`) are unrelated
  and fail on the clean tree too.
