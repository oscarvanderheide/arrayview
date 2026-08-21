# Integrated Browser intermittently drops initial localhost navigation in Remote Tunnel after idle

## Environment

- VS Code 1.132.1 (`c2d1b13fdc4a77628e5f3bb70173351c8f2fbad1`)
- Desktop connected to a Linux host through Remote Tunnel
- Desktop client OS was not captured; the remote host is Linux
- `workbench.browser.enableRemoteProxy: true`
- HTTP server listening only on remote `localhost`

## Reproduction

1. On the remote host, start a server that logs every request:

   ```sh
   node -e "require('http').createServer((req,res)=>{console.log(Date.now(),req.url);res.end('ok')}).listen(8123,'localhost')"
   ```

2. From a Remote Tunnel extension, open one fixed URL in the Integrated
   Browser:

   ```js
   await vscode.commands.executeCommand('workbench.action.browser.open', {
     url: 'http://localhost:8123/repro/fixed',
     openToSide: false,
     reuseUrlFilter: '/repro/**'
   });
   ```

3. Leave new-tab navigation idle for about 60 seconds.
4. Repeat the command. If the tab stays blank, repeat the same command with the
   byte-identical URL every ~1.5 seconds.

## Expected

The initial navigation waits for the remote browser connection and reaches the
remote server once.

## Actual

The tab opens, but the initial navigation can produce no server request and is
never replayed. Reissuing the identical URL eventually succeeds.

In one measured Remote Tunnel run after 56 seconds idle:

- the tab opened at `21:46:41.826` UTC;
- the remote backend remained reachable from the remote extension host in
  9–14 ms;
- the initial navigation plus two identical re-navigations produced no server
  request;
- the fourth identical navigation reached the server at `21:46:46.845`, 5.02 s
  after the initial command;
- the page rendered at `21:46:47.581`, 5.76 s after the initial command.

A separate run with retries disabled remained blank for 40 seconds with no
server request. Waiting alone therefore did not recover the navigation.

## Diagnostics / likely readiness gap

The current readiness promise covers applying Electron's local proxy
configuration, but the upstream remote connection is established later and
lazily. Proxy intent can also be enabled while proxy information is still
absent, with `undefined` representing both “off” and “not ready.” A transient
upstream failure resets the HTTP response and does not replay the navigation.

Relevant source:

- [`BrowserSessionRemote.whenReady`](https://github.com/microsoft/vscode/blob/c2d1b13fdc4a77628e5f3bb70173351c8f2fbad1/src/vs/platform/browserView/electron-main/browserSessionRemote.ts#L95-L138)
- [`BrowserView.loadURL`](https://github.com/microsoft/vscode/blob/c2d1b13fdc4a77628e5f3bb70173351c8f2fbad1/src/vs/platform/browserView/electron-main/browserView.ts#L689-L694)
- [Lazy upstream connection and reset-on-error](https://github.com/microsoft/vscode/blob/c2d1b13fdc4a77628e5f3bb70173351c8f2fbad1/src/vs/platform/tunnel/node/tunnelProxy.ts#L364-L407)

## Candidate fix and tests

Represent remote proxy state as `off | pending | ready | failed`, and hold the
initial navigation while proxying is pending. Include the upstream remote
connection in readiness, or replay the initial idempotent navigation once
after a transient wake/reconnect failure. Surface a bounded error rather than
leaving a blank tab.

Add regression tests that delay both proxy-info delivery and the upstream
tunnel connection, then assert the initial URL reaches the server exactly once
after readiness. Add a transient first-connect failure case and assert
automatic recovery without another user navigation.

## Related, not duplicate

- [#321294 — Browser in Remote Workspaces test plan](https://github.com/microsoft/vscode/issues/321294): Tunnel coverage, but no idle/reconnect case.
- [#324828 — localhost request fails before reaching backend](https://github.com/microsoft/vscode/issues/324828): similar symptom, but Dev Container/custom `.localhost` hostname and persistent behavior.
- [#321440 — managed connection proxy support](https://github.com/microsoft/vscode/issues/321440): earlier managed-connection problem, already fixed.

Desktop-side **Extension Host** output around a failed attempt can be provided
in a follow-up if needed; that is where the proxy runs.
