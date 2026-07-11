# ArrayView Remote Tunnel Test Handoff

Status: executed in real VS Code remote tunnel windows on 2026-07-11;
concurrent signaling was fixed and retested; disconnect/reconnect remains.

## Build under test

- Branch: `plan/robust-launch-lifecycle`
- Minimum implementation commit: `667a183`
- Python package: workspace checkout via `uv run`
- VS Code extension: `0.14.37`
- Bundled VSIX: `src/arrayview/arrayview-opener.vsix`
- ACK protocol: `1`
- Default tunnel test port: `8000`

Before testing, record actual values:

```bash
git branch --show-current
git rev-parse --short HEAD
uname -a
uv run python --version
uv run python -c "from arrayview import __version__; print(__version__)"
uv run python -c "from arrayview._vscode_extension import _VSCODE_EXT_VERSION; print(_VSCODE_EXT_VERSION)"
```

Record the actual commit. Do not test a commit older than `667a183`.

## Already proven locally

Do not repeat these unless a tunnel failure points back to them:

- 203 Python launch/lifecycle contract tests passed locally.
- `node --check vscode-extension/extension.js` passed.
- `node vscode-extension/test_lifecycle_helpers.js` passed.
- bundled VSIX source/version/content checks passed.
- the planner matrix covers CLI, Python, Jupyter, Julia, MATLAB, Explorer,
  local/SSH/remote placement, and macOS/Linux/Windows facts.
- registry locking, PID reuse, foreign-listener refusal, identity-safe stop,
  explicit session close, ACK correlation, and bounded timeouts have automated
  coverage.

## Required setup

Use a remote Linux or macOS host opened through VS Code Tunnel. Ideally keep two
local VS Code windows attached to the same tunnel so exact window routing can be
tested. Do not use a raw SSH terminal outside VS Code for the window-routing
scenarios.

```bash
uv sync --frozen --group dev --group test
uv run python -c "import numpy as np; np.save('/tmp/arrayview-tunnel-a.npy', np.arange(4096, dtype=np.float32).reshape(64, 64)); np.save('/tmp/arrayview-tunnel-b.npy', np.eye(64, dtype=np.float32))"
uv run arrayview doctor --json --port 8000 > /tmp/arrayview-doctor-before.json
uv run arrayview instances --json
```

Open the VS Code output/log locations before testing:

- `~/.arrayview/extension.log`
- `~/.arrayview/open-ack-v0100-*.json`
- VS Code Ports view for port `8000`

The Python environment should automatically install/update the bundled
extension. If VS Code asks for an extension-host reload, do it once, record it,
and rerun the scenario.

## Scenario 1: single tunnel launch and terminal ACK

From the integrated terminal in tunnel window A:

```bash
uv run arrayview /tmp/arrayview-tunnel-a.npy --window vscode --port 8000 --verbose 2>&1 | tee /tmp/arrayview-tunnel-single.log
```

Expected:

- the command returns successfully only after `backend_ready`;
- exactly one viewer opens in window A;
- `arrayview instances --json` reports one live instance on port 8000;
- the newest ACK contains matching non-empty `requestId`, `windowId`, and
  `serverId`, ending in `backend_ready`;
- `serverId` equals `/ping.instance_id` through the resolved viewer URL;
- extension log shows `claimed`, `port_resolved`, `panel_opened`,
  `visibility_verified`, then `backend_ready` for the same request ID.

Capture:

```bash
cp "$(ls -t ~/.arrayview/open-ack-v0100-*.json | head -1)" /tmp/arrayview-ack-single.json
cp ~/.arrayview/extension.log /tmp/arrayview-extension-single.log
uv run arrayview instances --json > /tmp/arrayview-instances-single.json
```

## Scenario 2: exact routing with two tunnel windows

Keep window A open. In tunnel window B run:

```bash
uv run arrayview /tmp/arrayview-tunnel-b.npy --window vscode --port 8000 --verbose 2>&1 | tee /tmp/arrayview-tunnel-window-b.log
```

Then run the A file once more from window A.

Expected:

- B opens only in window B; the second A request opens only in window A;
- ACK `windowId` values differ and match the registrations claimed by each
  extension host;
- all requests reuse the same compatible server instance;
- no broadcast request is claimed by the unfocused/wrong window.

Save both ACKs and the relevant extension-log section.

## Scenario 3: simultaneous launch convergence

Run these from two tunnel terminals at nearly the same time:

```bash
uv run arrayview /tmp/arrayview-tunnel-a.npy --window vscode --port 8000 --verbose > /tmp/arrayview-concurrent-a.log 2>&1 &
uv run arrayview /tmp/arrayview-tunnel-b.npy --window vscode --port 8000 --verbose > /tmp/arrayview-concurrent-b.log 2>&1 &
wait
uv run arrayview instances --json > /tmp/arrayview-instances-concurrent.json
```

Expected: one compatible server instance, two valid sessions/viewers, no bind
error, and two terminal `backend_ready` ACKs.

## Scenario 4: extension-host reload and tunnel reconnect

1. Leave the backend running.
2. Run `Developer: Restart Extension Host` in window A.
3. Launch the A file again and capture its ACK.
4. Disconnect and reconnect window A to the same tunnel.
5. Launch once more.

Expected: stale registrations/signals are reconciled, a fresh exact-window ACK
reaches `backend_ready`, and neither step requires `arrayview --kill`, manual
registry deletion, or a full VS Code window reload.

## Scenario 5: owned crash recovery

Obtain the owned instance PID:

```bash
uv run arrayview instances --json > /tmp/arrayview-before-crash.json
```

Terminate only the PID recorded in that file with `kill -9`, then relaunch:

```bash
uv run arrayview /tmp/arrayview-tunnel-a.npy --window vscode --port 8000 --verbose 2>&1 | tee /tmp/arrayview-after-crash.log
```

Expected: the stale registry record is removed automatically, a new instance ID
is registered, and the viewer reaches `backend_ready` without `--kill`.

## Scenario 6: foreign port safety

First stop all verified ArrayView instances:

```bash
uv run arrayview stop --all
uv run python -m http.server 8000 --bind localhost >/tmp/foreign-8000.log 2>&1 &
FOREIGN_PID=$!
```

Attempt the tunnel launch. Expected: ArrayView reports a foreign/fixed-port
conflict and does not signal the foreign PID. Verify it remains alive:

```bash
kill -0 "$FOREIGN_PID"
kill "$FOREIGN_PID"
```

## Scenario 7: forwarding and privacy evidence

In the VS Code Ports view and ACK/log evidence, record:

- the forwarded port is exactly 8000;
- the external URI is not `localhost`;
- whether privacy became Public automatically or required a manual action;
- whether a private/auth redirect appeared;
- HTTP and WebSocket viewer traffic both remain functional;
- `visibility_verified` is emitted only after strict ArrayView `/ping` identity
  validation through the final resolved URL.

If privacy requires a manual action, mark this scenario `BLOCKED`; do not call it
passed merely because the panel eventually opened.

## Safe cleanup

```bash
uv run arrayview instances --json
uv run arrayview stop --all
```

Do not use `lsof | kill`, `netstat | taskkill`, or kill by port. Remove leftover
test arrays only after evidence is captured:

```bash
rm -f /tmp/arrayview-tunnel-a.npy /tmp/arrayview-tunnel-b.npy
```

## Results

| Scenario | PASS / FAIL / BLOCKED | Evidence path | Notes |
|---|---|---|---|
| 1. Single launch + ACK | PASS | `/tmp/arrayview-tunnel-single.log`; `/tmp/arrayview-ack-single.json`; `/tmp/arrayview-extension-single.log`; `/tmp/arrayview-instances-single.json` | Targeted window `7dd4587ba8529632`; terminal `backend_ready`; ACK server `28192487-ddd3-48d4-b42c-f26296c3e8cc` matched `/ping.instance_id`. A one-time extension-host reload was required to activate bundled extension 0.14.37. |
| 2. Two-window exact routing | PASS | `/tmp/arrayview-window-current.log`; `/tmp/arrayview-window-other.log`; `~/.arrayview/extension.log` | Simultaneous targeted requests reached distinct tunnel windows `0557f0356fa0b1f8` and `47d8ad95a5ba4531`. Both returned `backend_ready` and reused server `9aef6d26-258e-4404-87fb-b5cfe521bca6`; neither was broadcast or claimed by the other window. |
| 3. Concurrent convergence | PASS | `/tmp/arrayview-concurrent-fixed-a.log`; `/tmp/arrayview-concurrent-fixed-b.log`; `/tmp/arrayview-instances-concurrent-fixed.json`; `~/.arrayview/extension.log` at 2026-07-11 11:01Z | The initial run reproduced a last-writer-wins failure. Extension 0.14.38 and Python signaling now use unique per-request queue files. Retest produced two exit-0 commands, two terminal `backend_ready` ACKs (`58ec41...` and `ec1564...`), two viewers, and one compatible backend. |
| 4. Reload + reconnect | BLOCKED | `~/.arrayview/extension.log` at 2026-07-11 09:56-09:58Z | Extension-host restart/window reload activated 0.14.37 and the backend survived, but a tunnel disconnect/reconnect was not performed. A terminal inherited across reload retained stale IPC/window state and fell back to broadcast; a newly targeted launch succeeded. |
| 5. Crash recovery | PASS | `/tmp/arrayview-before-crash.json`; `/tmp/arrayview-after-crash.log`; `/tmp/arrayview-after-crash-instances.json` | Registry-owned PID was killed. Stale instance `28192487-ddd3-48d4-b42c-f26296c3e8cc` was replaced automatically by `ab5134a2-55f1-4311-9d02-86b6a9d59783`; launch reached `backend_ready` without manual registry cleanup or `--kill`. |
| 6. Foreign port safety | PASS | `/tmp/arrayview-foreign-attempt.log`; `/tmp/foreign-8000.log` | Launch exited 1 with a foreign-port conflict; the foreign HTTP PID remained alive and the ArrayView registry stayed empty. |
| 7. Forwarding/privacy | PASS | `/tmp/arrayview-ack-single.json`; `/tmp/arrayview-extension-single.log` | Port 8000 resolved to `https://v54z0psh-8000.euw.devtunnels.ms`; privacy changed to public automatically; no auth redirect appeared; strict identity validation emitted `visibility_verified` then `backend_ready`; `/ping` reported active viewer WebSockets. |

### Tunnel run facts (2026-07-11)

- Branch/commit: `plan/robust-launch-lifecycle` at `530573e`.
- Host: Linux `roodnoot`; Python 3.12.12; ArrayView 0.29.2.
- Bundled/active extension after the concurrency fix and reload: 0.14.38.
- Cleanup after scenario 6: no registered ArrayView instances and the foreign
  test server was terminated. VS Code may retain the forwarded port entry after
  backend shutdown/reload; that UI entry is client-owned forwarding state.

## Questions the tunnel agent must answer

1. Does every successful command correspond to a terminal `backend_ready` ACK
   with matching request/window/server identity?
2. Does each request open in its originating window when two tunnel windows are
   active?
3. Does the startup lock converge concurrent launches on one backend?
4. Do extension-host restart and tunnel reconnect recover without manual kill or
   VS Code window reload?
5. Is public/private forwarding achieved without an unsupported manual step or
   authentication redirect?
6. Are any ArrayView processes, registry entries, signals, or forwarded ports
   left behind after identity-safe cleanup?

Return this file with the result table filled in and link every captured log or
ACK artifact. Any `FAIL` or `BLOCKED` row remains a release blocker.
