# VS Code tunnel launch — debug log

Plan: `plans/tunnel/PLAN.md` · Requirements: `plans/tunnel/REQUIREMENTS.md`

## Session started: 2026-07-25

### State at session start

- Working paths: none verified in this session. `dev/launch-execution-ledger.md`
  records a 2026-07-22 tunnel pass at opener 0.14.70; the current build is
  0.14.82 and that evidence is not carried forward.
- Broken paths: VS Code tunnel, both entry points. Intermittent — succeeds on
  early attempts and stops on later ones.
- Last known-good commit: unknown. `dev/launch-execution-ledger.md` 2026-07-22
  is the last recorded real-tunnel pass.
- Extension at session start: 0.14.80 bundled; windows live on 0.14.80 and
  0.14.76 simultaneously.

### Observations from the pre-session logs (2026-07-25 19:34–19:42)

Read from `~/.arrayview/archive/extension-20260725-2200.log`.

| Time | Window | Event |
|------|--------|-------|
| 19:36:36 | `d428d121` | ACK failed — "Viewer panel closed before its first frame rendered" |
| 19:40:48.120 | `1bdd1ffc` | ACK failed — "Viewer did not render a frame before timeout" (`51fef3af`) |
| 19:40:48.820 | `1bdd1ffc` | `LOCK: isProcessingSignal=true` (`45e5a0bb`) |
| 19:40:49–19:42:17 | `1bdd1ffc` | `SKIP: isProcessingSignal=true`, once per second, ~90 s |
| 19:42:18.068 | `1bdd1ffc` | `SIGNAL: expired ageMs=240014 maxAgeMs=240000` → lock released only by expiry |
| 19:42:18.854 | `1bdd1ffc` | `4bbaa131` expired unserved at `ageMs=215046` — queued behind the wedge |

Throughout, window `910d7fbf` (running 0.14.76) logged `CLEANUP: retained
active claim ...` against window `1bdd1ffc`'s claim files every 5 s.

---

## Attempt 0: baseline reset and development loop

**Date**: 2026-07-25
**Hypothesis**: none — this attempt fixes the ability to test, not the bug.
**Change**:
- `vscode-extension/extension.js`: added `_arrayviewPackageArgs()`, so the uv
  candidate honours a configured package spec and uses `--with-editable` for an
  absolute path. Command shape otherwise unchanged.
- `vscode-extension/extension.js`: added `_reportExtensionVersionSkew()`, called
  after window registration. Scans peer `window-*.json` for live pids on a
  different `extensionVersion`, logs `SKEW:`, and shows a warning with a
  "Reload Other Windows" action. Addresses R11.
- `vscode-extension/package.json`: declared `arrayview.packageSpec`
  (`machine-overridable`); version 0.14.80 → 0.14.82.
- `src/arrayview/_vscode_extension.py`: `_VSCODE_EXT_VERSION` → 0.14.82.
- `~/.vscode-server/data/Machine/settings.json`: `arrayview.packageSpec` set to
  the repo path.
- Environment cleanup as recorded under "Baseline established" in the plan.

**Regression risk**: `_arrayviewPackageArgs()` defaults to `--with arrayview`,
so released behaviour is byte-identical when the setting is unset. The skew
check is read-only over the signal directory and cannot claim or delete. Neither
touches port selection, URL construction, environment detection, or the request
lifecycle.

**Test procedure**:
1. Reload every open VS Code window (`Developer: Reload Window`).
2. In the securo tunnel window, click
   `arrayview-tests/small_64.npy` in the Explorer.

**What success looks like**: the 64×64 array is visible in a tab.
**What failure looks like**: no tab, an empty tab, or a spinner that never ends.

**Result**: **FAILED** (2026-07-25). No array displayed. The development loop
itself worked: `PYTHON: using local checkout /localscratch/oheide/projects/arrayview`
and `--with-editable` in the launch line confirm Explorer clicks now run the
working tree. Evidence:
`~/.arrayview/archive/extension-20260725-2230-attempt0.log`.

This run produced a complete, uncontaminated trace and root-caused both H1 and
H2. Sequence for request `0dde3d47` (`prescan_images.npy`):

| Time (UTC) | Event |
|---|---|
| 20:23:36.697 | `LOCK` |
| 20:23:40.886 | `REMOTE: desktop integrated-browser proxy uses backend URL directly` |
| 20:23:40.929 | ACK `panel_opened` |
| 20:23:41.517 | `panel-phase transport-warmup-failed` |
| 20:23:41.6 → 20:23:59.8 | `panel-phase iframe-loaded` every ~1.5 s, 12 times, then silence |
| 20:23:57 / 20:24:11 | two `small_64.npy` requests queued, both `SKIP`ped |
| 20:26:46.529 | `ERROR: Viewer did not render a frame before timeout` |
| 20:26:46.555 | `UNLOCK` — 190 s after the lock |
| 20:27:57 | both queued requests expired unserved |

Backend was healthy throughout: `/ping` ok, the session page served HTTP 200 /
1.89 MB locally, and both sessions were registered. The decisive counter was
`viewer_connections_seen: 0` — the viewer never made a single HTTP request to
the backend.

**H2 root cause (confirmed).** `resolveRemoteViewerUrl` short-circuited on
`workbench.browser.enableRemoteProxy`, returning the raw
`http://localhost:8000/...` URL. That shortcut assumed VS Code's integrated
browser would proxy the request through the remote. Commit `8bde7fe` moved
tunnel viewers from Simple Browser to webview panels; `simpleBrowser` no longer
appears anywhere in `extension.js`. A webview panel's iframe runs on the
desktop, so the loopback URL resolved against the Mac's own port 8000 and
reached nothing. The shortcut's own `arrayViewStatusOk` guard could never catch
this: it runs in the extension host on the remote, where loopback always
succeeds — it validated the wrong side of the connection.

**H1 root cause (confirmed).** `viewerTimeoutMs` was
`Math.max(1, remainingSignalMs)` — the whole remaining signal lifetime. A
launch that can never render holds the queue lock for ~190 s, and requests
queued behind it exceed their own `maxAgeMs` and die unserved.

---

## Attempt 1: stop handing webview panels a loopback URL (H2)

**Date**: 2026-07-25
**Hypothesis**: the tunnel viewer never displays because the webview iframe,
which runs on the desktop, is given a URL that only resolves on the remote.
**Change**: `vscode-extension/extension.js` — removed the
`enableRemoteProxy` short-circuit from `resolveRemoteViewerUrl`, and removed the
matching branch that accepted a loopback answer from `asExternalUri`. Tunnel
resolution now always goes through cached route → external URI promotion.
**Regression risk**: confined to `remoteName === 'tunnel'`. Non-tunnel and
Remote-SSH resolution are untouched; local VS Code never enters this function.
The removed branches were reachable only when the setting was enabled, which is
precisely the failing configuration.
**Test coverage**: `test_tunnel_desktop_loopback.js` asserted the removed
behaviour and was rewritten to assert the corrected contract — a desktop tunnel
promotes to a public URL regardless of `enableRemoteProxy`.

## Attempt 2: bound the viewer readiness deadline (H1)

**Date**: 2026-07-25
**Hypothesis**: a failed launch starves later requests because it may hold the
queue lock for the entire signal lifetime.
**Change**: `vscode-extension/extension.js` — added
`VIEWER_READY_TIMEOUT_MS = 45000` and capped `viewerTimeoutMs` with it. All the
slow work (reading the array off its storage tier) completes before the backend
publishes its URL, so first frame is a seconds-scale operation; this does not
conflict with R8.
**Regression risk**: shortens a timeout only. Cannot prevent a launch that would
otherwise have succeeded within 45 s of `port_resolved`. If a real environment
ever needs longer, the constant is the single place to change.

**Attempts 1 and 2 shipped together** in opener 0.14.83. They are separable in
the log: Attempt 1 shows as a non-loopback `REMOTE:` resolution followed by
`script-loaded`/`ws-open`/`frame-rendered`; Attempt 2 shows as `UNLOCK` within
~45 s rather than ~190 s on failure. Bundling was chosen because Attempt 2 alone
cannot display an array and Attempt 1 alone leaves every failure costing three
minutes.

**Test procedure**:
1. Reload every open VS Code window.
2. In the securo window, click `arrayview-tests/small_64.npy` in the Explorer.
3. Then click `arrayview-tests/vol_128.npy`.

**What success looks like**: both arrays visible, in their own tabs.
**What failure looks like**: no tab, empty tab, or an endless spinner — but now
giving up within ~45 s rather than ~3 minutes.

**Result**: **PARTIAL — both changes verified working; a new blocker surfaced.**
(2026-07-25, evidence `~/.arrayview/archive/extension-20260725-2255-attempt12.log`.)
The tab still showed "Opening small_64.npy in ArrayView…" indefinitely, but for a
different and now-visible reason.

- Attempt 1 verified: resolution no longer returns loopback. The log shows
  `REMOTE: asExternalUri(http://localhost:8000/) attempt=1` followed by a real
  promotion attempt, where 0.14.82 would have returned the loopback URL.
- Attempt 2 verified: `LOCK` 20:54:12.165 → `UNLOCK` 20:54:15.925. Under four
  seconds, against 190 s previously, with a clean terminal message
  (`VS Code viewer failed to become ready: Failed to resolve remote viewer URL`).
  R7 and R10 now hold on this path.

The exposed blocker:

```
PORT: privacy retry returned "Could not forward port:
  HTTP status 429 Too Many Requests ...
  {"title":"Resource limit exceeded.","status":429,
   "detail":"...would exceed the limit for 'PortsPerTunnel'"}
```

## Attempt 3: free the tunnel's port quota (R1)

**Date**: 2026-07-25
**Hypothesis**: port 8000 cannot be promoted to public because the tunnel is at
its ports-per-tunnel quota, so no viewer can ever load regardless of ArrayView's
behaviour.

**Evidence**: the tunnel service journal
(`journalctl --user -u code-tunnel`) showed VS Code forwarding ports 22, 25, 80,
1716, 17077, 34619 and dozens of ephemeral ports — SSH, SMTP, HTTP and other
system daemons on a shared multi-user host, several belonging to other users.
`remote.autoForwardPorts` defaults to on and scans listening processes, so every
window reconnect re-forwarded the machine's entire listening set. This is
environmental, not an ArrayView defect.

**Change** (all outside the repository):
- `~/.vscode-server/data/Machine/settings.json`: `remote.autoForwardPorts:
  false`. ArrayView requests port 8000 explicitly via `asExternalUri`, which is
  unaffected by this setting.
- Same file: `remote.portsAttributes` trimmed from 11 entries to `8000`, `8001`,
  `8002`. The dropped entries (`8123`, `8124`, `38263`, `42247`, `42771`,
  `43129`, `46691`, `50925`) were leftovers from ArrayView runs on scanned
  ports.
- `code-tunnel --cli-data-dir ~/.vscode/cli tunnel restart` on tunnel `roodnoot`
  / `quick-horse-ztth1dg`, with the maintainer's approval, to clear port
  registrations held on the relay. The CLI has no port-management subcommand and
  `~/.vscode/cli/token.json` is an opaque credential, not an API bearer, so
  there is no narrower way to release them.

**Verification**: tunnel reports `"tunnel":"Connected"` at 21:04:39Z and
`journalctl` records zero `Forwarding port` events after the restart.

**Regression risk**: none in the repository — no code changed. Environmental
effect: VS Code no longer offers to auto-forward dev servers started in a
terminal; they can be forwarded from the Ports view. Reverting is a one-line
settings edit.

**Follow-up to consider once display works**: ArrayView should detect a
ports-quota failure and report it as a distinct, actionable message rather than
the generic `Failed to resolve remote viewer URL`. R10 is only partly satisfied
while a quota error reads as a resolution error.

**Test procedure**:
1. Reload every VS Code window (they reconnect after the tunnel restart).
2. In the securo window, click `arrayview-tests/small_64.npy`.
3. Then click `arrayview-tests/vol_128.npy`.

**Result**: **PASS** (2026-07-25). Arrays displayed in tunnel tabs for the first
time this session. Port promotion worked:
`PORT: changed privacy to public via command` →
`https://v54z0psh-8000.euw.devtunnels.ms`, then the full viewer chain
`script-loaded` → `ws-open` → `metadata-loaded` → `frame-rendered` →
`backend_ready`. R1 satisfied. Cold launch ~24 s including port promotion;
subsequent launch reused the cached route and reached first frame in 4.4 s.
The maintainer opened many further arrays successfully, then hit a hang.
Evidence: `~/.arrayview/archive/extension-20260725-2330-attempt3.log`.

Acceptance rows now passing: T1, T2, T3, T4, T5 (Explorer path, repeated).

---

## Attempt 4: report load failures instead of hanging on them

**Date**: 2026-07-25
**Hypothesis**: the reported hang is a file that cannot be loaded at all, and a
failed background load is indistinguishable from a slow one, so the opener waits
out its full deadline instead of reporting the error.

**Evidence**: the hang was `recons/SEC_014/rank2/initial_PD_block_svd.npy`.
Loading it directly raises immediately:

```
ValueError: This file contains pickled (object) data. If you trust the file you
can load it unsafely using the `allow_pickle=` keyword argument
```

`_load_in_background` in `_routes_loading.py` caught the exception, printed a
traceback to the daemon's stderr — which nothing surfaces — and returned. Its
comment stated the intent directly: *"The opener will fail closed when metadata
never becomes ready."* So `/metadata/<sid>` stayed 404 forever and the extension
waited its full 150 s, holding the queue lock, before reporting the misleading
`Viewer session expired before a panel could be opened; retrying the command
will create a fresh session`. Retrying cannot help — the file will never load.

It then happened twice in a row. `_fastLoadViaDaemon` gives up after 12 s and
falls back to `_spawnPythonForFile`, which loaded the same file a second time
and queued a second request behind the first: two consecutive 150 s waits from
one click. The 12 s fallback deadline and the 150 s readiness wait describe the
same operation and disagreed by more than tenfold.

**Change**:
- `src/arrayview/_session.py`: added `FAILED_PENDING_SESSIONS: dict[str, str]`
  (global state stays in this module per the project's non-negotiables) and
  exported it.
- `src/arrayview/_routes_loading.py`: the background loader records
  `str(exc)` against the sid, and a fresh load for the same sid clears any
  previous entry.
- `src/arrayview/_routes_query.py`: `/metadata/<sid>` returns **422** with the
  reason as the body when the sid's load failed, instead of 404 forever.
- `vscode-extension/extension.js`: added `httpSessionProbe` /
  `waitForSessionReady`, which treat 422 as terminal and carry the message.
  The wait now throws `ArrayView could not open this file: <reason>`.
- `vscode-extension/extension.js`: `_fastLoadViaDaemon` no longer falls back to
  spawning Python when the ACK reached `claimed` — a claim means a window owns
  the request and is still working, so the fallback only duplicated the load.

**Regression risk**: 422 is a new status on one route; the previous 404 path for
genuinely-pending and unknown sids is unchanged, and good files still return 200
(verified). The fast-load change narrows when a duplicate process is spawned; it
cannot suppress the fallback when no window claimed the request, which is the
case the fallback exists for.

**Verification** (real server, the maintainer's actual file):

| Probe | Before | After |
|---|---|---|
| `/metadata/<sid>` for the pickled `.npy` | 404 forever, 150 s wait, twice | **422 within ~2 s** with `This file contains pickled (object) data…` |
| `/metadata/<sid>` for `vol_128.npy` | 200 | 200 (unchanged) |

Ten of eleven Node tests pass; `test_integrated_browser_placeholder_cleanup.js`
is the known pre-existing failure removed from CI in `d4c4ad3`.
`tests/test_loading_server.py` passes; `tests/test_api.py` has 4 failures that
reproduce identically with these changes stashed (thumbnail aspect ratio, RGB
thumbnails, stale window registration, preferences schema) — pre-existing and
unrelated.

**Known remaining gap (not addressed here)**: the readiness wait still holds the
global `isProcessingSignal` lock, so one genuinely slow array still blocks other
requests for the duration. R8's "a slow load must not block anything else" is
not yet satisfied; only the failure case is now fast. Fixing it means making the
lock per-request rather than global — a larger change, deliberately deferred so
this round stays testable.

**Test procedure**:
1. Reload every VS Code window (extension is now 0.14.84).
2. Click `recons/SEC_014/rank2/initial_PD_block_svd.npy` — the file that hung.
3. Then click any array that worked before, e.g. `arrayview-tests/vol_128.npy`.

**What success looks like**: step 2 fails within a few seconds with a message
naming pickled object data; step 3 then opens normally.
**What failure looks like**: step 2 hangs again, or step 3 is blocked by step 2.

**Result**: **PARTIAL** (2026-07-25). The block is gone — the failing file no
longer holds up anything behind it, and the duplicate-loader spawn stopped. But
the tab still showed only `Opening initial_PD_block_svd.npy in ArrayView...`;
the reason never reached the UI. The 422 and its message existed but had nowhere
to go: the placeholder's error path only fires when `launchArrayViewFile` itself
throws, and this failure happens later, asynchronously, inside signal
processing, which held no reference to the placeholder.

---

## Attempt 5: durable fixes — surface failures, name the quota error, stop port sprawl

**Date**: 2026-07-25
**Hypothesis**: three defects keep failures invisible or environment-specific.
The maintainer's machine works only because of a hand-edited
`remote.autoForwardPorts: false`, which ArrayView does not ship and must not
write on a user's behalf. What ArrayView can own is: reporting failures where
the user looks, naming the quota failure precisely, and not contributing to port
sprawl itself.

**Change**:

*Failures reach the user (R10)*
- `extension.js`: added `_reportFailureToPlaceholder(data, message)`, correlated
  to the waiting tab by the request's `handoffPath`. Called from
  `processSignalData`'s catch and from the URL-resolution failure path, it
  replaces "Opening …" with the actual reason. When no placeholder is waiting —
  a terminal launch, or the tab was closed — it falls back to
  `showErrorMessage`. Message text is HTML-escaped.

*The quota error says what is wrong (R1, R10)*
- `extension.js`: added `_forwardingDiagnostic(result, port)`, which recognises
  `PortsPerTunnel` / `Resource limit exceeded` / `429` in VS Code's forwarding
  result and produces: *"this tunnel has reached its limit on forwarded ports,
  so the port cannot be made public and the viewer cannot connect. Close
  forwarded ports you no longer need in the Ports view, or set
  `remote.autoForwardPorts: false` …"*. A generic `Could not forward port` is
  passed through with its own text. Recorded in `_lastForwardingDiagnostic`,
  cleared at the start of each resolution, and used instead of the opaque
  `Failed to resolve remote viewer URL`.

*ArrayView stops contributing to port sprawl (R1)*
- `_vscode_extension.py`: added `_port_has_listener(port)` and
  `_stale_arrayview_port_keys(attrs, keep_port)`.
  `_configure_vscode_port_preview` now prunes its own dead entries while writing
  the current one. Previously every launch on a fresh port left its
  `remote.portsAttributes` entry behind permanently — the maintainer's machine
  had accumulated eleven, eight of them dead, each still instructing VS Code to
  forward a port nothing serves.

**Deliberately not done**: ArrayView does not write `remote.autoForwardPorts`.
That is a user's policy for their machine, and the auto-forwarded ports that
exhausted this tunnel (22, 25, 80, 1716, …) were system daemons ArrayView has no
claim over. The honest split is that ArrayView cleans up after itself and
explains the quota failure; deciding the auto-forward policy stays with the
user.

**Regression risk and how it was checked**:
- Pruning only considers entries whose `label` is `ArrayView`, never the port
  being configured, and never a port with a live listener. Verified directly:
  entries for other tools, non-dict values, unparseable keys, and a **live
  concurrent ArrayView server on 8055** were all preserved, while only the dead
  entry was pruned.
- `_reportFailureToPlaceholder` is read-only with respect to launch flow and
  returns false when no placeholder matches; the notification fallback cannot
  fire when a tab was updated.
- `_forwardingDiagnostic` only changes message text; resolution logic is
  untouched.

**Test results**: 10/11 Node tests pass (`placeholder_cleanup` is the known
pre-existing failure from `d4c4ad3`). Python: 352 passed, 4 failed — the same
four that fail with all changes stashed. One run additionally showed
`test_cli_vscode_terminal_requires_extension_readiness_ack` timing out; that run
was polluted by a `_serve_empty` test server left listening on 8011 (37.7 s vs
11.4 s for the clean run). It passes in isolation and `tests/test_cli.py` passed
19/19 on two consecutive clean runs. Not a regression.

**Test procedure**:
1. Reload every VS Code window (extension 0.14.85). This also drops the viewer
   sockets on the daemon still running from the previous round, so the next
   click starts a backend carrying the new Python code.
2. Click `recons/SEC_014/rank2/initial_PD_block_svd.npy`.
3. Then click `arrayview-tests/vol_128.npy`.

**What success looks like**: step 2's tab shows "ArrayView could not open
initial_PD_block_svd.npy" with the pickled-object-data reason, within seconds;
step 3 opens normally.

**Result**: **FAILED** (2026-07-26). The tab still showed only `Opening
initial_PD_block_svd.npy in ArrayView...`. Attempt 4's backend fix covered the
wrong code path.

**Analysis of the miss**: the log shows `FASTLOAD: no daemon on port 8000`,
so the launch went down the **cold-start** path — `python -m arrayview <file>
--window vscode`, where `_serve_daemon` loads the file itself at startup. The
`/load` endpoint in `_routes_loading.py`, which Attempt 4 fixed, is only used
when a daemon is *already* running. Two independent code paths create a pending
session, and only one of them recorded failures. The reason the maintainer saw
this on a fresh window is that the first click of a session is always a cold
start.

---

## Attempt 6: record load failures on the cold-start daemon path too

**Date**: 2026-07-26
**Hypothesis**: the cold-start daemon loses its load error, so no path exists to
turn it into a 422 and the opener waits out its deadline exactly as before.

**Evidence**: `_load()` inside `_serve_daemon` (`_launcher.py`) catches the
exception, emits a `session.load_failed` trace event, and re-raises — on a
daemon thread, where re-raising only kills the thread and writes to a stderr
nothing surfaces. `PENDING_SESSIONS.discard(sid)` then runs in `finally`, so
`/metadata/<sid>` falls through to a plain 404 forever.

**Change**: `src/arrayview/_launcher.py` — the handler now records
`_session_mod.FAILED_PENDING_SESSIONS[sid]` and emits a `_vprint` line naming
the file and reason, before re-raising as before.

**Completeness check** (the step missed in Attempt 4): every site that calls
`PENDING_SESSIONS.add` was enumerated. There are exactly two —
`_routes_loading.py:268` and `_launcher.py:4172` — and both now record failures.

**Regression risk**: the handler still re-raises, so existing behaviour and the
`session.load_failed` trace are unchanged; the only additions are a dict entry
and a log line. No change to the success path.

**Verification** (real daemon, the maintainer's file, on the exact cold-start
entry point `_serve_daemon(...)`):

| Probe | Before | After |
|---|---|---|
| `/metadata/<sid>`, pickled `.npy` | 404 forever | **422 at t=1 s** with `This file contains pickled (object) data…` |
| `/metadata/<sid>`, `vol_128.npy` | 200 | 200 (unchanged) |

Python: 377 passed, 4 failed — the same four pre-existing `test_api.py` failures
(thumbnail aspect ratio, RGB thumbnails, stale window registration, preferences
schema), now also covering `test_lifecycle_contract.py` and
`test_launch_blackbox.py`.

**Test procedure**:
1. Reload every VS Code window (extension 0.14.86).
2. Click `recons/SEC_014/rank2/initial_PD_block_svd.npy`.
3. Then click `arrayview-tests/vol_128.npy`.

**What success looks like**: step 2's tab is replaced within seconds by
"ArrayView could not open initial_PD_block_svd.npy" and the pickled-object-data
reason. Step 3 opens normally.

**Result**: **PASS** (2026-07-26), confirmed by the maintainer and by the log.

Step 2 — the unloadable file:

```
22:19:00.239  CUSTOM-EDITOR: resolveCustomEditor for initial_PD_block_svd.npy
22:19:00.265  FASTLOAD: no daemon on port 8000        ← genuine cold start
22:19:03.571  ERROR: ArrayView could not open this file: This file contains
              pickled (object) data. …
22:19:03.601  CUSTOM-EDITOR: reported failure in placeholder
22:19:03.603  UNLOCK: isProcessingSignal=false
```

3.3 s from click to a named reason in the tab, against 150 s of silence before,
and the queue lock released 2 ms after the report.

Step 3 — `parameter_maps.nii` immediately afterwards: `ws-open` →
`metadata-loaded` → `frame-rendered` → `backend_ready` at 22:19:29. Not blocked
by the preceding failure.

**Correction**: an earlier version of this entry claimed the failed cold-start
daemon exited on its own, inferred from the next click logging `FASTLOAD: no
daemon on port 8000`. The maintainer had killed port 8000 by hand between the
two clicks. The inference was wrong and the opposite is true — see the defect
below.

### Defect found: a failed cold-start daemon holds port 8000 forever

`_serve_daemon(..., persist=True)` starts uvicorn before loading, and the load
runs on a separate thread. When that load raises, the server keeps listening
with no session and nothing ever shuts it down. This also reproduced during
Attempt 6's verification: the daemon started on port 8021 with the unloadable
file had to be killed manually.

This violates the project's standing rule that shutdown must be automatic and no
orphan processes are left behind. It is worse than a stray process: the orphan
answers `/ping`, so the next launch takes the fast-load path against a daemon
that holds no session and never will.

---

## Attempt 7: release the port when the initial load fails

**Date**: 2026-07-26
**Hypothesis**: a daemon whose only reason to exist failed keeps the port and
answers `/ping` until `_PERSIST_DAEMON_CONNECT_TIMEOUT_SECONDS` (210 s) expires.

**Evidence**: measured directly rather than inferred this time. A daemon started
on port 8031 with the unloadable file was still listening 40 s after its load
raised, reporting `sessions: 0 viewers: 0 owner: persistent`. It is bounded, not
permanent — the 210 s connect timeout eventually reaps it — but nothing should
hold a port for three and a half minutes after failing in the first second.

**Change**: `src/arrayview/_launcher.py`
- `_initial_load_failed = threading.Event()`, set by `_load`'s handler.
- New `_FAILED_LOAD_EXIT_GRACE_SECONDS` (default 30, env-overridable via
  `ARRAYVIEW_FAILED_LOAD_EXIT_GRACE_SECONDS`).
- A watcher thread waits on the event, sleeps the grace period so the opener can
  still read the 422 from `/metadata/<sid>`, then — **only if `SESSIONS` and
  `VIEWER_SOCKETS` are both empty** — unregisters and exits.

The grace period exists precisely so Attempt 6's fix keeps working: releasing
the port immediately would replace the 422 with a connection refusal and lose
the reason again.

**Regression risk**: the watcher blocks on an event that is only set when the
initial load raises, so a successful daemon never reaches the exit path. The
`SESSIONS`/`VIEWER_SOCKETS` guard additionally prevents exiting if anything
else began using the server during the grace window.

**Verification**:

| Case | Result |
|---|---|
| Unloadable file — 422 still readable during grace | 422 at t=1, 2, 3 s |
| Unloadable file — port released after grace | port 8041 free, daemon gone |
| **Good file — daemon must not exit** | still serving after 25 s (5× the grace), `/metadata` 200, no viewer ever connected |

Python suites: 396 passed. Five failures, all confirmed environmental or
pre-existing: the four long-standing `test_api.py` failures, plus
`test_lifecycle_contract.py::test_remote_vscode_spawned_daemon_keeps_backend_persistent`,
which needs port 8000 and fails identically with all changes stashed because the
maintainer's live daemon (three open arrays) holds it.

**Note**: no extension rebuild — `extension.js` is unchanged this round and the
bundled VSIX matches it byte-for-byte. No window reload is required; daemons are
spawned per cold start and pick up the working tree via `--with-editable`.

**Test procedure**: click `initial_PD_block_svd.npy` on a cold start, confirm
the error still appears, then check that port 8000 is free about half a minute
later instead of staying occupied.

**Result**: (pending — awaiting user test)

---

## Attempt 8: show the viewer while the array is still loading

**Date**: 2026-07-26
**Trigger**: maintainer reported ~10 s of a blank tab on a cold start, and
several seconds on a terminal launch, before the animating logo appears.
Committed `ae7b19c` first at their request so there was a known-good fallback.

**Measurement before changing anything** (cold start, `parameter_maps.nii`):

| Phase | Time |
|---|---|
| spawn `uv` → process running | 1.25 s |
| Python boot → server listening | 1.02 s |
| wait for session (array loading off `/smb`) | 4.13 s |
| panel created → viewer booted | 0.55 s |
| first frame rendered | 4.72 s |
| **total** | **~11.7 s** |

The terminal launch showed the same shape: a constant 4.16 s between `LOCK` and
the first route check, entirely inside the session wait. That is genuine work —
`sense_images_denoised_SEC.npy` is 419 MB of complex64 and `np.load` alone takes
3.89 s. About 9 of the 11.7 s is real loading and rendering.

**The actual defect**: none of it was visible. `_serve_daemon` starts uvicorn
before loading on purpose — *"Start uvicorn immediately — the window can open
before data is ready"* — and the viewer has a cold-start loading spinner for
exactly this. The opener discarded that by waiting for `/metadata/<sid>` to
return 200 *before* creating the panel.

**Change**:
- `extension.js`: removed the pre-panel session wait. Order is now resolve URL →
  open panel → await session readiness (still 422-aware) → await first frame.
- `_viewer.html`: added `fetchMetadataWithRetry` / `showMetadataError`. The
  viewer previously treated any non-2xx metadata reply as fatal and printed
  "Session not found or expired" — precisely the race the pre-panel wait
  existed to prevent, so the reorder was impossible without this. It now uses
  the three states the backend already distinguishes: `404` + `Retry-After` =
  still loading, retry; bare `404` = unknown or released, stop; `422` = load
  failed, show the reason.
- `extension.js`: `waitForViewerReady`'s timeout became an **inactivity**
  budget, reset on each newly observed phase. With the panel opening before the
  load completes, a flat cap would fail a launch for being slow rather than
  stuck. A stalled launch still fails after 45 s of silence — strictly better
  than Attempt 2's flat cap.
- `extension.js`: added `PANEL_MIN_REMAINING_MS`. Opening a panel is a visible
  side effect and is now refused for a request with under a second of life left;
  the removed wait used to consume the remaining lifetime and enforce this
  implicitly.

**Regression caught by the tests, twice.** `test_request_deadline` asserts an
expired request must not create a panel. Removing the wait broke that — first
the panel opened anyway, then the terminal ACK was not recorded because the old
path reached expiry through `ensureActive`/`_expireProtocolRequest`. Both fixed;
the guard now fences a genuinely expired request and otherwise writes the failed
ACK directly.

**Verification**: against a real daemon with the 419 MB file — the page serves
200 within 0.5 s of start, `/metadata` returns `404 + Retry-After: 1` for ~4 s,
then 200. Unknown sid returns a bare 404 (viewer stops), failed load returns
422. All 10 Node tests pass.

**On the browser suite**: a combined four-suite run showed 40 failures against a
35 baseline, which looked like five regressions. It was not. Run alone the suite
gives 35 vs a 34 baseline, and the single differing test
(`TestKeyboard::test_R_registration_overlay_and_n_cycles_compare_target`) fails
3/3 both with and without the changes. `tests/test_browser.py` is flaky and
order-dependent with ~34 pre-existing failures, including a missing
`tests/snapshots/` directory. **Never compare raw counts on this suite — diff
failure sets by test name.**

**Expected effect**: logo/spinner at ~2.5 s instead of ~10 s. Total time to
first frame unchanged; this is feedback, not throughput.

**Result**: (pending — awaiting user test; needs a window reload to 0.14.87)

Acceptance rows now passing: T1–T6, T10. R7, R9, R10 hold on the Explorer path.

### Session summary — what was actually wrong

Five independent defects, each masking the next:

1. **Version skew.** Windows kept running whatever build they loaded at
   activation; two extension versions shared one signal directory and raced.
   Plus `~/.vscode/extensions/extensions.json` pointed at a directory renamed
   `…-0.14.70.broken`, the silent no-load failure from `plans/webview/LOG.md`.
2. **Loopback URL handed to a desktop webview.** `resolveRemoteViewerUrl` kept a
   shortcut written for the Simple Browser after tunnel delivery had moved to
   webview panels; its own health check ran on the remote, so it could never
   detect the problem. Nothing ever reached the backend
   (`viewer_connections_seen: 0`).
3. **A failed launch held the queue for 190 s**, so every later click was
   dropped — the "works the first time, not the third" symptom, and the reason
   all earlier diagnosis was contaminated.
4. **Tunnel port quota exhausted.** VS Code auto-forwarded every listening port
   on a shared multi-user host (22, 25, 80, 1716, …), so port 8000 could not be
   promoted to public and the WebSocket could never connect.
5. **Load failures were invisible by design**, on two separate code paths, and
   the reason had no route to the UI.

Only #2, #3 and #5 were ArrayView defects. #1 and #4 were environmental, and are
now either detected (skew warning) or explained (quota diagnostic).

---

## 2026-07-27 — 0.14.91: stale-request starvation and self-inflicted session kill

**Input**: "it still hangs — inspect the logs, zoom out, fix". Not a single-bug
report, so the starting point was the whole log corpus rather than one incident:
`~/.arrayview/extension.log` plus 7 archives, 34,167 lines.

**Method note on statistics.** The naive aggregate reads 491 claimed / 300 ready
/ 181 failed (37%), and surfaces bugs like `pingUrl is not defined` that were
fixed on 2026-07-08. The archive reaches back far enough to pollute any total.
Scoped to the current architecture (2026-07-26 onward, v0.14.87–90) the real
numbers are **31 claimed / 22 ready / 8 failed (26%)**. Always scope first.

Per-request timing (queue wait = WATCH rename → ACK claimed) is the measurement
that made the failure class visible: healthy requests spend 0.4–6 s, while two
requests burned 87.9 s and 75.6 s and *then* failed, and one live request waited
**68.7 s in queue and rendered in 1.3 s** once it finally ran.

**RC1 — stale detection deferred behind the slowest path.** 22:13:49→22:14:03:
cached-route probe returns no verdict three times (1.5 s / 4 s / 8 s), then all
six `asExternalUri` attempts time out through 22:15:17, then at 22:15:17.695 a
probe answers **in 111 ms** with `REMOTE: cached route stale`. The information
was available in milliseconds and was consulted 88 s late. Because the signal
loop serialises on `isProcessingSignal`, that dead wait is charged to every
later click — this is the mechanism behind "works once, then hangs".

Fix: `localBackendIdentity()`, a strict three-way loopback verdict consulted
before any remote work and again between retries. The asymmetry is the whole
design: abandoning is irreversible, so only positive proof of a foreign
`instance_id` may do it. **A not-yet-listening port must be indistinguishable
from success** — that is exactly the state the 22:13:49 request was in. This is
deliberately stricter than `probeArrayViewStatus`, which reports `ECONNREFUSED`
as `PROBE_DEAD` because that code is absent from `TRANSIENT_PROBE_ERRORS`;
reusing that verdict here would have broken every slow-loading large array.

**RC2 — the extension killed the session it was waiting on.** 12:28 cluster:
`placeholder disposed for sense_images.npy` (27.804) → `PANEL: release
sid=e48419c1 ok=true` (27.820) → `ERROR: Backend stopped answering before the
viewer was ready` (28.838). VS Code preview-tab reuse disposes the placeholder
when the next single-click replaces the tab; the dispose handler released the
session `waitForViewerReady` was still waiting on, and the resulting error
blamed the backend for the extension's own act. Release is now deferred to the
terminal path, and the truthful panel-closed error is checked before the ping.

**Regression I caused and fixed.** The loopback pre-check broke three passing
tunnel tests: they stub `https.get` only, so the new probe reached the real
arrayview on port 8000 (instance `b8ea8af9`), read LOCAL_FOREIGN, and correctly
abandoned. Test outcomes depending on what occupies a dev-machine port is a
defect in the test seam, not bad luck — all three now stub `http.get`.

**Evidence**: `component` only. All 13 extension tests pass, including new
`test_local_identity_shortcircuit.js`. The tunnel display boundary was **not**
validated on a real host — the user was asleep and CLAUDE.md forbids installing
into their profile or reloading their window without permission. RC2 has **no
automated coverage**: the only test that drove `_processSignalDataBody` for
placeholders is the integrated-browser test, now obsolete (extension.js pins
`useIntegratedBrowser` to false at the `if (false)` guard).

**Result**: (pending — needs install of 0.14.91 and a window reload)

### 0.14.92 — blank panel when the relay is saturated

**Report**: three Explorer clicks rendered fine, then `arrayview
initial_pd_UI.npy` from a terminal opened a tab with nothing in it.

**First hypothesis was wrong and worth recording.** The terminal path calls
`openInWebviewPanel` (new panel) while Explorer navigates an existing custom-
editor placeholder, so "the terminal path is broken" was the obvious read. It
does not survive the log: both paths build identical HTML from the same
`_viewerPanelHtml`, `_backendPortMapping` returns null for a devtunnels host so
neither gets a port mapping, and `openInWebviewPanel` succeeded twice on
2026-07-26 (iframe-loaded in ~400 ms). The path is not the variable.

**What the log actually shows**: `transport-warmup-complete` at 09:25:21.616,
then silence — no `iframe-loaded`, no error — until `ERROR: Viewer did not
render a frame before timeout` 45 s later. Meanwhile the backend served that
exact sid in **30 ms over loopback** (1.9 MB, HTTP 200). So `/ping` crossed the
relay fine and the page request did not.

The variable that *does* track the failure is how many viewers were already
streaming: the backend reported `active_viewer_sockets: 3` when the fourth load
stalled, and closing the tabs made the identical command succeed. Treat the
saturation mechanism as unconfirmed (n=1) — what is confirmed is that a request
through the relay can stall indefinitely while the backend stays healthy.

**Defect**: a stalled navigation fires neither `load` nor `error`, so the
wrapper's retry loop never saw it. `scheduleReload()` was armed only inside the
`load` handler — it covered "page arrived, viewer never booted" and structurally
could not cover "page never arrived". Fixed by arming at `frame.src` assignment
and re-arming on each retry, with a longer budget (8 s) than the boot watchdog
(1.5 s) because it must cover a real 1.9 MB transfer. Re-assigning `frame.src`
cancels the stuck request rather than adding another, so the retry frees the
connection instead of competing for one.

**Evidence**: `component` (new `test_panel_navigation_watchdog.js`, 14/14 green)
plus `real process` for the backend timing. The stall itself was reproduced by
the user, not by the suite — the relay is not drivable from this host.

**Still open**: the signal loop serialises on `isProcessingSignal` for the whole
request including the viewer wait, so any slow load blocks every later click.
The watchdog shrinks that window from 45 s to ~8 s but does not remove it.

### 2026-07-27 later — four clicks, four different causes

The user's summary was "still really really bad": one slow, one slower, one
that would not load, one that hung. The log separates them into four unrelated
things, which is the point worth recording — a single bad session is not a
single bug, and treating it as one is how earlier rounds went wrong.

| file | shape | outcome |
|---|---|---|
| `coil_sensitivities.npy` | (224,240,210,64), 5.8 GB | rendered, 11.6 s |
| `initial_pd_UI.npy` | (224,240,1,204), 87 MB | rendered, 4.8 s |
| `sampling_mask.npy` | — | **0 bytes on disk**; "No data left in file" was correct |
| `sense_image.npy` | (11289600,), 90 MB | hung to timeout |

**Not bugs**: the empty file (the user guessed this himself), and 5.8 GB
resolving in 11.6 s.

**The slowness is the relay, not the code.** For the first click the 1.9 MB
viewer page took 6.8 s to cross (~280 KB/s) and the /ping warmup 1.85 s; the
second took 3.0 s. That fixed cost is paid on every viewer open and is the
single biggest lever on perceived speed. Caching the page is untried.

**The hang was two defects stacked.** A 1-D array yields a 1-D slice, so the
colormapped result is `(N, 4)` and `h, w = rgba.shape[:2]` reads the 4 RGBA
channels as the image width — the header then advertises a 4-pixel-wide frame
and PIL raises `buffer is not large enough` on every frame. **Size was never
the variable**: a 10,000-element array fails identically, so the 90 MB and the
tunnel were both red herrings. `sense_image.npy` is 224·240·210 flattened.

The second defect is the one that generalises: the websocket handler logged the
exception to the server's stdout and closed the socket without telling the
viewer, so a hard backend failure presented as a silent hang and the only
artifact was in a console nobody was reading. Any render failure now sends
`render_error` and the viewer shows it.

**Evidence**: `real process` — reproduced in Chromium via
`tests/test_1d_arrays.py` before the fix (1-D never rendered, 2-D fine) and
passing after, including the reported 11,289,600 shape.

**Method note**: the first hypothesis was that the *large* 1-D array was too
big. Parametrising over size killed it in one run — 10,000 elements failed the
same way. Vary the suspected dimension before building on the theory.

### 0.14.93 — an undrawable array must not take the queue with it

Direct request: a wrong click should give a decent message and must not break
subsequent loads.

Sending `render_error` to the viewer (previous entry) fixed the message and not
the collateral damage. The opener holds `isProcessingSignal` for the whole
request, so a panel that never gets a frame still waited out its full timeout —
45 s in the observed incident — with every later click dead behind it. That is
the mechanism behind "it broke the next few too", and it is the same
head-of-line blocking that produced the original 68.7 s queue wait.

Chain now: viewer `reportParent('render-error')` → wrapper cancels its pending
reload and posts `viewer-failed` → `waitForViewerReady` finishes with the
backend's own message. Milliseconds instead of 45 s.

**The asymmetry is the design.** A stalled transfer (0.14.92) is worth
retrying because it may succeed; an undrawable array is not, because the retry
reproduces the failure exactly. Same-looking symptom, opposite correct
response — worth checking which one applies before adding any future retry.

**Still open**: the queue is still strictly serial. Both fixes shorten how long
a bad request holds it; neither removes the coupling, so one genuinely slow
load still blocks later clicks. Fixing that means letting the viewer wait
happen outside the lock, which changes request interleaving and deserves its
own change.

**Note**: 1-D arrays now render as a single row. The user does not need that
capability, but it is correct, tested, and strictly better than the error it
replaced, so it stays.

### 0.14.94 — the queue coupling itself

Three rounds of fixes each shortened how long a bad request held the signal
queue — 88 s of stale-route backoff, 45 s of stalled navigation, 45 s of an
undrawable array. None touched the reason any of that mattered: the lock
spanned the whole request, including the wait for the viewer's first frame.

That wait is a network wait. It is *supposed* to take tens of seconds on a
large array. Holding the queue across it means a perfectly healthy slow load
blocks every later click, and there is no bug to fix in that case — only a
design to change.

Split at `panel_opened`. Above it a request claims shared state: the route
cache, a pending placeholder, an entry in `_openPanels`. Below it a request
waits on its own panel and its own backend and shares nothing. The queue is
handed on at that line.

**The part worth remembering**: releasing early means a request can finish
while a *later* one holds the queue, so a plain `isProcessingSignal = false` in
a `finally` would release someone else's lock and put two requests in the
critical section at once — strictly worse than the original problem. The lock
now carries an owner ticket and releasing is a no-op unless the caller still
owns it. Any future early-release needs the same care.

**Evidence**: `component`. `test_signal_queue_handoff.js` asserts the queue is
free while a request awaits its first frame, that a second request opens its
own panel meanwhile, and that out-of-order settling never releases another
request's lock. Not yet exercised on a real host.

**Now genuinely open**: nothing bounds how many requests can be in their
readiness wait at once. User clicks are self-limiting, so this is untested
rather than known-bad.

### 0.14.94 follow-up — two install traps, both self-inflicted

The queue change was fine. The stuck "Opening … in ArrayView…" was a launch
failure that never wrote a signal at all, and both causes were mine.

**Trap 1: the bundled VSIX.** The Python package ships its own copy of the
opener at `src/arrayview/arrayview-opener.vsix`, and `_ensure_vscode_extension`
compares what is installed on disk against *that* copy — not against whatever
was most recently built. Four version bumps (0.14.91 → 0.14.94) were each
packaged into a scratch directory and installed with `code
--install-extension`, so the extension host ran the new build while Python kept
comparing against a bundled 0.14.90. Result: `_VSCODE_EXT_RELOAD_REQUIRED`, and
the message "reload this exact window once, then retry" — advice no reload
could satisfy, because the stale half was on disk.

`vscode-extension/AGENTS.md` states the rebuild target
(`vsce package -o ../src/arrayview/arrayview-opener.vsix`),
`.mex/context/lifecycle.md` repeats it, and
`tests/test_lifecycle_contract.py::test_bundled_vscode_vsix_matches_release_lifecycle_source`
already asserts `package["version"] == _VSCODE_EXT_VERSION`. **The guard
existed and was never run.** Any version bump must run that file.

**Trap 2: `uv tool install --force .` is not enough.** uv keys its build cache
on the project version. `pyproject.toml` stayed at 0.39.0 all day, so three
consecutive `--force` installs silently reused the first wheel: the tool env
still held `_VSCODE_EXT_VERSION = "0.14.92"` and a 0.14.90 VSIX long after the
source said 0.14.94. `--force` reinstalls the *package*; it does not rebuild it.
Use `uv tool install --force --reinstall .`, and verify by grepping the
installed copy in the tool env rather than trusting the command's output.

This is the second time in this log that version skew between the Python half
and the extension half produced a failure that looked like something else. It
is the single most reliable way to waste an afternoon here.

**Verification that actually means something** — grep the installed tool env,
not the working tree:

    grep _VSCODE_EXT_VERSION ~/…/uv-tools/arrayview/lib/python*/site-packages/arrayview/_vscode_extension.py

All three of today's Python-side fixes confirmed present in the tool env after
`--reinstall`: the 1-D reshape, the `render_error` send, and the viewer's
`reportParent('render-error')`.

### 2026-07-27 — "why so slow": 31 s to first frame on a 5-D file

`arrayview recons/parameter_maps_all.npy`, shape (224,240,204,6,9) float32,
2.37 GB on the `/smb` mount. Log timings for request 81d80328:

    signal → mode-change      1.1 s     (the whole opener path)
    mode-change → frame-rendered  31.2 s

So nothing in the opener, the tunnel, or the queue is implicated — the gap is
entirely the backend producing the first frame.

**Measuring it warm is a trap.** Repeating the extraction after the daemon had
already read the file gives 0.22 s for all 204 slices and 0.16 s for both
percentiles — 0.38 s against an observed 31.2 s. That number is meaningless:
the page cache was warm from the very run being investigated. The same pattern
against a file nothing had touched (`sec_001_to_009_excl_007_tight_ras_maps.npy`,
1.05 GB) takes **13.8 s for 43.9 MB — an effective 3.18 MB/s**.

**Cause**: the default view of this array is a mosaic over dim 2, so the first
frame needs all 204 slices — 43.9 MB — not one 210 KB slice. Worse, the display
dims are the *outermost* axes, so each slice is a strided gather (strides
10575360, 44064 bytes) scattered across the whole 2.37 GB. At ~3 MB/s over SMB
that is ~15 s of pure transfer for a 1 GB file and ~31 s for this one.

The percentile pass over 11 M elements is 0.12–0.16 s and is not worth touching.

**Not a bug, and not fixable by making the code faster** — the bytes have to
cross the mount. What is fixable is the *wait*: render the middle slice first
(0.1 s once its pages are in) and fill the mosaic in behind it, so first paint
is ~1 s instead of 31 s. Untried, and it changes what the user sees first, so
it needs agreement before implementing.

### Progress reporting for slow mosaic builds

Correction to the previous entry first: the "3.18 MB/s" figure there was
measuring *useful* bytes, not transferred bytes, and led to the wrong
conclusion. The right model is page-granularity. Exact arithmetic over the
strides (10575360, 44064, 216, 36, 4):

    full mosaic (204 slices)   useful 43.9 MB   pages touched 2368.9 MB   31.2 s
    one middle slice           useful  0.2 MB   pages touched  220.2 MB    2.9 s
    file size                                                 2368.9 MB

Pages touched equals the file size exactly, and 2368.9 MB at 76 MB/s is 31.2 s
— the observed time to the digit. Two unrelated files both landed on 76 MB/s
when computed as size/time, which was the clue: the mount is fine, the access
pattern reads everything. mmap is working as intended; it just cannot help when
4 wanted bytes sit in every 216-byte block and the blocks span the whole file.

Consequence for the earlier proposal: showing one slice first is worth ~3 s, not
~1 s. Still a 10× improvement, but do not oversell it.

**What was built instead**: honest progress. `render_mosaic` takes a progress
callback invoked per slice; the reporter hands each update to the event loop
(it is called from the render thread) and the viewer draws a bar plus an ETA
derived from measured elapsed time. Throttled to 200 ms, and silent below 24
slices so quick loads never flash a bar.

**Test note worth keeping**: mosaic mode needs a 4-D array. A 3-D array leaves
`dim_z` at -1 after pressing `z`, so a mosaic test built on `(24,24,60)` passes
through the non-mosaic path and silently proves nothing. Assert `dim_z >= 0`
before measuring anything about a mosaic.

**Second test note**: passing `--browser chromium` to a non-browser suite like
`test_api.py` produces ~49 failures and ~68 errors that have nothing to do with
the code under test. Run browser and non-browser suites separately.

**Still open**: first paint still waits for the whole mosaic. Rendering the
middle slice first and filling in behind it is the real latency fix and is
untried.

---

### 2026-07-27 — "backend stopped answering" was the relay, and a scoping lesson

**Input**: a launch failed with `Backend stopped answering before the viewer was
ready` while `/ping` on loopback answered `ok:true` with two live sessions, and
the next identical launch 57 s later reached `frame-rendered`.

**Cause, two parts, both about reading a probe result.**

`probeArrayViewStatus` treated any non-200 as `PROBE_DEAD`. A devtunnel answers
**502 in ~250 ms** whenever its connector is not attached — an error about the
relay, not the backend, and one that fails a launch *faster* than a stall does.
Measured directly against the live tunnel URL.

The readiness gate in `handleOpenRequest` was a single 1500 ms probe whose
negative answer abandoned the request. Timing confirms a timeout, not a
rejection: `panel_opened` 20:34:22.351 → `ERROR` 20:34:23.900 = 1.549 s.
`arrayViewStatusOk` collapses OK/UNKNOWN/DEAD into a boolean, which deletes the
distinction `_verifiedCachedTunnelBase` was already built around — so of the
five probe call sites, the one that abandons a user's launch was the only one
without the protection. The other four survive by sitting in retry loops.

Fixed in `b4f0cfc`: relay statuses (502/503/504/52x) off loopback are
`PROBE_UNKNOWN`; loopback non-200 stays fatal. The gate retries while the answer
means nothing (relay 3/5/8 s, loopback 1.5/2.5 s), abandons only on
`PROBE_DEAD`, and otherwise falls through to `waitForSessionReady`, which polls
to its own deadline. A port taken over by a newer backend now says so.

**Scoping lesson — this log's own 0.14.91 method note, ignored and re-learned.**
An unscoped sweep of all 37k log lines produced: 33 "did not render a frame",
24 integrated-browser, 17 "did not become ready", 7 "backend stopped answering"
— which reads as "the fix targeted 5% of failures". Scoped to >= 2026-07-26
(85 requests, 66 rendered, 16 failed) the real distribution is **7 backend-stopped,
4 panel-closed, 3 no-frame, 2 genuine bad files**. It was the plurality at 44%.
The archive is dominated by architecture fixed weeks ago. *Scope before ranking.*

**Second trap in the same analysis**: most failing requests lack a `ws-open`
phase, which looks causal. For anything abandoned at the readiness gate it is
reverse causation — the request dies before the viewer connects. 6 of the 7
were that. Exclude earlier-gate failures before treating a missing phase as a
cause.

### The one real gap that survived scoping: no WebSocket open timeout

`initWebSocket` retried forever on `onclose`/`onerror`, but a socket stuck in
CONNECTING fires neither. The viewer then walked through layout, emitted
`mode-change`, and waited silently for a frame that could never arrive; the
opener saw only its own deadline expire and reported "did not render a frame".
Worth ~2-5 requests in 85 — small, but the failure is a 30 s silent hang ending
in a misleading message.

Bounded the open: 6 s x 3 attempts, then a real message
("The viewer could not connect to the ArrayView backend.") plus a `render-error`
report, which the wrapper already turns into a terminal `viewer-failed` instead
of a timeout.

Two things that had to stay intact, and shaped the design:

- **Only the first connection is bounded.** After `_wsEverConnected`, a drop is
  a reconnect (server restarting) and must keep retrying forever.
- **An attempt is consumed by a hang, not by a fast close.** A socket refused
  immediately is the backend still binding its port; that retry is load-bearing
  during startup and stays unbounded. The counter increments inside the timeout
  callback, never at socket creation. Getting this backwards would have turned a
  ~1.5 s startup race into a hard failure.

Verified against a real served viewer with three real sockets: normal (renders,
no error); a listener that accepts TCP and never upgrades (gives up at 18.9 s
with the message, exactly 3 main-socket attempts, no retries after); and a port
that refuses fast 8 times then works (9+ attempts, no give-up, renders).

**Test note**: `reportParent` no-ops when `window.parent === window`, so a
top-level page emits no phases at all — phase-based assertions on a directly
loaded viewer prove nothing. Assert user-visible outcomes, or load it in an
iframe. Also `/ws/shell` is a second socket; count `/ws/<sid>` only.

---

### 2026-07-28 — 0.14.97: the relay has no slow tail, and the ladder was built for one

**Input**: "loaded decently fast, closed the tab, ran it again ten seconds
later, took a minute". Two launches of the same 5.8 GB `coil_sensitivities.npy`
five minutes apart, request `2af364bc` then `b4b14217`.

    2af364bc  signal -> frame-rendered   1.4 s
    b4b14217  signal -> frame-rendered  29.0 s

The 29 s decomposes into three parts, and the first two were pure waste:

    13.6 s  cached-route probe: no verdict x3 (1.5 / 4 / 8 s budgets)
     6.5 s  asExternalUri + privacy promotion, re-deriving the route
     8.3 s  the 1.9 MB viewer page crossing the relay (known fixed cost)

**The re-derivation returned the byte-identical URL** the probe had just
discarded — `https://v54z0psh-8000.euw.devtunnels.ms`, already sitting in
`tunnel-routes.json` when the request started.

**Measured cause.** Probing the live relay directly (30 fresh connections,
`/ping`, backend healthy on loopback throughout, no VS Code involved):

    answered  25/30    median 185 ms   p90 233 ms   max 639 ms
    stalled    5/30    no response, no reset, no error, to a 15 s budget

**There is nothing between 639 ms and never.** Roughly one connection in five
is accepted by the relay and then black-holed; every connection that answers
answers inside 640 ms. So the escalating ladder was waiting in a region where
no answer has ever arrived. Widening a budget cannot recover a request that is
already lost — only a *new connection* can, and stalls are largely independent
per connection: in a staggered test a hedge opened 1.5 s after a stalled probe
answered in 205 ms.

This also kills two plausible-looking theories. It is **not** a cold relay:
correlating all 81 probe episodes since 2026-07-26 against idle time, OK after
35,636 s idle takes 0.5 s while UNVERIFIED after 10.7 s idle takes 13.7 s —
median idle is 98.6 s for OK and 84.8 s for UNVERIFIED, i.e. uncorrelated. It is
**not** Node's keep-alive socket pool either: the reproduction stalls with
`reusedSocket=false`, on a brand-new connection.

**The probe never once caught a bad route by stalling.** All 6 UNVERIFIED
episodes in the scoped log ended at the identical URL they had discarded, 6.5-90 s
later. One of them (20:34:08 on 2026-07-27) was pushed past a downstream
deadline into `Backend stopped answering before the viewer was ready` — the
delay did not merely cost time, it manufactured a failure.

**Fix, three parts.**

1. *Hedge, don't escalate.* `hedgedProbeStatus` opens overlapping attempts
   (3 x 2000 ms, 700 ms stagger) and takes the first real verdict. A stall now
   costs the stagger, not the budget. `agent: false` keeps attempts on
   independent connections — pooling would let a hedge inherit the wedged
   socket. Applied to the cached-route check and to relay readiness; loopback
   keeps sequential retry, because a loopback port has no black-hole mode.
2. *A no-answer may not discard a route.* `_verifiedCachedTunnelBase` became
   `_usableCachedTunnelBase`: only PROBE_DEAD discards. This completes the
   asymmetry argument the old comment already made but stopped halfway through.
   The panel is the better probe — it navigates on its own connection and its
   watchdog re-arms `frame.src` — so an unverifiable route goes to the panel
   rather than to 90 s of resolver backoff.
3. *A 502 is information; a stall is not.* New `PROBE_RELAY_DOWN`, split out of
   PROBE_UNKNOWN. A detached connector answers in ~200 ms saying the relay is up
   but not carrying the port; port promotion is what **fixes** that, so it must
   keep falling through to promotion instead of short-circuiting on the cache.
   Collapsing it into "no answer" would have regressed that path — the relay was
   in exactly this state while this entry was being written.

`deadBases` threads proof through the two or three cache consultations in one
request, so a candidate proven foreign cannot be resurrected by a later probe
that happens to stall.

**Expected effect on the reported incident**: 13.6 s + 6.5 s -> under 1 s, so
~29 s becomes ~9.5 s. The remaining 8.3 s is the 1.9 MB page transfer, which
this change does not touch. Caching that page is still untried and is now the
dominant term.

**Evidence**: `real process` for the measurements (live relay, 30+12 probes,
plus the new hedge run against it end to end through the real `extension.js`)
and `component` for the logic (17/17 extension tests, 77 Python lifecycle/ack
tests). The stall path itself was **not** observed live during the fix run — the
relay was uniformly 502 by then — so it is covered by the 17% measurement plus
deterministic unit tests, not by a live stall. Real-host launch evidence needs
0.14.97 installed and the windows reloaded.

**Method note worth keeping**: three theories (cold relay, stale socket pool,
relay saturation) were all killed by cheap measurements — an idle-time
correlation over the existing log, `reusedSocket` instrumentation, and a
staggered two-connection test. None needed VS Code. When a launch symptom points
at the relay, probe the relay directly first; it is a two-minute experiment and
it has now overturned the obvious explanation twice.
