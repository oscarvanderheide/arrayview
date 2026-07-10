# Launch Reliability Plan

Status: implementation in progress, 2026-07-10. This supersedes the first
launch-routing plan.
The earlier work added environment snapshots and several targeted fixes, but the
shared planner and end-to-end recovery protocol were not completed.

Current implementation status:

- complete locally: executable planner, CLI and `view()` display execution,
  diagnostics, invocation matrix, instance identity/registry/startup lock,
  identity-safe stop, explicit handle cleanup, structured opener results,
  versioned VS Code request/ACK, verified backend readiness, doctor/instances/
  stop commands, host docs, bundled VSIX, and three-OS CI configuration;
- foundation complete: shared `SessionSpec` and registration capability model;
- requires real tunnel evidence: exact multi-window routing, forwarded URL and
  privacy behavior, reconnect/reload recovery, and ACK timing;
- still future work: authenticated viewer access for public tunnel exposure,
  full registration migration to `SessionSpec`, and automatic browser/notebook
  lease expiry beyond explicit close/panel release.

## Outcome

Opening ArrayView should become boring:

- the same request behaves consistently from the CLI, Python, Jupyter, MATLAB,
  Julia, a VS Code terminal, or the VS Code Explorer;
- local, SSH, and VS Code tunnel launches select the correct display without the
  caller knowing about ports or server processes;
- repeated launches and several simultaneous VS Code windows reuse the correct
  backend and open in the originating window;
- crashed or outdated ArrayView backends recover automatically when safe;
- ArrayView never kills an unrelated process because it happens to own a port;
- `--diagnose` explains the exact plan that real execution will use;
- Windows is a supported CI target even though the primary development machines
  are macOS and Linux.

This is a lifecycle and routing redesign. It is not a sequence of extra
environment checks in `_launcher.py`.

## Current State

The repository already has useful foundations:

- `_launch_plan.py` defines launch dimensions and an environment snapshot;
- `_platform.py` detects Jupyter, MATLAB, Julia, VS Code, SSH, and native GUI
  capability;
- the FastAPI `/ping` route identifies ArrayView servers;
- the VS Code extension registers windows, forwards ports, resolves external
  URIs, and opens webview panels;
- `.mex/context/lifecycle.md` defines ownership and shutdown expectations;
- `tests/lifecycle_matrix.py` groups several lifecycle checks.

The foundation is incomplete:

- `LaunchPlan` exists, but `plan_launch(...)` does not;
- CLI and `view()` still compute policy through separate helpers and branches in
  `_launcher.py`;
- `--diagnose` independently reconstructs policy instead of printing the plan
  used by execution;
- port discovery, backend identity, compatibility, ownership, and stale-process
  recovery are not represented as one atomic decision;
- opening a VS Code tab is an asynchronous signal-file attempt, not a
  request/acknowledgement protocol;
- predictable tunnel ports and public visibility are operational requirements,
  but setup, readiness, and failure recovery are spread across Python and the
  extension;
- MATLAB is detected as a special host but is absent from the launch model;
- automated tests cover many helpers, but not the complete invocation x host x
  lifecycle contract.

## Model The Problem On Independent Axes

Do not encode combinations such as "VS Code Julia over SSH" as special modes.
Represent the axes independently:

| Axis | Examples |
|---|---|
| Invocation | CLI, Python API, VS Code Explorer, MATLAB bridge, Julia bridge |
| Host | terminal, Jupyter kernel, MATLAB, Julia/IJulia |
| Placement | local, plain SSH, VS Code remote/tunnel |
| OS | macOS, Linux, Windows |
| Display intent | auto, native, browser, VS Code, inline, none |
| Server scope | transient viewer, caller-owned, kernel-owned, persistent user server |
| Server state | absent, compatible, incompatible, stale record, foreign port owner |
| Registration | in-process array, HTTP file load, HTTP byte upload, daemon startup, relay |

The full Cartesian product is too large to hand-code or manually test. The
planner applies orthogonal rules, and tests cover each rule plus high-risk
pairwise combinations.

## Target Architecture

```text
CLI / view() / Jupyter / MATLAB / Julia / VS Code Explorer
                         |
                  normalized LaunchIntent
                         |
             snapshot facts + discover instances
                         |
               pure plan_launch(intent, facts)
                         |
           LaunchPlan + ordered recovery actions
                         |
       +-----------------+------------------+
       |                 |                  |
 server executor   registration adapter   display opener
       |                 |                  |
       +---------- launch result -----------+
                         |
            request id, URL, server id, ACK
```

### 1. Thin invocation adapters

Each public entry point only validates input and builds a `LaunchIntent`.
It does not detect environments, choose a port, kill a process, or choose a
fallback display.

Add explicit invocations for `vscode_explorer`, `matlab`, and `julia`; Jupyter
is a host/placement fact as well as a display option. Preserve the existing API
surface while converting it internally.

### 2. One fact snapshot

Extend `LaunchEnvironmentSnapshot` into the only source of launch facts. It
must be immutable and cheap to build before importing NumPy or the server stack.
Include:

- OS, host, local/SSH/tunnel placement, notebook state, and GUI backend;
- requested and configured display intent, with the source of the choice;
- candidate ArrayView instances and their protocol/package versions;
- port owner state: free, compatible ArrayView, incompatible ArrayView, or
  foreign process;
- exact VS Code window registration and extension capability/version;
- whether tunnel forwarding, external URI resolution, and required privacy are
  confirmed—not merely requested.

Detection functions return evidence and confidence. A cached heuristic must
not silently override a later explicit fact.

### 3. One pure planner

Implement:

```python
plan_launch(intent: LaunchIntent, facts: LaunchFacts) -> LaunchPlan
```

The plan is data-only and serializable. It selects:

- server instance and ownership scope;
- transport and registration adapter;
- display opener and fallback chain;
- port policy;
- recovery actions and which are safe automatically;
- lifecycle release policy;
- user-visible warnings or terminal errors;
- reasons for every non-default decision.

CLI, `view()`, `--serve`, Explorer opens, bridges, and `--diagnose` must all use
this same function. Executors may report that an action failed, but they must
return to the planner for a fallback instead of inventing policy.

### 4. Instance registry, identity, and compatibility

Stop treating a port as the identity of a backend. Maintain an atomic,
per-user runtime registry outside the repository. Each entry contains:

- random server instance ID and management token;
- PID plus process start identity, hostname, port, and start time;
- ArrayView package version and launch-protocol version;
- owner scope and owner ID (viewer, process, kernel, or persistent user);
- capabilities and last successful health check.

Use a filesystem lock around discovery/startup so two simultaneous launches do
not start competing daemons. Validate PID and process-start identity to avoid PID
reuse. Treat missing processes as stale records and clean records automatically.

A port hosting another application is a foreign owner. Never kill it. Either
choose another port or, where a stable tunnel port is explicitly required, fail
with a precise explanation. Replace an ArrayView instance automatically only
when identity and ownership prove that it is safe and the requested server scope
permits replacement.

The health handshake must distinguish:

- alive and protocol-compatible: reuse;
- alive and backward-compatible: reuse through advertised capabilities;
- alive but incompatible: start a separate instance or request an explicit
  upgrade/restart;
- dead/stale registry entry: remove and restart;
- TCP listener that is not ArrayView: leave untouched.

Make `8000` the canonical explicit/default port for both CLI and `view()`.
During one compatibility release, discover a healthy legacy API server on
`8123`, report that migration in diagnostics, and reuse it only when the plan
says compatibility is safe. Local automatic fallback ports are registry-owned
implementation details; tunnel/persistent ports remain stable configuration.

Server startup is a transaction: acquire the scope lock, recheck discovery,
spawn with an expected nonce and retained log/process handle, wait for an HTTP
status matching nonce, PID, instance ID, and protocol, register the session,
verify that exact SID, then open the display. On failure, clean temporary state
and terminate only the process identity created by this transaction.

### 5. Explicit lifecycle scopes

Use a small set of ownership contracts:

| Scope | Used by | Lifetime |
|---|---|---|
| Transient viewer | ordinary CLI/browser/native launch | through the last attached viewer |
| Caller-owned | long-running Python process | explicit handle close or caller shutdown policy |
| Kernel-owned | Jupyter/IJulia | through kernel lifetime; iframe loss is not a hard kill |
| Persistent user | `--serve`, tunnel workflows that opt into persistence | explicit stop or managed idle policy |

Sessions are reference-counted independently from servers. Every display open
receives a lease token; explicit panel/window close releases that lease. A crash
is handled by heartbeat/expiry, not by assuming every WebSocket disappearance is
intentional. Starting a new invocation must never require manual `--kill` for a
stale registry or dead ArrayView process.

Keep `--kill` as an administrative command, but change it to target a proven
ArrayView instance ID or all owned ArrayView instances. A port alone is not
sufficient authority. A separately named `--force-kill-port` may remain as an
explicit dangerous escape hatch, with confirmation and no use in normal repair.

### 6. Display openers return acknowledgements

Define one interface for native, browser, inline, VS Code, and no-display
openers. An opener returns a structured result: accepted, ready, timed out,
unsupported, or failed, with evidence.

For VS Code, replace fire-and-forget signaling with a request protocol:

1. Python writes an atomic request containing request ID, exact window ID,
   server identity, local URL, expiry, and required capabilities.
2. Only the registered target window claims it.
3. The extension ensures forwarding and the required privacy, calls
   `asExternalUri`, verifies backend health from the extension host, and creates
   or reuses the panel.
4. The extension writes an ACK containing the resolved URL, panel ID, and
   forwarding state, or a typed error.
5. Python waits for the ACK with a bounded timeout and applies the planned
   fallback or prints one actionable diagnostic.

Shared/broadcast signaling remains only as a compatibility fallback. With
multiple windows and no provable target, fail closed; never guess which window
the user meant.

### 7. Tunnel setup becomes managed state

The extension—not shell instructions—owns VS Code port forwarding. The desired
state is explicit: port, protocol, privacy, source instance, and external URI.
It reconciles this state on activation, server restart, panel reopen, and tunnel
reconnect.

Because a public forwarded port may be reachable beyond the current user, add
an unguessable per-instance access token to viewer HTTP/WebSocket requests before
making automatic public exposure the default. Do not put reusable management
credentials in viewer URLs or logs.

First prototype a supported private extension-host proxy/tunnel that carries
both HTTP and WebSocket traffic. Prefer it if it removes the public-port
requirement. Do not depend on VS Code's private `remote.tunnel.privacypublic`
command as the only production path. If public forwarding remains necessary,
capability authentication and explicit verified privacy state are release
blockers.

If a tunnel platform cannot programmatically set the required privacy, the ACK
must report `manual_privacy_required` and identify the exact port. The user does
the manual step once; subsequent launches confirm and reuse it.

### 8. Registration adapters share one semantic contract

File load, byte upload, in-process arrays, Julia/MATLAB serialization, relay,
compare, overlay, RGB, directories, and array-key selection must yield the same
`SessionSpec`. Centralize validation and feature capability checks before
transport-specific execution.

MATLAB and Julia are adapters, not environment exceptions inside `view()`:

- both use an out-of-process server path where host/GIL constraints require it;
- both pass serialized intent and receive a structured launch result;
- Julia terminal and IJulia select different display openers through facts;
- MATLAB desktop, batch, and remote sessions are separately classified;
- unsupported combinations fail before a server is started and say exactly what
  is unsupported.

The VS Code Explorer adapter must use an explicitly configured or VS Code
Python-extension-selected interpreter. It must not independently guess a
workspace `.venv`, `uv`, or `python3` and silently launch a different ArrayView
version from the terminal/notebook environment.

## Recovery Policy

Recovery should be automatic only when ArrayView can prove ownership.

| Failure | Automatic action |
|---|---|
| Registry entry exists, process is gone | remove record and start a compatible instance |
| Compatible ArrayView instance is healthy | reuse it |
| Owned transient ArrayView is hung | authenticated graceful stop, then replace |
| Foreign process owns preferred local port | select another port |
| Foreign process owns required tunnel port | stop with exact port-owner diagnostic |
| Exact VS Code target is stale and one window exists | refresh registration, then retry once |
| Exact VS Code target is unknown and several windows exist | fail closed with window-target diagnostic |
| Extension is missing/outdated | install/update once, wait for activation ACK, retry once |
| Forwarded port disappeared | extension recreates it and resolves a fresh external URI |
| Native opener fails | use the plan's browser fallback; do not change server ownership |
| Caller crashes | lease expires; release session; stop transient server when last lease is gone |

Every retry has an idempotency key and a fixed budget. Avoid layered Python and
extension retry loops that multiply attempts.

## User-Facing Contract

Keep the common path quiet. On failure, print one summary and one next action.

Add:

```text
arrayview doctor
arrayview doctor --json
arrayview instances
arrayview stop <instance-id>
arrayview stop --all
```

`doctor` calls the same snapshot and planner as launch, then performs optional
non-destructive probes. It should show:

- detected invocation/host/placement/OS;
- requested display and where that preference came from;
- chosen server, transport, display, target VS Code window, and fallback;
- instance compatibility and ownership evidence;
- tunnel forwarding/privacy/extension state;
- typed warnings with copyable remediation.

Keep `--diagnose` as a deprecated alias until users have migrated. Add a short
request ID to verbose output so Python and extension logs can be correlated.
Add extension commands for ArrayView status, repair connection, and open/copy
redacted diagnostics; the blank/error panel should expose those same actions.

## Implementation Phases

Each phase lands as a separate feature branch/PR and preserves behavior unless
its acceptance criteria explicitly change it.

### Phase 0: Baseline and executable contract

- inventory every public entry point and current ownership rule;
- turn the lifecycle table into parameterized contract fixtures;
- record current known failures without encoding accidental behavior as the new
  contract;
- add deterministic fakes for OS, process table, ports, clocks, VS Code windows,
  and extension acknowledgements.
- lock down known platform hazards: local Linux VS Code versus remote/tunnel
  classification, exact macOS multi-window routing, and Windows parent-process,
  interpreter, process-control, and SSH-guidance behavior.

Exit: every supported combination has an owner in the matrix and is marked
automated, CI-emulated, or manual hardware verification.

### Phase 1: Complete the planner

- add `LaunchIntent`, richer facts, `plan_launch`, reasons, and typed failures;
- route `--diagnose`/`doctor` through the planner;
- adapt CLI and `view()` to execute plans without changing launch behavior;
- delete the old resolver helpers only after parity tests prove they are unused.

Exit: the executed plan and diagnosed plan are byte-for-byte equivalent after
removing volatile fields; no launch policy remains in CLI/API executors.

### Phase 2: Instance manager and safe self-healing

- add locked instance discovery, identity handshake, compatibility negotiation,
  and lease management;
- migrate `--serve`, reuse, stale cleanup, and stop operations;
- remove blind port-based process killing;
- add simultaneous-launch, crash, PID-reuse, version-skew, and foreign-port
  process tests.

Exit: 100 concurrent launch-planning/start attempts create at most one instance
per intended scope; stale ArrayView state needs no manual `--kill`; foreign
processes are never terminated.

### Phase 3: Display opener protocol

- define structured opener results and fallback execution;
- make native/browser/inline openers conform;
- add request IDs and readiness evidence;
- ensure display failure cannot silently mutate registration or server policy.

Exit: each opener has deterministic timeout/failure tests, and fallback follows
the serialized plan exactly.

### Phase 4: VS Code ACK and tunnel reconciliation

- add versioned request/ACK files or a local authenticated IPC channel;
- make exact window targeting mandatory when several windows are registered;
- move forwarding/privacy/external-URI readiness behind the extension ACK;
- evaluate and select the supported private-proxy path or the authenticated
  public-forwarding fallback before migration;
- reconcile tunnel state after reconnect and backend restart;
- keep one release of compatibility with existing signal files;
- add viewer access tokens before automatic public exposure.

Exit: repeated launches from two local and two tunnel windows always open in the
originating window; a tunnel reconnect or server restart recovers without VS
Code reload, manual `--kill`, or rewriting port settings.

### Phase 5: Registration and foreign-host adapters

- converge all transports on `SessionSpec`;
- move MATLAB and Julia to explicit adapters;
- cover Jupyter, IJulia, MATLAB desktop/batch, VS Code Explorer, file keys,
  compare, overlays, and directory registration;
- negotiate capabilities when a compatible older server is reused.

Exit: equivalent data/options produce equivalent session metadata across every
adapter, or a typed preflight error before side effects.

### Phase 6: Cross-platform gate and rollout

- run planner and process tests on macOS, Ubuntu, and Windows CI;
- run extension tests on Linux and Windows path/filesystem semantics;
- keep macOS/Linux native and real tunnel smoke checks as scheduled/manual jobs;
- add upgrade tests with the previous released Python package and VSIX;
- publish concise migration/recovery docs after the implementation stabilizes.

Exit: the support matrix is green, compatibility behavior is documented, and
release notes contain no routine instruction to use `--kill` or reload VS Code.

### Remote tunnel test handoff

At the end of every tunnel-sensitive phase, and once more at the end of the
complete implementation, write or refresh:

```text
dev/launch-tunnel-test-handoff.md
```

This is an executable handoff for another agent running inside a real VS Code
remote tunnel window. It must contain:

- commit and branch under test, package/VSIX versions, remote OS, and Python;
- what changed and which local automated checks already passed;
- required VS Code window/tunnel setup, test data, ports, and preconditions;
- exact commands and UI actions for each remaining real-tunnel scenario;
- expected server ID, window ID, request/ACK states, URL/forwarding state, and
  lifecycle outcome;
- log locations and request IDs needed to correlate Python and extension events;
- a result table with `PASS`, `FAIL`, or `BLOCKED`, plus evidence paths;
- safe cleanup commands that target ArrayView identities, never arbitrary port
  owners;
- known limitations and the precise questions the tunnel agent must answer.

The handoff must not ask the remote agent to repeat tests already proven locally.
It should isolate only behavior that requires the real extension host, forwarded
port, external URI, remote window identity, or reconnect lifecycle. A
tunnel-sensitive phase is not complete until the handoff exists and the remote
result has either passed or been recorded as an explicit release blocker.

## Verification Matrix

At minimum, automate or explicitly schedule these scenarios:

| Scenario | Required assertions |
|---|---|
| CLI local, macOS/Linux/Windows | correct opener; reuse; close releases transient instance |
| Python short script | viewer survives caller long enough to attach; no orphan after close |
| Python long process | handle close releases its session without affecting other sessions |
| Jupyter and VS Code notebook | kernel-owned reuse; disappearing iframe does not kill server |
| Julia terminal and IJulia | subprocess isolation; correct browser/inline selection |
| MATLAB desktop and batch | no false VS Code detection; native/browser/no-display contract |
| Plain SSH | printed/selected forwarding contract; no accidental remote browser |
| VS Code local Explorer/terminal | exact originating window; extension version compatibility |
| VS Code tunnel, several windows | exact target; ACK; forwarded port; reconnect recovery |
| Two simultaneous launches | one compatible instance; two independent sessions |
| Dead registry / hung owned server | automatic bounded recovery |
| Foreign preferred-port owner | never killed; alternate port or precise fixed-port error |
| Old Python/new extension and inverse | capability negotiation or explicit upgrade error |

Use real subprocesses and sockets for lifecycle tests; mocks alone cannot prove
process shutdown, lock behavior, or port ownership. Use CI fault injection for
crash points between registry write, bind, session registration, signal claim,
port forwarding, ACK, and lease release.

## Success Measures

The redesign is complete when:

- routine local and tunnel launches do not require `--kill` or VS Code reload;
- the same intent produces the same serialized plan across public entry points;
- every started process has a recorded owner and deterministic cleanup path;
- multiple VS Code windows never receive a best-guess launch;
- diagnostics reproduce the actual decision and correlate Python/extension logs;
- a foreign process is never terminated by ArrayView recovery;
- Windows planner, lifecycle, and browser/VS Code paths pass in CI;
- release verification includes real macOS, Linux, and tunnel smoke evidence.
- another agent can execute the final remote-tunnel handoff without reconstructing
  context from chat or git history.

## Explicit Non-Goals

- one immortal global daemon for every environment;
- hiding a security-sensitive public-port change from the user;
- preserving undocumented behavior caused by current branch order;
- testing the full Cartesian product manually;
- adding more launch policy to `_app.py` or the viewer frontend.

## Recommended First Slice

Implement Phase 0 and Phase 1 together, without changing lifecycle behavior.
That slice makes the current decisions observable and forces CLI, `view()`, and
diagnostics through one planner. Only then introduce the instance registry and
self-healing; otherwise recovery will hard-code another copy of today's routing
policy.
