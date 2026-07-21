# ArrayView Launch Execution Convergence Plan

Status: proposed; implementation not started  
Date: 2026-07-21  
Scope: CLI, Python, Jupyter, Julia, MATLAB, VS Code local/remote, and plain SSH launch behavior

## Executive decision

The previous launch redesign was directionally correct but stopped halfway. It
centralized launch *planning* without making that plan authoritative during
execution. The solution is not another large `LaunchCoordinator` class and not
another mechanical split of `_launcher.py`.

The recommended design is:

- one dependency-free source of launch policy;
- one small phase orchestrator that executes that policy and records evidence;
- separate server-ownership, registration, and display strategies for the
  mechanisms that are genuinely different;
- no cross-strategy routing, environment-default selection, lifecycle choice,
  or fallback selection inside those strategies;
- incremental migration of one complete invocation slice at a time, deleting
  the old path for each migrated slice;
- black-box and host-gated evidence before a slice is called complete.

This is an execution-convergence project, not a rewrite. The existing planner,
instance registry, readiness ACKs, `SessionSpec`, and lifecycle work should be
completed and connected, not replaced under new names.

## Why this plan is different from the previous one

`docs/launch-routing-refactor.md` already proposed a shared intent, fact
snapshot, planner, instance registry, registration contract, display openers,
and verification matrix. Much of its foundation was implemented and then
described as complete. Its own exit criteria were not met:

- `LaunchPlan.fallback_display` and `fallback_allowed` have no production
  consumer;
- `SessionSpec` is only used by its tests, not by a real invocation path;
- `view()` converts the plan back into `window`, `inline`, and `_force_*`
  booleans and continues through legacy routing;
- CLI execution converts the plan back into `window_mode` and
  `use_native_shell` and implements fallback locally;
- `_open_browser()` still detects VS Code/remote state and therefore remains a
  second router;
- old resolver functions with no production callers are still tested;
- the actual native, VS Code, and Jupyter handoffs are marked manual in the
  lifecycle matrix;
- earlier extraction work moved helpers without establishing one behavioral
  owner, so `_launcher.py` remained where new launch policy accrued.

The earlier architecture was therefore not disproved. Its migration and
completion discipline failed. This plan makes production call sites, legacy
deletion, hermetic tests, and real handoff evidence explicit exit gates.

## Evidence behind the diagnosis

The current launch stack has several independent decision points:

- `_launch_plan.py` chooses display, server ownership, registration, and a
  fallback.
- `_launcher.py` rechecks server and environment state, reconstructs display
  intent as booleans, and contains multiple fallback branches.
- `_vscode_browser.py::_open_browser()` detects local VS Code, remote VS Code,
  SSH, and the operating system, then chooses a mechanism itself.
- `_platform.py::_native_window_gui()` answers whether a native backend exists,
  while `_can_native_window()` also embeds an auto-routing preference against
  VS Code terminals. The planner and executor can therefore disagree about the
  same explicit native request.
- spawned CLI, existing-server CLI, in-process Python, existing-server Python,
  Julia serialization, and SSH relay register sessions differently.
- readiness can mean TCP bound, HTTP `/ping` ready, SID declared, data loaded,
  shell WebSocket connected, viewer WebSocket connected, first frame rendered,
  or VS Code ACK received. These milestones are currently combined through
  fixed timeouts and local heuristics.

The repository history supports this being systemic rather than a single bug.
Since 2026-06-01, 66 commits touched `_launcher.py`, `_launch_plan.py`,
`_platform.py`, or `_vscode_browser.py`, adding 3,007 lines and removing 1,182.
The current `_launcher.py` is 4,723 lines.

The lifecycle matrix also demonstrates the coverage gap. It passes the pure
planner, CLI helpers, protocol helpers, packaging, and one real daemon/WebSocket
shutdown probe. It leaves native, live VS Code, and Jupyter as manual. In a
normal development workspace with an ArrayView daemon on port 8000, two
supposedly isolated lifecycle contract tests currently divert into that live
server and fail. A test suite for launch ownership must not depend on whether a
developer happens to have ArrayView open.

## Goals

1. The explicit request made by the caller remains available until the launch
   succeeds or returns a typed failure.
2. The plan is the only authority for server-scope constraints, permitted
   registration strategies, display choice, fallback order, and retry budget.
   Dynamic outcomes such as the actual port, created-versus-reused server, or
   current VS Code target are transaction results, not frozen predictions.
3. Execution may revalidate mutable resources, but it may not invent a new
   policy.
4. Large-file startup remains concurrent: data loading, forwarding, and display
   acceptance may overlap safely.
5. All invocation paths share contracts and results while retaining the
   mechanism-specific code their environments require.
6. Existing servers are reused only when identity, protocol, and required
   capabilities match.
7. Every process and session has an owner, a correlation ID, and a deterministic
   cleanup path.
8. Tests execute real entry points, servers, sockets, and registration paths.
   Mocks remain useful for pure policy and fault injection, but cannot be the
   only release evidence.
9. No migrated launch path can fall back into a legacy router.
10. Startup performance and lazy-import behavior do not regress.

## Non-goals

- Do not force every environment into the same server ownership model.
- Do not put Jupyter, native, browser, and VS Code mechanics into one class.
- Do not force in-memory arrays, file paths, HTTP uploads, Julia temp files, and
  SSH relay through one physical transport.
- Do not perform a big-bang rewrite or a mechanical `_launcher.py` split.
- Do not change public CLI/API defaults, default ports, or documented fallback
  behavior as part of the refactor. Those require separate product decisions.
- Do not remove early display startup. Replace unsafe timing assumptions with
  explicit readiness dependencies.
- Do not claim real VS Code tunnel coverage from a mocked extension host.
- Do not add launch logic to `_app.py`.

## Architectural rule: unify authority, not mechanisms

The design has four layers:

```text
CLI / Python / Jupyter / Julia / MATLAB / VS Code Explorer
                           |
                    LaunchRequest
                    + SessionSpec
                           |
            stable host evidence + discovery facts
                           |
                   pure plan_launch()
                           |
              thin launch phase orchestrator
            /              |               \
 ownership strategy   registrar strategy   display adapter
 CLI subprocess       file / HTTP           native
 Python in-process    in-memory              system browser
 kernel-owned         serialized temp        VS Code
 persistent server    relay                   inline / none
            \              |               /
                    LaunchResult + trace
```

The orchestrator should stay small. A provisional limit is roughly 150 lines
of control flow, excluding data contracts and tests. Its responsibilities are:

- execute the selected plan;
- enforce phase prerequisites;
- propagate one launch/correlation ID;
- apply the plan's retry and fallback budget;
- maintain a cleanup stack for resources created by the current launch;
- return a structured result and trace.

It does not load arrays, start PyWebView, talk to IPython, or configure VS Code;
those mechanisms remain in strategies/adapters. Fallback desirability remains
planner policy and fallback advancement remains orchestrator behavior.

## What stays separate

### Ownership strategies

These are intentionally different:

| Strategy | Required behavior |
|---|---|
| CLI transient subprocess | survives CLI handoff; stops after final viewer |
| Python caller-owned | survives as required by script semantics; explicit handle cleanup |
| Jupyter kernel-owned | reused across cells; iframe loss is not server death |
| Julia/PythonCall subprocess | never starts in-process because of GIL constraints |
| Persistent user/tunnel server | survives viewers until explicit stop or defined idle policy |

They share `ServerRequest`, `ServerHandle`, and readiness results, not an
implementation.

### Registration strategies

These remain different transports:

- in-process array registration;
- existing-server HTTP file registration;
- existing-server serialized array upload;
- new-daemon startup registration;
- Julia/MATLAB temporary serialization;
- SSH relay.

Every strategy must accept the same semantic `SessionSpec` plus the applicable
out-of-band `SourceBindings`, validate required capabilities, and return the
same `RegisteredSession` contract.

### Display adapters

Native, system browser, VS Code, inline Jupyter, IJulia, and no-display retain
their mechanism-specific implementations. An adapter receives a complete
`DisplayRequest` and returns a `DisplayResult`. It may perform mechanism-local
bounded attempts within the budget assigned by the plan, but it cannot detect
another environment or choose another adapter. The orchestrator accounts for
the consumed budget and alone advances to the next fallback adapter.

## Core data contracts

Use frozen, JSON-serializable descriptors/events without NumPy, FastAPI,
uvicorn, IPython, or PyWebView imports. Keep live arrays, process handles,
callbacks, IPython return artifacts, and cleanup objects in a separate runtime
context; do not pretend they serialize.

### `LaunchRequest`

Preserves caller intent without booleans that erase meaning:

- invocation adapter;
- requested display: `auto`, `native`, `system_browser`, `vscode`, `inline`,
  or `none`;
- whether the display request was explicit and where it came from: API, CLI,
  config, or environment default;
- explicit fallback policy;
- requested port and persistence;
- `SessionSpec` reference;
- an out-of-band `SourceBindings` reference for live Python arrays or other
  non-serializable payloads;
- optional test/diagnostic correlation ID.

`window=True`, `window=False`, legacy Julia behavior, and CLI flags are
normalized once at the public boundary. Downstream code never sees those legacy
forms.

### `HostEvidence` and `DiscoverySnapshot`

Do not freeze all launch facts into one snapshot. Host evidence is stable for a
launch and must be captured once before planning, especially before Julia or
another wrapper can strip environment markers:

- invocation host and placement;
- OS and display-server facts;
- native backend capability, separate from auto-routing preference;
- VS Code terminal/window evidence;
- SSH/tunnel evidence;
- config values and their source.

Detector failures must be represented as `unknown` plus evidence, not silently
converted to `False`. A failed cached IPC lookup must have an expiry or explicit
invalidation rule.

Ports, server health/capabilities, VS Code window registrations, forwarding,
and extension ACK state are dynamic. Capture them in a `DiscoverySnapshot` for
planning, then revalidate them transactionally by the responsible strategy
immediately before a side effect. Revalidation can produce `resource_changed`
and return to the orchestrator; it cannot change display or ownership policy
itself. A timeout means `unknown`, not `foreign` or `incompatible`.

### `LaunchPlan`

Must contain all decisions execution needs:

- ownership strategy and scope;
- server reuse/start policy;
- required protocol and server capabilities;
- registration strategy;
- primary display adapter;
- ordered fallback adapters;
- required shared milestones and desired outcome; the selected adapter declares
  which shared milestones it supports and its mechanism-internal sequence;
- one overall deadline, per-adapter attempt budgets, and typed recovery actions;
- lifecycle/lease policy;
- reasons and warnings.

The plan must preserve the difference between “explicit native with system
browser fallback” and “automatic system browser.” It must never collapse both
to `display=browser`.

### Results

Define stable serializable result descriptors rather than booleans:

- `ServerHandle`: instance ID, PID/process identity, port, protocol,
  capabilities, owner scope, created/reused.
- `RegisteredSession`: primary SID, related SIDs, pending/ready state,
  registration mechanism, lease token.
- `DisplayResult`: adapter, `accepted`/`ready`/`unsupported`/`failed`, evidence,
  and process/panel/request identity descriptors.
- `LaunchEvent`: phase, monotonic timestamp, IDs, outcome, typed error.
- `LaunchResult`: plan, server, session, display, and trace location.

A separate `LaunchRuntime` owns non-serializable `SourceBindings`, live process
handles, callbacks, cleanup stack, and the IPython/IJulia artifact returned to
the caller.

## Readiness is a partial order, not a linear pipeline

A large dataset must not block all display work. The correct dependency graph
is:

```text
SERVER_RESERVED
   |              \
   |               +----> FORWARDING_REQUESTED ----> FORWARDING_READY
   |               |
   |               +----> optional DISPLAY_PROCESS_PREPARED
   |
BACKEND_HTTP_READY ----> SID_DECLARED ----> DATA_LOADING ----> SESSION_READY
         |                    |                                      |
         +--------------------+----> DISPLAY_PRESENTED/ACCEPTED      |
                                      ^                              |
                                      +---- prepared/forwarded ------+
                                                                     |
                              observable host render <---------------+
                                           |
                              DISPLAY_READY or terminal accepted result
```

Important distinctions:

- TCP bound is not HTTP ready.
- HTTP ready is not session ready.
- a SID may be declared and exposed as pending while data loads;
- VS Code forwarding may begin once the port is reserved, and an adapter may
  prepare a native process before HTTP readiness if the mechanism supports
  preparation without navigation;
- presenting or navigating to the backend waits for the adapter's declared
  shared prerequisites, normally `BACKEND_HTTP_READY` and `SID_DECLARED`;
- display `accepted` is not display `ready`;
- a host-side `frame-rendered` event is the strongest readiness evidence where
  the adapter can observe it. System-browser dispatch, SSH guidance, and
  no-display have different terminal outcomes and must not claim first-frame
  readiness;
- no adapter may choose a fallback after its deadline. It returns a typed
  failure and the orchestrator applies the planned fallback.

Each adapter declares its minimum shared prerequisite and preparation support.
For example, native process preparation may overlap backend startup, navigation
waits for `BACKEND_HTTP_READY`, tab/session notification waits for
`SID_DECLARED`, and readiness waits for a host-side rendered-frame event where
available. This keeps the current cold-start overlap without relying on whether
a 20 ms TCP poll happens to beat uvicorn initialization.

## Server reuse and startup transaction

Server acquisition must be atomic at the scope where competing launches can
race:

1. acquire a startup lock keyed by runtime namespace and the intended
   ownership/compatibility/port scope, so unrelated launches do not serialize;
2. re-read registry and port state;
3. validate process identity;
4. query `/ping` or `/status`;
5. compare protocol and advertised capabilities with `SessionSpec`;
6. reuse, select another port, or start a new owned instance according to the
   plan;
7. for a new process, pass a startup manifest and nonce through a supported
   internal entry point rather than an opaque `python -c` string;
8. wait for an HTTP identity response matching nonce, instance ID, PID, port,
   and protocol;
9. release the startup lock after the server decision is committed.

Never decide compatibility from `service=arrayview` alone. Never kill a process
using a port as the only proof of ownership. An incompatible but healthy
ArrayView server is not “stale”; use another owned instance or return a typed
compatibility error according to the plan.

## Migration rules

1. Migrate vertical slices, not layers across every invocation at once.
2. The first slice is the exact recent failure:
   `arrayview FILE --window native` from a local VS Code terminal, new daemon.
3. A slice includes normalization, facts, plan, server acquisition,
   registration, display opening, fallback, cleanup, and real-process evidence.
4. Delete or make unreachable the old execution branch for that slice in the
   same PR. Do not maintain two active authorities for a migrated slice.
5. No new contract type lands without a production caller.
6. No production migration is declared complete while its real display boundary
   is only mocked.
7. Do not split `_launcher.py` merely to move code. First invert dependencies
   and migrate callers; physical extraction follows proven boundaries.
8. During migration, old and new planning may run in shadow and compare data,
   because that is side-effect free. Old and new executors must never both run
   for one request.
9. A temporary internal executor switch is allowed for rollback, but every
   migration PR must name its removal gate and deadline. It cannot become a
   permanent compatibility mode.
10. Keep lazy imports and measure the fast path in every phase.

## Phased implementation

Each phase should be a separate branch/PR with its own evidence bundle. A phase
does not start by editing production routing; it starts by making its acceptance
test fail for the right reason.

Phases 0–2 are a proof of the architecture, not automatic authorization for the
rest of the program. Stop after Phase 2 and review the evidence before approving
Phase 3.

### Phase 0 — Reproduce, isolate, and establish the coverage ledger

Deliverables:

- Create an invocation ledger listing every supported entry point, placement,
  ownership strategy, registration strategy, default display, explicit display
  overrides, fallback, and current production function.
- Mark every row `legacy`, `migrated`, or `unsupported`; never use “mostly.”
- Make lifecycle tests hermetic:
  - isolated runtime/registry directory;
  - isolated HOME/XDG paths where VS Code files are involved;
  - dynamically assigned ports;
  - scrubbed VS Code/SSH/Jupyter environment;
  - process cleanup registered before spawn.
- Fix tests whose result changes when a developer has a server on port 8000.
- Add a read-only JSONL launch trace with a correlation ID. Initially it records
  current behavior without changing it.
- Pair traces with external observations: process tree, server identity, SID,
  actual signal-file activity, and host-side display/frame evidence where the
  real adapter is under test. A trace alone is not proof of what opened.
- Record startup baselines for:
  - tiny `.npy`, local system browser;
  - tiny `.npy`, local native on a real supported host;
  - `debug/parameter_maps.nii`, local native from VS Code terminal (host-only;
    this large file is not a clean-checkout CI fixture);
  - local VS Code protocol integration and, separately, a real Extension
    Development Host panel;
  - existing compatible server reuse.
- Document current known failures rather than normalizing them as expected
  behavior.

Exit gates:

- The lifecycle test result is identical with and without a real daemon running
  on the developer's default port.
- Every test-created process is reaped and every test-created settings/runtime
  file lives under a temporary directory.
- The exact NIfTI command has a saved trace plus process-tree and external
  evidence distinguishing a native process/frame from a VS Code signal/panel.
- A generated deterministic slow-registration fixture or phase barrier
  reproduces the same ordering on a clean checkout.
- The ledger has no unowned production branch.

### Phase 1 — Make intent and policy complete

Deliverables:

- Define `LaunchRequest` and explicit display/fallback enums, then feed them to
  planning, diagnosis, and tracing without changing legacy execution yet.
- Separate physical native capability from auto-display preference.
- Split stable `HostEvidence` from dynamic `DiscoverySnapshot`; add
  evidence/unknown states and cache invalidation rules.
- Extend `LaunchPlan` so all fallback, retry, lifecycle, and capability decisions
  are serializable.
- Make `--diagnose` emit the same request/host/discovery/plan objects used by a
  real launch.
- Add module dependency guards that prevent display adapters from importing one
  another or the planner from importing heavy mechanisms.
- Add a shrinking legacy-call allowlist tied to migrated ledger rows. Do not ban
  private calls globally until their row migrates; broad AST pinning would
  recreate the old refactor's module-coupled tests.

Exit gates:

- Every explicit intent survives plan serialization without being collapsed to
  the selected mechanism.
- Diagnosis and real launch planning are byte-equivalent after volatile facts
  are removed.
- No new test-only policy helper is added.
- Import-time benchmark stays within the Phase 0 budget.

### Phase 2 — Introduce the thin orchestrator through one vertical slice

Scope only:

```text
CLI file + local machine + local VS Code terminal + explicit native
+ new transient daemon + system-browser fallback
```

Deliverables:

- Add the phase orchestrator and structured cleanup stack.
- Use the existing minimal `SessionSpec` for this production slice, with its
  file source supplied through `SourceBindings`; do not introduce another
  parallel registration request.
- Adapt existing subprocess server, file registration, native opener, and
  system-browser opener behind typed strategy interfaces.
- Add explicit phase barriers for tests: before HTTP ready, after SID declared,
  while data is loading, before native ACK, and before fallback.
- Migrate this slice end to end.
- Remove the legacy branch for this slice.
- Ensure the native adapter never imports or invokes VS Code behavior.

Required tests:

- exact `debug/parameter_maps.nii` command on macOS as host-only regression
  evidence, plus a clean-checkout deterministic slow-registration fixture;
- built-wheel public CLI through a real daemon, HTTP/session registration, and
  an external display boundary. Do not monkeypatch the planner, orchestrator,
  process creation, routes, or signal writer in this gate;
- delayed HTTP readiness;
- delayed data readiness with early shell accepted;
- serialized native attempt budget and fallback sequence are asserted exactly:
  mechanism-local retries are distinct from advancing to system browser;
- explicit native under `TERM_PROGRAM=vscode` creates zero files in an isolated
  observed VS Code signal directory;
- native child crash is reaped;
- 100-run deterministic transition soak with fake mechanisms, reported only as
  orchestrator coverage;
- 100-run real-process handoff soak using the real CLI/server/registration and
  an external recording display boundary, with process-identity/tree checks;
- a smaller repeated real-PyWebView host gate on every OS whose production
  branch is migrated. If Phase 2 migrates only macOS, keep other OS ledger rows
  on legacy execution until their host gate passes.

Exit gates:

- The trace has one server decision, one session identity, and at most one
  visible display attempt at a time. Failed attempts are positively cleaned
  before fallback; background preparation/forwarding may still overlap backend
  and data phases.
- The selected fallback is read from `LaunchPlan`, not reconstructed from the
  environment or a string comparison.
- Import and compatible-server reuse regress by no more than the tighter Phase 0
  fast-path budget (provisionally 20 ms at p95). Cold display acceptance/first
  frame regress by no more than 10% or 100 ms at p95, whichever is larger,
  unless explicitly approved with evidence.

### Phase 2 go/no-go review

Continue only if the first vertical slice demonstrates all of the following:

- the orchestrator stays mechanism-free and near the proposed size;
- the production request reaches it without being projected back into legacy
  booleans;
- the old branch for the slice is gone, not hidden behind a second fallback;
- a black-box trace and external evidence agree on the actual server, SID,
  adapter, and cleanup outcome;
- phase barriers reproduce the former race deterministically;
- the partial-order readiness model preserves or improves startup latency;
- the architecture test prevents reintroducing a second router;
- a reviewer can describe the complete slice from request to cleanup without
  tracing unrelated invocation branches.

Stop and redesign rather than proceeding if:

- mechanism-specific logic accumulates in the orchestrator;
- native, browser, or VS Code adapters need to detect and invoke one another;
- correctness still depends on arbitrary sleep duration;
- the new path must leave the old executor authoritative;
- real evidence cannot distinguish accepted from ready;
- the performance budget is missed without a clearly measured tradeoff;
- the slice adds abstractions but deletes no competing production authority.

### Phase 3 — Converge server acquisition and compatibility

Deliverables:

- Add `ServerRequest`/`ServerHandle` and ownership strategy interfaces.
- Make registry discovery, startup locking, identity verification, and
  capability negotiation one transaction.
- Populate planner facts with protocol, package version, owner scope, and
  advertised capabilities.
- Replace service-name-only reuse with required-capability checks.
- Add a supported daemon startup manifest/entry point and retained process/log
  identity.
- Migrate CLI absent-server, compatible-server, incompatible-server, foreign
  port, and simultaneous-launch cases.
- Remove corresponding CLI port/server branches after each row migrates.

Exit gates:

- 100 independent-process acquisition attempts create exactly one compatible
  server for the intended scope; every launch succeeds with a distinct valid
  session/lease, and final cleanup leaves no process.
- Compatible instances are reused; incompatible ones are never mutated or
  killed; foreign listeners are untouched.
- A server dying between snapshot and registration produces a bounded typed
  recovery, not a silently different plan.
- Protocol fixtures pass on every relevant PR, and actual pinned previous-wheel
  versus current-wheel black-box interop passes for the migration PR and release
  wherever compatibility is promised.

### Phase 4 — Complete CLI `SessionSpec` migration

Deliverables:

- Extend `SessionSpec` only where current supported semantics require it:
  directory collections, overlay names/options, stacking policy, and any other
  currently missing launch inputs.
- Make all migrated CLI registration strategies accept `SessionSpec` plus
  `SourceBindings` and return `RegisteredSession`.
- Centralize required capability derivation.
- Preserve pending-session behavior so forwarding/display startup can overlap
  loading.
- Compare session metadata across new-daemon startup and existing-server HTTP
  registration.
- Remove duplicated CLI feature validation after parity is proven.

Exit gates:

- Equivalent CLI requests yield equivalent session graphs, metadata, and
  capabilities across existing and new server topology. Identifiers remain
  distinct valid random SIDs.
- File keys, RGB, compare, overlays, vector fields, watch, and directory
  collections either work identically or fail before side effects with a typed
  capability error.
- `SessionSpec` has real production callers and its old parallel argument lists
  are removed for migrated paths.

### Phase 5 — Complete local CLI display migration

Migrate, in this order:

1. explicit `none`;
2. explicit system browser;
3. automatic native/system-browser local behavior;
4. explicit native with existing server;
5. overlays/compare paths that cannot use native shell injection;
6. local VS Code terminal and Explorer.

Deliverables:

- Split routing out of `_open_browser()`. The system-browser adapter only opens
  the system browser; the VS Code adapter only speaks the VS Code protocol; SSH
  guidance is its own adapter/result.
- Make every adapter return accepted versus ready explicitly.
- Apply fallback only in the phase orchestrator.
- Migrate CLI rows and delete their legacy opener branches.

Exit gates:

- No CLI entry path calls the legacy routing opener.
- Explicit native can never become VS Code unless the serialized plan explicitly
  says so for remote placement.
- The emitted request sequence exactly matches the serialized adapter retry and
  fallback chain, failed visible attempts are cleaned before the next adapter,
  and there is exactly one terminal success.
- A built-wheel public CLI/Explorer request is claimed by a real local VS Code
  Extension Development Host, its panel reports a host-side rendered frame, and
  panel disposal releases the SID. Real native host-gated tests pass before
  deleting the corresponding CLI compatibility switch.

### Phase 6 — Migrate Python and Jupyter without flattening ownership

Deliverables:

- Before migration, pin the current short-script attachment lifetime as an
  explicit compatibility contract or make a separate product decision. Phase 6
  is blocked until that duration/condition is no longer ambiguous.
- Normalize Python API input into `LaunchRequest` once.
- Migrate Python existing-server and in-process registration to `SessionSpec`.
- Preserve script versus long-running-process ownership semantics.
- Preserve kernel-owned Jupyter server reuse and iframe semantics.
- Run planner facts only once; runtime strategies revalidate resources but do
  not call environment routing helpers.
- Remove Python-specific fallback policy and boolean reconstruction.

Required black-box tests:

- a real temporary Python script calls `view()`, exits, and leaves the viewer
  alive only for its defined attachment window;
- closing the last viewer reaps transient resources;
- a real ipykernel executes two cells and reuses the kernel-owned server;
- iframe disappearance does not kill the kernel-owned server;
- explicit handle close releases only its session;
- explicit native from a local VS Code Python script follows the same contract
  as CLI.

Exit gates:

- Python and CLI with equivalent intent produce equivalent serialized display
  and fallback policy, while retaining their distinct ownership scopes.
- `view()` no longer re-probes `_can_native_window()` or `_is_vscode_remote()`
  after planning.

### Phase 7 — Migrate Julia, MATLAB, and plain SSH

Deliverables:

- Keep Julia/PythonCall out of process and preserve the IJulia side-effect
  display requirement.
- Model MATLAB desktop/batch placement explicitly rather than as a detector
  exception.
- Give plain SSH its own display/guidance adapter; never open a browser on the
  remote host accidentally.

Exit gates:

- Pinned Julia/PythonCall runs as a required CI job, reaches a host-observable
  viewer outcome without GIL hang, and cleans up.
- A real IJulia run proves the side-effect display path when IJulia remains a
  supported ledger row.
- MATLAB desktop and batch have dated real-host evidence, or their ledger rows
  are explicitly marked unsupported before legacy deletion.
- Plain SSH is tested through a real local `sshd`/`ssh -L` harness.

### Phase 8 — Migrate remote VS Code and tunnel reconciliation

Deliverables:

- Make VS Code local/remote use the same request/ACK contract while retaining
  remote forwarding/reconciliation mechanics in the extension adapter.
- Ensure exact window identity, forwarding state, external URI, server ID, SID,
  and host-side frame evidence share one correlation ID.
- Remove environment-driven fallback from extension/browser helpers.

Exit gates:

- Local VS Code protocol tests cover two windows, stale target, concurrent
  requests, ACK loss, and extension restart.
- A built-wheel public entry point through a real Extension Development Host
  passes local panel open/frame/dispose/release; extension-only tests are labeled
  component coverage.
- A dated real tunnel handoff verifies two client windows, forwarding privacy,
  external URI reachability, host-side rendered frame, extension-host restart,
  and tunnel reconnect. Evidence includes trace, screenshot/video where
  applicable, and cleanup/process-tree results.

### Phase 9 — Lifecycle convergence and legacy deletion

Deliverables:

- Attach explicit lease/cleanup handles to every successful display.
- Ensure every spawned process is supervised and every temporary file has one
  owner.
- Delete unused resolver helpers and tests that exercise no production code.
- Delete the temporary executor switch.
- Shrink the architecture allowlist to its final form.
- Only now extract physically coherent modules from `_launcher.py`, if doing so
  reduces coupling. Movement alone is not a deliverable.
- Update lifecycle/architecture documentation to match actual production calls.

Final source-boundary invariants:

- environment detection is called only while building facts;
- fallback selection is read only from `LaunchPlan`;
- display adapters do not import or call one another;
- entry adapters do not probe ports, processes, or displays;
- all launch-time registration starts from `SessionSpec` plus `SourceBindings`;
  in-viewer drop/import routes remain outside this migration unless explicitly
  brought into the ledger;
- server reuse checks identity, protocol, and required capabilities;
- no launch subprocess is started through an opaque Python code string;
- no dead policy helper remains under test;
- no migrated invocation can enter legacy execution.

## Verification architecture

### 1. Pure policy tests

Keep the cross-product planner matrix, but assert complete plans rather than
only final display values:

- ownership scope;
- registration strategy;
- primary/fallback adapters;
- prerequisites;
- capabilities;
- retry/recovery budget;
- reasons.

These tests are fast and cross-platform. They do not count as execution
coverage.

### 2. Deterministic orchestrator contract tests

Use injected strategies and a fake monotonic clock to cover every transition
and failure edge. These are the only tests allowed to replace server/display
mechanisms with fakes. They must assert emitted `LaunchEvent`s, cleanup order,
and that fallback follows the plan.

Fault points include:

- port changes after snapshot;
- process exits before HTTP readiness;
- SID remains pending;
- registration loses the server;
- native shell accepts but never reaches first frame;
- opener returns unsupported;
- VS Code ACK has the wrong server/window/request ID;
- fallback fails;
- caller cancellation at every phase.

### 3. Hermetic black-box process tests

Run the built wheel through its real public entry point in a fresh subprocess.
Do not monkeypatch the planner, environment detectors, `Popen`, HTTP routes, or
registration. Each test uses:

- temporary HOME/XDG/runtime registry;
- scrubbed and explicitly constructed environment;
- dynamic port reservation;
- real server process and HTTP/WebSocket connections;
- trace/stdout/stderr/process-tree artifact capture;
- cleanup registered before process creation.

A pass needs both internal and external evidence:

- phase trace with invocation, facts, intent, server ID, SID, adapter, and
  correlation IDs;
- external `/status`, `/metadata/{sid}`, host-side `frame-rendered` evidence
  (Playwright canvas/parent message, native shell ACK, or VS Code panel ACK)
  where observable, adapter acceptance evidence, and expected process
  survival/death. Backend metadata or image bytes alone do not prove a rendered
  frame.
- system-browser command recording proves only OS URL dispatch. Pair the same
  URL with Playwright for viewer readiness and report the two results
  separately; do not label dispatch as real system-browser frame readiness.

### 4. Protocol integration tests

A separate Node fake extension host may consume the real signal-file protocol,
perform live backend/session/frame checks, and write real correlated ACKs. This
can validate concurrency, filesystem atomicity, targeting, and recovery in CI.
It must be reported as “VS Code protocol integration,” not “real VS Code.”

Remote protocol tests may add a forwarding proxy that returns a non-loopback URL
and simulate reconnect. They still do not replace a real tunnel gate.

### 5. Required CI and host-gated automated tests

Required CI where the dependencies are installable:

- real ipykernel with MIME output and Playwright viewer readiness;
- pinned Julia/PythonCall process;
- real local `sshd`/`ssh -L` plain-SSH workflow.

Required host-gated evidence before deleting the old executor for the relevant
slice:

- real PyWebView on macOS and Windows; Linux under a supported Xvfb/backend;
- built-wheel public CLI/Explorer through a real local VS Code Extension
  Development Host via `@vscode/test-electron`;
- real Remote-SSH on a configured/self-hosted runner where practical;
- real IJulia and MATLAB host runs for supported ledger rows.

### 6. Manual pre-release evidence

Only behavior that genuinely requires the hosted VS Code Tunnel service or
unstable desktop window-manager observation may remain manual. Manual rows must
produce a dated artifact containing commit, versions, environment, commands,
request IDs, logs, result, and cleanup. A permanent green `MANUAL` label is not
evidence.

## Required scenario matrix

| Scenario | Evidence class | Minimum assertion |
|---|---|---|
| CLI local, native | real host | host-side frame; close reaps transient server |
| CLI local, explicit browser | OS-opener integration + Playwright | system-browser dispatch only; viewer URL renders; no VS Code request |
| CLI in local VS Code, auto | protocol CI + real extension host | exact window and host-side frame ACK |
| CLI in local VS Code, explicit native | real native host | native or declared system-browser fallback; zero VS Code request |
| Python short script | black-box process | pinned attachment lifetime; no orphan after viewer close |
| Python long process | black-box process | handle/session release does not affect siblings |
| Jupyter | required real-ipykernel CI | kernel reuse across cells; iframe loss does not kill server |
| Julia terminal | required Julia/PythonCall CI | subprocess isolation and correct browser semantics |
| IJulia | real host or explicit unsupported row | side-effect inline display and cleanup |
| MATLAB desktop/batch | dated real host or explicit unsupported row | no false VS Code classification; correct ownership |
| Plain SSH | required real-sshd CI | `localhost` forwarding guidance; no remote browser |
| VS Code Explorer/terminal | protocol CI + real extension host | originating window, first frame, disposal, version compatibility |
| VS Code tunnel, multiple windows | protocol CI + dated real tunnel | exact target, forwarded URL, frame, reconnect/restart recovery |
| Existing compatible server | independent-process black box | reuse with equivalent registration graph/metadata |
| Existing incompatible server | independent-process black box | separate instance or typed error; never mutate/kill |
| Foreign listener | independent-process black box | untouched; alternate port or typed fixed-port error |
| Simultaneous launches | 100 independent public CLIs | exactly one backend, all sessions valid, complete cleanup |
| Server dies after discovery | fault-controlled black box | bounded revalidation/recovery within one policy |
| Native startup race | 100 real-process recording-boundary runs + smaller real GUI gate | no double-open, wrong adapter, or orphan |
| Current/previous version skew | protocol fixtures + actual pinned wheels/VSIX | behavior matches explicit compatibility table |

## Performance and safety gates

Before each migration PR merges:

- import-time and CLI fast-path benchmarks stay within the recorded budget;
- p50 and p95 time to display accepted and first frame are reported separately;
- large-file load remains concurrent with display/forwarding where allowed by
  phase prerequisites;
- fault soak produces zero orphan processes and zero persistent test files;
- no unexpected adapter request, double-open, dead URL, or unbounded timeout;
- no server is reused without required capabilities;
- no foreign/incompatible process is killed;
- `localhost` remains the public loopback spelling;
- all affected ledger rows are green at the proper policy, process, protocol,
  and host-gated levels.

Do not merge the migration slice, or revert it if already merged, when any of
these gates fail. A temporary switch may exist only before the gates pass and
with a named deletion deadline. Do not patch around a failing gate in a
downstream adapter.

## Pull-request sequence

Keep reviews small enough to connect behavior to evidence:

1. **Evidence and isolation** — coverage ledger, hermetic runtime, trace,
   baselines; no routing changes.
2. **Intent and plan completion** — data contracts and architectural guards;
   no execution changes.
3. **Explicit-native vertical slice** — thin orchestrator plus exact regression
   and soak evidence; delete old slice.
4. **Server transaction** — identity/capability acquisition and race tests.
5. **CLI `SessionSpec` migration** — registration parity and capability gates.
6. **CLI display completion** — local browser/native/VS Code adapters and old
   CLI routing deletion.
7. **Python/Jupyter migration** — ownership-specific strategies and real kernel
   tests.
8. **Julia/MATLAB/SSH migration** — bridge and foreign-host tests.
9. **VS Code remote/tunnel migration** — protocol, host-gated, and real tunnel
   evidence.
10. **Lifecycle cleanup and legacy deletion** — remove switch/allowlist/dead
    tests, then consider physical module extraction.

No PR should combine several public invocation migrations merely because they
share a helper.

## Risks and mitigations

### Recreating a monolith under a new name

Mitigation: the orchestrator only manages phases, fallback, correlation, and
cleanup. Ownership, registration, and displays remain separate strategies. Set
a size/review threshold and reject mechanism-specific imports in the
orchestrator.

### Losing intentional startup concurrency

Mitigation: implement the partial-order prerequisites before migration and
measure display accepted and first frame independently. Never impose a global
`SESSION_READY` barrier.

### Permanent dual execution paths

Mitigation: ledger every slice, allow planning-only shadow comparison, and
delete the old branch in the same PR that migrates a slice. Every temporary
switch has a named deletion gate.

### Tests passing against fakes while real handoffs fail

Mitigation: mocked tests prove policy and transition handling only. Old-path
deletion requires black-box and host-gated evidence for the affected adapter.

### Startup/import regression

Mitigation: keep contracts dependency-free, preserve lazy imports in strategies,
and enforce Phase 0 performance budgets per PR.

### Breaking Jupyter or Julia by over-unifying

Mitigation: share contracts, not ownership/display mechanics. Keep kernel reuse,
IJulia side effects, and Julia subprocess/GIL isolation as explicit strategies
with real host tests.

### Version-skew surprises

Mitigation: derive capabilities from `SessionSpec`, check them before reuse, and
maintain a current/previous compatibility matrix using protocol fixtures plus
required pinned-wheel tests. Include old/new Python-package and VSIX combinations
where cross-version compatibility is promised.

### Refactor blocked by monkeypatch-heavy tests

Mitigation: add dependency seams and black-box harnesses before moving files.
Delete tests of dead helpers; test public contracts and emitted events instead
of module-local function placement.

## Decisions deliberately deferred

The refactor must preserve current behavior until these are decided separately:

- whether CLI and Python should share one default port;
- how long short Python scripts should keep an unattached viewer alive;
- which old server protocol versions remain reusable;
- whether automatic public VS Code tunnel forwarding requires additional viewer
  authentication;
- whether persistent tunnel servers should be default or opt-in.

Encoding any of these as an incidental refactor choice would recreate the
current ambiguity.

For this migration, preserve the current documented explicit-native contract:
local native is attempted and system browser is its fallback; remote placement
may select VS Code only when the serialized plan explicitly says so.

## Definition of done

This project is complete only when all of the following are true:

- every supported invocation row is `migrated` or explicitly `unsupported`;
- diagnosis and execution use the same request, facts, and plan;
- explicit caller intent is never reconstructed from a selected adapter;
- the planner is the only fallback authority;
- execution revalidation cannot silently create a different plan;
- all launch-time registrations start from `SessionSpec` plus `SourceBindings`
  and return `RegisteredSession`;
- server reuse requires identity, protocol, and capabilities;
- every process/session/display has a cleanup owner;
- real black-box CLI, Python, Jupyter, Julia, SSH, native, and VS Code-local
  evidence is green; supported IJulia and MATLAB rows also have real evidence or
  are explicitly marked unsupported;
- real tunnel evidence is attached for the release candidate;
- the fault/race soak reports zero double-opens, wrong-adapter events, and
  orphans;
- the legacy executor, routing helpers, fallback branches, temporary switch,
  and tests of dead policy are deleted;
- `_launcher.py` may still be large, but it no longer owns competing launch
  policies. Any later file split is mechanical and low risk.

Until those conditions are met, the work should be described as partial
migration, not “robust launch lifecycle complete.”
