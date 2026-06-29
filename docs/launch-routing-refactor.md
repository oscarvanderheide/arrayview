# Launch Routing Refactor

ArrayView currently opens displays through several partially overlapping paths:
CLI, `view()`, Jupyter, Julia, VS Code local, VS Code tunnel, SSH, native
pywebview, and browser fallback.  The same policy is
encoded in multiple places, so fixes in one path can leave another path broken.

This refactor makes launch routing explicit, shared, and inspectable.

## Problem

The brittle behavior comes from policy being split across:

- environment detection in `_platform.py`
- CLI planning in `_launcher.py`
- Python `view()` planning in `_launcher.py`
- VS Code signal handling in `_vscode_browser.py` and `_vscode_signal.py`
- separate daemon, `/load`, and in-process session registration paths

Symptoms seen recently:

- CLI diagnose says native pywebview should open, but runtime falls back to a browser.
- Linux native detection allows GTK systems while native launch forced Qt.
- `view()` and CLI had different native readiness checks.
- Existing-server remote behavior must keep using the registered URL path.

## Target Model

Create a single launch plan used by both CLI and `view()`.

The plan should be data-only:

```python
Invocation = "cli" | "python" | "jupyter" | "julia" | "codex"
Environment = "terminal" | "vscode_local" | "vscode_remote" | "ssh" | "jupyter" | "julia"
Transport = "http" | "none"
ServerOwner = "existing" | "spawned_daemon" | "in_process" | "persistent" | "external"
Display = "native" | "browser" | "vscode" | "inline" | "none"
Registration = "http_load" | "daemon_startup" | "in_process_session" | "relay"
```

Executors may perform work, but should not invent policy.  Browser fallback,
native readiness and remote behavior must be selected by
the plan.

## Phase 1: Snapshot And Diagnose

Add a launch environment snapshot:

```python
snapshot_launch_environment(port, invocation, requested_window)
```

It should capture:

- platform
- relevant environment variables
- config default
- native backend: `qt`, `gtk`, `""`, or `None`
- port busy / ArrayView server alive / server PID / hostname
- VS Code local vs remote/tunnel
- SSH state
- Jupyter and Julia state

`arrayview --diagnose` must print the same snapshot and plan used by real launch.

## Phase 2: Shared Planner

Add:

```python
plan_launch(invocation, request, env) -> LaunchPlan
```

The planner decides:

- transport: HTTP or none
- display: native, browser, vscode, inline, none
- server owner: existing, spawned daemon, in-process, persistent, external
- registration: HTTP load, daemon startup, in-process session, relay
- fallback policy: whether fallback is allowed and where it goes

CLI and `view()` should both call this planner.

## Phase 3: Native Readiness

Use one native readiness primitive everywhere:

```python
open_native_shell(...)
wait_native_ready(shell_socket=True, viewer_socket=True, process_alive=True)
```

Rules:

- process crash is native launch failure
- shell WebSocket means the native window is usable
- viewer WebSocket means the content loaded
- missing viewer WebSocket alone is not native failure
- browser fallback happens only after native process/shell failure

## Phase 4: Remote Existing-Server Fix

Separate these cases explicitly:

- `Display=vscode`, `Transport=http`: open the registered HTTP URL in VS Code
Do not let display-opening helpers implicitly switch an existing registered URL
into a different transport.

## Phase 5: Registration Semantics

Align `/load`, daemon startup, and in-process registration:

- shared file/key selection behavior
- shared RAM guard behavior where possible
- shared RGB/vectorfield/overlay/compare semantics where possible

## Contract Tests

Prefer contract tests over broad visual smoke for launch routing:

- macOS terminal CLI native
- Linux Qt native
- Linux GTK native
- Linux no display browser fallback
- plain SSH guidance
- VS Code local tab
- VS Code remote forwarded HTTP/WebSocket
- VS Code remote existing HTTP URL
- Jupyter inline
- VS Code remote notebook
- Julia subprocess

Execution contracts:

- existing server native shell does not fall back only because viewer socket is late
- spawned daemon native shell does not fall back only because viewer socket is late
- `view()` native shell does not fall back only because viewer socket is late
- remote existing server opens registered URL
- GTK-only Linux passes `gui="gtk"` to pywebview

## Migration Order

1. Add launch data model and environment snapshot without behavior changes.
2. Make `--diagnose` use the snapshot and shared planner.
3. Move CLI planning into `plan_launch()`.
4. Move `view()` planning into `plan_launch()`.
5. Unify native readiness for CLI and `view()`.
6. Fix remote existing-server URL ambiguity.
7. Align `/load`, daemon startup, and in-process registration semantics.

Each step should have focused invocation tests and should avoid unrelated UI or
rendering changes.
