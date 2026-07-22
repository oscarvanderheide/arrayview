---
name: validate-launch-path
description: Real-first acceptance workflow for ArrayView startup, display handoff, ownership, reconnect, and cleanup changes.
triggers:
  - "launch"
  - "startup"
  - "window"
  - "browser opening"
  - "VS Code tunnel"
  - "orphan process"
last_updated: 2026-07-23
---

# Validate a Launch Path

Use this pattern for startup, registration, display routing, environment
detection, forwarding, and shutdown work.

## 1. Define the affected row

Record all three dimensions:

- Entry point: CLI, Python script, interactive Python, ipykernel/Jupyter,
  Julia/PythonCall, IJulia, MATLAB, or VS Code Explorer.
- Placement: ordinary local shell, local VS Code, Remote SSH, VS Code tunnel,
  or plain SSH.
- Display: auto, native, browser, VS Code panel, inline, or none.

Do not combine local VS Code, Remote SSH, and tunnel into one row.
Affected rows include the user's reported row, every row selected by the
changed detection/routing branch, and nearby rows that share the changed
ownership or fallback. Name any supported row excluded from real validation
and why.

## 2. Preserve the real reproduction

Before a broad edit, run the smallest public command in the actual target host.
Prefer the working tree while developing:

```bash
uv run arrayview <file> --window <mode>
```

Use `uv run`, not `uvx`, for branch validation. `uvx` may run published Python
code and a bundled extension that do not match the checkout. If the plain user
command fails, preserve it unchanged as the acceptance gate and make a separate
diagnostic run when needed:

```bash
ARRAYVIEW_LAUNCH_TRACE=/tmp/arrayview-launch.jsonl \
  uv run arrayview <file> --window <mode> --verbose
```

The trace variable is optional instrumentation; it must not become a user
requirement.

For Python, notebook, Julia, MATLAB, or SSH, use the public entry point the user
actually uses. Save the exact command unchanged as the post-fix gate.

Warn the user and obtain permission before opening a GUI, installing an
extension into an active profile, or reloading an IDE window. Do not create a
temporary or isolated VS Code profile as acceptance evidence; it changes the
host being diagnosed and may trigger account or keychain prompts.

## 3. Define success before editing

Require every applicable result:

- The explicit display choice is honored.
- The request produces one intended display outcome: either one new display or
  an explicitly designed reuse of the identified existing display, never an
  accidental sibling window or duplicate.
- The requested array reaches its first rendered frame. A process, port, tab,
  iframe load, `/ping`, or WebSocket connection is not enough.
- No-display mode completes registration without opening anything.
- The caller returns, blocks, or stays kernel-owned as promised.
- Closing the display releases only its sessions and owned transient process.
- A second identical launch succeeds without manual reload or retry.
- Remote/reconnect rows recover within the configured connect/idle bounds from
  `.mex/context/lifecycle.md`; record the effective values used.

If the change touches shared-server reuse, request claims, panel creation, or
session cleanup, launch five displays. Close one in the middle, verify the
remaining displays still work, then close all of them and prove SID/process/port
cleanup. This is the gate against accidental singleton behavior and failures
that appear only after several calls.

## 4. Diagnose the earliest failed boundary

Correlate one launch through:

1. captured request and host facts, including competing host registrations;
2. server identity and ownership;
3. session registration and exact SID;
4. display request and intended target;
5. document load and transport connection;
6. first rendered frame;
7. terminal success acknowledgement;
8. display close, session release, and process cleanup.

Fix the earliest failed boundary. Do not use a later fallback to hide it, and do
not begin a large refactor while the simple reproduction remains unexplained.

For VS Code, host facts include `vscode.env.remoteName`, the terminal IPC hook
and process ancestry, the exact active server root, extension host
registration/version/instance ID, local versus remote extension placement, and
all ArrayView registrations that could claim the request. Never use install
recency or the newest filesystem match as host identity.

## 5. Rerun real evidence first

After each meaningful change:

1. rerun the preserved public command in the same host;
2. verify the complete success contract;
3. repeat it once;
4. close it and verify cleanup;
5. then add a regression test for the proven cause.

When five-launch stress applies, it replaces the single repeat in step 3.

Run nearby sentinels whose policy or ownership could regress. A tunnel fix, for
example, normally needs local VS Code plus ordinary local terminal checks; it
does not require pretending that an unavailable MATLAB host was tested.

## 6. Classify evidence honestly

Report every affected row as one of:

- `real host`: actual GUI/kernel/SSH/tunnel boundary and public invocation.
- `real process`: public entry point plus real backend/process/browser
  automation, without the external host boundary.
- `component`: mocks, fixtures, unit tests, and source assertions.
- `unavailable`: host not accessible.
- `failed`: real gate did not satisfy the success contract.

Only `real host` closes host-dependent rows. `tests/lifecycle_matrix.py` may exit
successfully while showing `MANUAL`; those rows remain open.

## 7. Handoff record

Include:

- commit and live package/extension versions;
- environment and exact command;
- trace/log paths and correlation IDs without user data paths;
- observed first-frame evidence (screen capture or correlated
  `frame-rendered`/`visibility_verified` record, plus what was visibly shown);
- repeat/reconnect result;
- close/release/process result;
- adjacent rows checked;
- unavailable or failed rows still open.

For package or extension-install changes, also record archive sizes and inspect
archive membership. A zero exit status from `uv build` is not acceptance if the
sdist contains untracked user data, caches, editor state, or repository tooling.

Never summarize this as “robust everywhere” while a supported affected host row
has only component evidence.
