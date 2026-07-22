---
name: invocation-consistency
description: Use when implementing or diagnosing ArrayView startup, server ownership, display opening, environment detection, VS Code delivery, tunnel forwarding, session handoff, or shutdown. Requires real-first validation across affected CLI, Python, notebook, Julia, MATLAB, VS Code local/remote/tunnel, native, browser, inline, none, and SSH paths without treating mocked tests as host proof.
---

# Invocation Consistency

## Core rule

Prove the user-visible path before trusting the implementation.

For the affected environment, run the smallest real public invocation before
editing and preserve it as the acceptance gate. A helper test, open port, new
process, panel, or WebSocket does not prove that the array appeared.

When validating a working tree, use `uv run arrayview`, never `uvx arrayview`.
`uvx` may execute a released package whose Python code and bundled opener do
not match the branch under test. `ARRAYVIEW_LAUNCH_TRACE` is diagnostic evidence,
not a setup step that users should need for ordinary launches.

Read `.mex/patterns/validate-launch-path.md` for the evidence workflow. For VS
Code delivery failures, also read
`.mex/patterns/debug-vscode-extension-python.md`.

## Evidence classes

Use these labels exactly:

- `real host`: public invocation in the actual GUI/kernel/SSH/tunnel host.
- `real process`: public entry point with real subprocesses, server, sockets,
  and browser automation, but without the actual external host boundary.
- `component`: mocks, protocol fixtures, unit tests, and source assertions.
- `unavailable`: the required host is not accessible. State this explicitly.

Only `real host` closes a host-dependent acceptance row. Component tests may
prevent regressions but never upgrade the evidence label.

## Acceptance contract

For every affected row, verify all applicable outcomes:

1. The caller's explicit display choice remains authoritative.
2. The intended host/window receives exactly one display.
3. The requested array reaches its first rendered frame. For no-display mode,
   registration completes without any display side effect.
4. The caller returns, blocks, or remains kernel-owned as promised.
5. A second launch works without reloads or manual retries.
6. Closing the display releases only its sessions and reaps only its owned
   transient processes.
7. Remote/reconnect paths recover within a bounded transaction and do not open
   in a stale sibling window.

When a change affects request multiplexing, panel creation, shared-server
reuse, or session cleanup, exercise five launches before declaring the path
stable. Close a middle display and prove the others remain usable, then close
the rest and prove final cleanup. There is no acceptable hidden one-panel or
two-launch limit.

If a gate fails, retain the trace and diagnose the earliest failed boundary.
Do not patch a later symptom or start a broad refactor first.

## Invocation rows

Choose affected rows from all three dimensions; do not rely on a fixed "six
paths" list:

- Entry point: CLI, Python script, interactive Python, ipykernel/Jupyter,
  Julia/PythonCall, IJulia, MATLAB, VS Code Explorer.
- Placement: ordinary local shell, local VS Code, Remote SSH, VS Code tunnel,
  plain SSH.
- Display: auto, native, system browser, VS Code panel, inline, none.

Test the changed rows plus nearby sentinels that could regress. If a host such
as MATLAB, IJulia, Windows native, or a real tunnel is unavailable, leave that
row open and say so.

"Affected" includes the reported row, every row selected by changed
detection/routing logic, and rows sharing changed ownership or fallback. List
supported rows deliberately excluded from real validation and why.

## Ownership invariants

- Preserve the launch plan through execution; do not redetect the environment
  inside a display helper and invent a new policy.
- Keep Julia/PythonCall out of the in-process server path.
- Keep notebook servers kernel-owned across output disappearance.
- For in-process script launches, keep the calling process/server alive until
  the viewer closes or the bounded connect timeout expires.
- Keep transient CLI/SSH backends bounded and automatically reaped.
- Reuse a server only after verifying ArrayView identity and required
  capabilities. Never kill a foreign listener.
- Use `localhost` for backend bind URLs and direct viewer URLs. A tunnel's
  externally resolved `asExternalUri` URL is the deliberate exception.

## VS Code invariants

- Local VS Code, Remote SSH, and VS Code tunnel are separate host rows.
- Identify the active VS Code server from the launching terminal's process and
  environment evidence. Never select a server CLI, extension root, or window
  because it is the newest glob match.
- Record `vscode.env.remoteName`, terminal IPC/process ancestry, extension-host
  registration/version/instance ID, extension placement, and all competing
  ArrayView registrations before changing delivery logic.
- A terminal's `ARRAYVIEW_WINDOW_ID` may become stale after reload. Recover the
  target only from an exact live registration or a unique match to the active
  VS Code server root. If multiple windows remain possible, fail closed; never
  broadcast or use focus as a guess.
- Success requires an ACK correlated to the exact request, window, server, SID,
  extension instance/version, and `frame-rendered` event. An older or sibling
  extension host must not be able to complete the transaction.
- `asExternalUri`, port forwarding, panel creation, and WebSocket connection are
  intermediate states.
- Desktop tunnels have two valid URL routes. With VS Code's integrated-browser
  remote proxy enabled, a direct loopback backend URL is valid. Without that
  proxy, require a verified non-loopback `asExternalUri` URL and perform any
  supported visibility promotion before opening it. Do not apply tunnel-public
  rules to Remote SSH, and do not rewrite a valid proxied route merely because
  a port appears private in the Ports view.
- Closing the panel must release its URL sessions through the backend cleanup
  authority.
- Reconnect/reload must not duplicate panels, renew a transaction forever, or
  let a stale extension host claim the request.
- Never use a temporary or isolated VS Code profile/window as acceptance
  evidence; it changes the environment being diagnosed and can trigger account
  or keychain prompts. Obtain explicit permission before installing into or
  reloading the user's active IDE.

## Extension packaging

When any bundled VS Code extension source changes:

1. Bump `vscode-extension/package.json` and
   `src/arrayview/_vscode_extension.py` together.
2. Rebuild `src/arrayview/arrayview-opener.vsix`.
3. Verify source, bundled VSIX, installed version, and live extension-host
   registration separately.
4. Do not infer that a missing local VSIX hash means the installed extension is
   different. Compare installed content, backfill the marker when equivalent,
   and avoid an unnecessary reinstall/reload cycle.
5. Install through the exact active host, preserve unrelated installations,
   and do not signal until that exact window reports the required live version.
6. Obtain permission before installing into or reloading the user's active
   VS Code profile.

After any release build, inspect both archive membership and size. A successful
exit code is not sufficient if an sdist has swept in untracked user data,
editor state, caches, or repository tooling.

## Stop conditions

Stop and report the open evidence row when:

- the required real host is unavailable;
- success was inferred from a port, process, panel, or connection without a
  first frame;
- a fix for one display changes another display's policy or ownership;
- validation would require an unapproved GUI launch, IDE reload, or external
  profile mutation;
- cleanup leaves an owned process, session, request, or forwarded route behind.

Do not call the launch work robust or complete while any supported affected row
has only component evidence.
