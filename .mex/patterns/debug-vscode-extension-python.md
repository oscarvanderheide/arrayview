---
name: debug-vscode-extension-python
description: Diagnose VS Code opener failures across local, Remote SSH, and tunnel windows without mistaking component health for a rendered viewer.
triggers:
  - "ArrayView failed to open"
  - "vscode extension"
  - "remote tunnel"
  - "asExternalUri"
  - "port public"
edges:
  - target: patterns/validate-launch-path.md
    condition: always, to define the real invocation and acceptance evidence
  - target: context/conventions.md
    condition: when making code changes after confirming the failure boundary
last_updated: 2026-07-23
---

# Debug VS Code Extension Opening

Treat local VS Code, Remote SSH, and VS Code tunnel as different launch rows. A
process, forwarded port, panel, or successful `/ping` is not proof that the
array rendered.

## Before Running

- Record the exact public command and the VS Code window where it should open.
- Run checkout code with `uv run arrayview`, not `uvx arrayview`. The latter may
  select a released package and a different bundled opener.
- Warn before any step that can open windows or prompts.
- Do not create a temporary or isolated VS Code profile as acceptance evidence;
  it changes the host and may trigger account or keychain prompts. Obtain
  explicit permission before installing into or reloading the active profile.
- Check the live extension registration/version. Source, bundled VSIX,
  installed extension, and running extension host can all differ.
- Record `vscode.env.remoteName`, the launching terminal's IPC hook and process
  ancestry, the exact VS Code server root, extension placement, and every live
  ArrayView registration. Do not select a CLI, installation, or registration
  merely because it has the newest timestamp or version.

## Trace One Request End To End

1. Inspect the extension log:

   ```bash
   tail -n 160 "$HOME/.arrayview/extension.log"
   ```

   Correlate one request by `requestId`. Current useful markers include
   `SIGNAL`, `REMOTE`, `PORT`, `PANEL`, and `ACK`.

2. Inspect the protocol-v1 signal payload. It should identify the action, URL,
   request ID, ACK path, protocol version, server, and requested window. Do not
   expect the removed `mode: "url"` field.

   After a reload, `ARRAYVIEW_WINDOW_ID` in an existing terminal may be stale.
   Accept recovery only from an exact live registration or a unique match to
   the active VS Code server root. If competing windows remain possible, fail
   closed instead of broadcasting or guessing from focus.

3. Check the local backend using the actual signaled port. `/ping` proves only
   reachability; `/metadata` for the requested SID proves that the intended
   session exists.

4. Classify the remote URL route before judging it:

   - Desktop tunnel with VS Code's integrated-browser remote proxy enabled:
     the integrated browser may open the loopback backend URL directly. A
     private entry in the Ports view does not make this route invalid.
   - Tunnel without that proxy: configure forwarding, perform supported public
     visibility promotion, and require a verified non-loopback `asExternalUri`
     URL that retains the SID query parameters. Account for the developer-tunnel
     trust/consent page rather than mistaking it for the viewer.
   - Remote SSH: a local forwarded loopback URL may be correct. Do not run
     tunnel-only public-port commands or policy against it.

   In every case, verify that the selected route belongs to the expected server.

5. Follow the visible readiness chain: wrapper loaded, viewer script loaded,
   WebSocket connected, metadata received, `frame-rendered` emitted,
   `visibility_verified` passed, then terminal `backend_ready` ACK correlated to
   the exact request, window, server, SID, extension instance, and extension
   version. Stop at the first missing boundary instead of refactoring later
   layers.

6. Close the panel and confirm release, then repeat the same command. If request
   multiplexing, shared-server reuse, or cleanup changed, open five displays,
   close a middle one, confirm the others remain live, then close all and verify
   SID, process, registration, and port cleanup. For a persistent remote route,
   also verify a bounded reconnect and eventual idle cleanup.

## Packaging Checks

If extension source changes, keep these synchronized:

- `vscode-extension/package.json`
- `_VSCODE_EXT_VERSION` in `src/arrayview/_vscode_extension.py`
- `src/arrayview/arrayview-opener.vsix`

Then separately verify the installed version and the version reported by the
live extension host.

A missing installed `.vsix_hash` is a cache miss, not proof that reinstall is
required. Compare installed content with the bundle, ignoring only known VS
Code-added package metadata; if equivalent, backfill the marker without forcing
a reload. When installation is required, use the exact active host, preserve
other installations, and wait for that exact window to report the required
live version before sending a request.

After rebuilding the Python distributions, inspect archive size and membership.
Do not accept a successful build that includes untracked datasets, editor state,
caches, or repository tooling.

## Reporting

Label each result `real host`, `real process`, `component`, or `unavailable`.
State whether a human-visible first frame appeared. Follow
`.mex/patterns/validate-launch-path.md` for the full acceptance contract; a
green test suite or lifecycle matrix with `MANUAL` rows is not a substitute.
