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
last_updated: 2026-07-22
---

# Debug VS Code Extension Opening

Treat local VS Code, Remote SSH, and VS Code tunnel as different launch rows. A
process, forwarded port, panel, or successful `/ping` is not proof that the
array rendered.

## Before Running

- Record the exact public command and the VS Code window where it should open.
- Warn before any step that can open windows or prompts.
- Do not install/reload an extension or create a temporary VS Code profile
  without explicit permission.
- Check the live extension registration/version. Source, bundled VSIX,
  installed extension, and running extension host can all differ.

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

3. Check the local backend using the actual signaled port. `/ping` proves only
   reachability; `/metadata` for the requested SID proves that the intended
   session exists.

4. For a tunnel, verify that the external URL is non-loopback, retains the SID
   query parameters, targets the expected server, and is reachable through the
   public route. Remote SSH may legitimately resolve to a local forwarded URL;
   do not apply tunnel-only public-port rules to it.

5. Follow the visible readiness chain: wrapper loaded, viewer script loaded,
   WebSocket connected, metadata received, `frame-rendered` emitted,
   `visibility_verified` passed, then terminal `backend_ready` ACK. Stop at the
   first missing boundary instead of refactoring later layers.

6. Close the panel and confirm release, then repeat the same command. For a
   persistent remote route, also verify a bounded reconnect and eventual idle
   cleanup.

## Packaging Checks

If extension source changes, keep these synchronized:

- `vscode-extension/package.json`
- `_VSCODE_EXT_VERSION` in `src/arrayview/_vscode_extension.py`
- `src/arrayview/arrayview-opener.vsix`

Then separately verify the installed version and the version reported by the
live extension host.

## Reporting

Label each result `real host`, `real process`, `component`, or `unavailable`.
State whether a human-visible first frame appeared. Follow
`.mex/patterns/validate-launch-path.md` for the full acceptance contract; a
green test suite or lifecycle matrix with `MANUAL` rows is not a substitute.
