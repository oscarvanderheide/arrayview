# Launch execution ledger

Status values are exact: `legacy` means the current executor still owns the
row, `migrated` means the coordinator path owns it, and `unsupported` means the
product does not promise it. This ledger is updated in every launch-reliability
PR.

| Entry point | Placement | Server / registration | Display policy | Current production executor | Status |
|---|---|---|---|---|---|
| CLI file | local terminal, new server | transient daemon / startup args | browser by default; explicit native/browser/none | `arrayview` → `_handle_cli_spawned_daemon` → `_open_cli_spawned_view` | legacy |
| CLI file | local terminal, compatible server | existing / HTTP load | browser by default; explicit native/browser/none | `arrayview` → `_handle_cli_existing_server` → `_open_cli_existing_server_view` | legacy |
| CLI file | local VS Code, new server | shared transient daemon / startup args | VS Code by default; explicit native/browser/vscode/none | `arrayview` → `_handle_cli_spawned_daemon` → `_open_cli_spawned_view` | legacy |
| CLI file | local VS Code, compatible server | existing / HTTP load | VS Code by default; explicit native/browser/vscode/none | `arrayview` → `_handle_cli_existing_server` → `_open_cli_existing_server_view` | legacy |
| CLI file | VS Code Remote / tunnel | persistent daemon or existing / startup args or HTTP load | VS Code forwarded panel; native redirects by declared policy | CLI handlers + `_vscode_browser._open_browser` + signal/ACK helpers | legacy |
| CLI file | plain SSH | transient daemon or existing / startup args or relay | printed forwarded URL / system-browser guidance | CLI handlers + relay branch + `_vscode_browser._open_browser` | legacy |
| CLI `--serve` | any supported host | persistent empty daemon | no display | `arrayview` → `_serve_empty` subprocess | legacy |
| Python `view()` | local process, new server | in-process server / direct session | native by default when available; browser otherwise; explicit overrides | `view` in `_launcher.py` | legacy |
| Python `view()` | compatible server | existing / HTTP load | planned native/browser/vscode/inline/none | `view` → `_load_session_from_filepath` | legacy |
| Python `view()` | Jupyter / notebook | kernel-owned in-process / direct session | inline by default; explicit overrides | `view` + inline iframe helpers | legacy |
| Julia / PythonCall `view()` | Julia process | detached daemon or existing / file startup or HTTP load | browser/VS Code/inline/native by plan | `view` → `_view_julia` → `_view_subprocess` | legacy |
| MATLAB `view()` | MATLAB-hosted Python | in-process or existing / direct session or HTTP load | native/browser by plan | `view` Python path with MATLAB invocation classification | legacy |
| VS Code Explorer / custom editor | extension host | extension-owned subprocess | VS Code custom editor/panel | `vscode-extension/extension.js`; Python planner is advisory only | legacy |
| Codex helper | local Codex browser | persistent replacement daemon / startup args | printed localhost URL | `_codex_open.main` → `_start_loaded_server` | legacy |

## Phase 0 acceptance

- [x] Detailed convergence plan committed separately.
- [x] Every supported production entry point has an owner in this ledger.
- [x] The black-box gate isolates runtime, registry, HOME, VS Code signal files, environment, and ports.
- [x] Ambient-sensitive lifecycle tests pass unchanged with compatible listeners active on both ports 8000 and 8123.
- [x] Opt-in JSONL traces correlate caller, daemon, session, adapter, and cleanup without recording paths or data names.
- [x] A real-subprocess Chromium gate observes the public CLI, server identity, SID readiness, display handoff, first frame, and process cleanup.
- [x] The exact `debug/parameter_maps.nii --window native` case has host evidence showing one native first frame and zero VS Code signal requests.
- [ ] A deterministic slow-registration barrier reproduces the ordering independently of file size.

## Known failures and evidence gaps

- The real local VS Code panel gate is not green yet. A temporary-profile
  Extension Development Host reproduced and isolated the reload livelock, but
  macOS keychain prompts contaminated post-fix first-frame validation.
- Several `view()` tests mock `_launcher` aliases while the planner probes
  `_launch_plan` and `_platform` directly, so ambient ports, VS Code variables,
  config, and native dependencies can change their result.
- IJulia and MATLAB still require host evidence; mocked policy tests are not
  substitutes. The VS Code tunnel first-frame, repeated-launch, native-policy,
  and cleanup gates passed on 2026-07-22, but the phase-level external-URI and
  same-server multi-window gates remain open.

## 2026-07-21 exact native observation

`debug/parameter_maps.nii --window native` was run with local VS Code-terminal
facts and the current trace. The plan selected native and wrote zero VS Code
requests. One detached PyWebView process connected to the exact backend/SID,
reported the correlated first frame, survived CLI exit, and reaped its transient
daemon after the native window closed. The former speculative first process and
double-open path have been removed.

The same exact-frame and cleanup gate also passed from a Python script and from
Julia/PythonCall. A real ipykernel inline render/shutdown and a real temporary
`sshd` plus `ssh -L` browser workflow passed on the same date.

## 2026-07-21 local VS Code panel observation

A real Extension Development Host accepted the exact request and loaded the
viewer. Backend traces exposed a deterministic livelock: the wrapper reloaded
the iframe 1.5 seconds after every load, even after `script-loaded`, repeatedly
disconnecting a healthy viewer before its first frame. Version 0.14.51 cancels
wrapper reloads after document bootstrap while keeping success gated on
`frame-rendered`.

After that change the test host held one stable WebSocket and completed wrapper,
metadata, and title phases. Its fresh temporary macOS profile also raised a
keychain modal, so the absence of a later frame is not counted as a valid
product failure or pass. A same-origin headless iframe rendered the NIfTI and
emitted `frame-rendered`. The remaining acceptance gate is 0.14.51 in an
ordinary existing VS Code window, including tab disposal and daemon cleanup.

## 2026-07-22 VS Code tunnel observation

Opener 0.14.70 was exercised from the existing VS Code tunnel window without a
temporary profile. An explicit VS Code launch and a default/auto launch each
targeted only window `cc23ad5c9220f8b3`, opened one new integrated-browser tab,
and completed `wrapper-started`, `script-loaded`, `ws-open`,
`metadata-loaded`, and `frame-rendered` before a correlated `backend_ready` ACK.
The second launch reused the exact server generation. A simultaneous
Remote-SSH registration running opener 0.14.47 did not claim either request.

Closing both tabs dropped viewer and shell sockets to zero; after the bounded
disconnect grace, both SIDs were released, the daemon exited, its registry
entry disappeared, and port 8000 was free. Explicit `--window native` printed
the declared unsupported-remote warning, redirected to one client-side VS Code
tab, reached the same first-frame ACK, and created no remote GUI.

VS Code 1.128 used its integrated-browser remote proxy with the backend
`localhost` URL directly. This avoided a public developer-tunnel URL and its
consent page, so the older non-loopback external-URI evidence is not claimed;
that phase-level forwarding/privacy gate remains separate from the successful
first-frame and cleanup evidence above.
