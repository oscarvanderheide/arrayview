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
- [ ] The exact `debug/parameter_maps.nii --window native` case has host evidence showing native or declared browser fallback and zero VS Code signal requests.
- [ ] A deterministic slow-registration barrier reproduces the ordering independently of file size.

## Known failures and evidence gaps

- Explicit native execution is still split across an early preload attempt, a
  post-TCP retry, and `_open_browser`; tracing must prove which adapter actually
  won before this branch is migrated.
- The current subprocess gate calls `_serve_daemon` directly. It proves daemon
  shutdown but not the public CLI, display handoff, or built-wheel behavior.
- Several `view()` tests mock `_launcher` aliases while the planner probes
  `_launch_plan` and `_platform` directly, so ambient ports, VS Code variables,
  config, and native dependencies can change their result.
- Native, real VS Code Extension Host, tunnels, MATLAB, and notebook first-frame
  readiness still require host evidence; mocked policy tests are not substitutes.

## 2026-07-21 exact native observation

`debug/parameter_maps.nii --window native` was run with local VS Code-terminal
facts and the Phase 0 trace. The plan selected native and wrote zero VS Code
requests. The external behavior was still not acceptable:

- the early native process advertised its ready flag but failed activation;
- the CLI terminated that process and launched a second native process;
- the second process reached only the shell WebSocket, not a viewer first frame;
- no `viewer.connected` event occurred, and the transient daemon later exited.

This is a real failure, not a passing native gate. It confirms that the current
two-attempt native path treats process/shell evidence as success too early. The
Phase 2 slice must produce one serialized native attempt and require viewer
first-frame evidence before reporting success.
