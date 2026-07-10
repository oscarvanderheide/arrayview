# TODO

- [x] Align invocation and server ownership with `docs/lifecycle.md`.
- [x] Keep local VS Code CLI launches tab-shared with one backend and last-tab shutdown.
- [x] Keep VS Code file-click and custom-editor launches extension-owned and subprocess-based when possible.
- [x] Keep `view(arr)` alive after script exit until viewer instances close.
- [x] Keep Jupyter kernel-owned cleanup explicit, not tied to iframe disappearance.
- [x] Keep remote/tunnel persistence limited to the cases that need `--serve` or direct-server behavior.
- [x] Keep plain SSH on localhost port-forward behavior with transient shutdown.
- [ ] Launch reliability Phase 0: executable invocation/lifecycle contract matrix.
- [ ] Launch reliability Phase 1: shared `LaunchIntent` and `plan_launch()` used by diagnostics, CLI, and `view()`.
- [ ] Launch reliability Phase 2: owned instance registry, safe recovery, and lifecycle leases.
- [ ] Launch reliability Phase 3: structured display opener results and fallback execution.
- [ ] Launch reliability Phase 4: VS Code request/ACK and tunnel reconciliation.
- [ ] Launch reliability Phase 5: shared `SessionSpec` plus MATLAB/Julia/Explorer adapters.
- [ ] Launch reliability Phase 6: macOS/Linux/Windows gates and compatibility rollout.
- [ ] Write `dev/launch-tunnel-test-handoff.md` for remote-window verification.
