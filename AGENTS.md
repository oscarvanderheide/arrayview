Read `src/arrayview/ARCHITECTURE.md` for codebase orientation.

# ArrayView

I haven't written or read a single line of code in src so when you ask me questions/input,
keep it simple with some simple examples.

## Skills

Load the relevant skill before touching the corresponding area.

| Skill | When |
|-------|------|
| `ui-consistency-audit` | Any visual/UI change |
| `frontend-designer` | Styling/layout changes to `_viewer.html` |
| `vscode-simplebrowser` | Extension, signal-file IPC, `_VSCODE_EXT_VERSION` |
| `invocation-consistency` | Server startup, display-opening, env detection |
| `viewer-ui-checklist` | Any UI change (smoke tests, help overlay, README sync) |
| `docs-style` | README, help overlay, docstrings (formatting rules) |

## Non-Negotiables

- Always use `localhost` (not `127.0.0.1`) -- required for VS Code port forwarding
- Never `--force` reinstall the extension if correct version is on disk
- Do not add logic to `_app.py` -- compat shim only
- Avoid orphan processes; shutdown must be automatic
- Do not regress working display paths when fixing another
- For visual/animation features, propose 2-3 options BEFORE implementing
- UI visibility changes go through reconcilers (`_reconcileUI`/`_reconcileLayout`/`_reconcileCompareState`/`_reconcileCbVisibility`), not inline `style.display` or `classList` toggles in mode functions
- All colorbar state (animation, window/level, hover, drag) flows through `primaryCb` ColorBar instance — never read/write legacy globals. Multiview colorbars sync via `primaryCb`.
- Keybinds flow through the command registry (`commands` / `keybinds` in `_viewer.html`), not inline keydown branches. The help overlay auto-generates from command `title` fields — do not hand-edit it.

## Contributing

Before creating a PR or making user-facing changes, read `CONTRIBUTING.md`. It defines the design
language, keybinding conventions, overlay/popup patterns, and testing requirements. All PRs that
touch the viewer must follow it.

## Execution

Always use **subagent-driven development** for implementation. Commit completed work automatically.

## Testing

```bash
uv run pytest tests/test_api.py -v                    # HTTP API
uv run pytest tests/test_browser.py -v                # Playwright
uv run pytest tests/test_mode_roundtrip.py -v         # mode state round-trip
uv run pytest tests/test_command_reachability.py -v   # command when-clause matrix
uv run python tests/visual_smoke.py                   # screenshots
```

After any UI change, use `/ui-consistency-audit` to verify across all modes.
