# ArrayView

I haven't written or read a single line of code in src so when you ask me questions/input,
keep it simple with some simple examples.

## Skills

Load the relevant skill before touching the corresponding area.

| Skill | When |
|-------|------|
| `ui-consistency-audit` | Explicit full visual audit, cross-mode UI regression work, or pre-release validation |
| `frontend-designer` | Styling/layout changes to `_viewer.html` |
| `vscode-simplebrowser` | Extension, signal-file IPC, `_VSCODE_EXT_VERSION` |
| `invocation-consistency` | Server startup, display-opening, env detection |
| `viewer-ui-checklist` | Release prep for UI changes, or when syncing smoke coverage/help overlay/docs explicitly |
| `docs-style` | README, help overlay, docstrings (formatting rules) |

## Non-Negotiables

- Always use `localhost` (not `127.0.0.1`) — required for VS Code port forwarding
- Never `--force` reinstall the extension if correct version is on disk
- Do not add logic to `_app.py` — compat shim only
- Avoid orphan processes; shutdown must be automatic
- Do not regress working display paths when fixing another
- For visual/animation features, propose 2-3 options BEFORE implementing
- UI visibility changes go through reconcilers (`_reconcileUI`/`_reconcileLayout`/`_reconcileCompareState`/`_reconcileCbVisibility`), not inline `style.display` or `classList` toggles in mode functions
- All colorbar state (animation, window/level, hover, drag) flows through `primaryCb` ColorBar instance — never read/write legacy globals. Multiview colorbars sync via `primaryCb`.
- Keybinds flow through the command registry (`commands` / `keybinds` in `_viewer.html`), not inline keydown branches
- When adding or changing a keybind, also update `GUIDE_TABS` in `_viewer.html` — the help overlay renders from that static data structure, it is NOT auto-generated from command titles

## Contributing

Before creating a PR or making user-facing changes, read `CONTRIBUTING.md`. It defines the design
language, keybinding conventions, overlay/popup patterns, and testing requirements. All PRs that
touch the viewer must follow it.

## Execution

Always use **subagent-driven development** for implementation. Do not ask — just use it.

Work in **feature branches**. Create a branch for each feature, commit work there.

## Testing

**Browser MCP:** Use the `chrome-devtools` MCP (not playwright) for interactive browser automation during development.

**During development:** Only verify the specific feature works. Do not run the full test suite or a full visual audit unless the user explicitly asks for it.

**When the user explicitly asks for the full validation pass or asks for a new release:**
1. Run the full test suite
2. Run the full visual audit for UI work (`/ui-consistency-audit`, smoke tests, screenshots as relevant)
3. Verify no regressions across modes/features
4. Squash and merge into main — no merge commit

```bash
uv run pytest tests/test_api.py -v                    # HTTP API
uv run pytest tests/test_browser.py -v                # Playwright (Python library, for CI)
uv run pytest tests/test_mode_roundtrip.py -v         # mode state round-trip
uv run pytest tests/test_command_reachability.py -v   # command when-clause matrix
uv run python tests/visual_smoke.py                   # screenshots
```

For UI work, only use `/ui-consistency-audit` when the user explicitly asks for a full visual check, when tracking a cross-mode visual regression, or during release validation.

## Commands

Always use `uv` — no system Python.

- Test: `uv run pytest tests/`
- CLI: `uvx arrayview path/to/file.npy`
- Build: `uv build`

## After Every Task

Update `.mex/ROUTER.md` project state and any `.mex/` files that are now out of date. If no pattern existed for the task you just completed, create one in `.mex/patterns/`.

## Navigation

Consult `.mex/ROUTER.md` when you need project state, routing guidance, or a matching task pattern.
Use it as a dispatcher: load only the relevant context or pattern files for the task at hand.
