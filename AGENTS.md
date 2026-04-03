Read `src/arrayview/ARCHITECTURE.md` for codebase orientation.

# ArrayView

## Skills

Load the relevant skill before touching the corresponding area.

| Skill | When |
|-------|------|
| `ui-consistency-audit` | Any visual/UI change |
| `frontend-designer` | Styling/layout changes to `_viewer.html` |
| `vscode-simplebrowser` | Extension, signal-file IPC, `_VSCODE_EXT_VERSION` |
| `invocation-consistency` | Server startup, display-opening, env detection |
| `docs-style` | README, help overlay, docstrings |

## Non-Negotiables

- Always use `localhost` (not `127.0.0.1`) -- required for VS Code port forwarding
- Never `--force` reinstall the extension if correct version is on disk
- Do not add logic to `_app.py` -- compat shim only
- Avoid orphan processes; shutdown must be automatic
- Do not regress working display paths when fixing another
- For visual/animation features, propose 2-3 options BEFORE implementing
- UI visibility changes go through reconcilers (`_reconcileUI`/`_reconcileLayout`/`_reconcileCompareState`/`_reconcileCbVisibility`), not inline `style.display` or `classList` toggles in mode functions
- All colorbar state (animation, window/level, hover, drag) flows through `primaryCb` ColorBar instance — never read/write legacy globals. Multiview colorbars sync via `primaryCb`.

## Execution

Always use **subagent-driven development** for implementation. Commit completed work automatically.

## Testing

```bash
uv run pytest tests/test_api.py -v       # HTTP API
uv run pytest tests/test_browser.py -v   # Playwright
uv run python tests/visual_smoke.py      # screenshots
```

After any UI change, use `/ui-consistency-audit` to verify across all modes.
