# ArrayView

Consult `.mex/ROUTER.md` for task routing, project state, and context loading.

## Skills

Load the relevant skill before touching the corresponding area.

| Skill | When |
|-------|------|
| `frontend-designer` | Styling/layout changes to `_viewer.html` |
| `invocation-consistency` | Server startup, display-opening, env detection |
| `ui-consistency-audit` | Explicit full visual audit or pre-release validation |
| `viewer-ui-checklist` | Release prep — syncing smoke/help/docs |
| `docs-style` | README, help overlay, docstrings |

## Non-Negotiables

- Use uv run python instead of python
- Use `localhost`, not `127.0.0.1`
- Do not add logic to `_app.py` — compat shim only
- Keep `_viewer.html` as a single file — no build step
- UI visibility changes go through reconcilers (`_reconcileUI` / `_reconcileLayout` / etc.), not inline `style.display` or `classList` toggles
- Keybind changes must update both the command registry and `GUIDE_TABS`
- Do not regress working display paths when fixing another
- Avoid orphan processes; shutdown must be automatic
- For animation changes, verify with frame captures before claiming completion (see `.mex/patterns/animation-verify.md`); propose 2–3 options before implementing

## Execution

Use **subagent-driven development**. Work in **feature branches**.

Read `CONTRIBUTING.md` before any user-facing change or PR.

For follow-up work in `src/arrayview/_viewer.html`, do not run broad searches.
Do not use regex alternations or generic keyword sweeps across `_viewer.html`.
Search for one exact identifier at a time: an id, function name, command id, or
section marker already suggested by the user or current context. After each hit,
read only one narrow `sed` window around the match. If the needed identifier is
not known, ask or infer from recent context instead of exploring broadly. If more
than three exact searches would be needed, stop and explain why before continuing.
Do not reload `.mex` docs or skills on small follow-up UI fixes unless the task
clearly needs fresh context.

## Testing

Verify narrowly — do not run the full suite unless asked.

```bash
uv run pytest tests/test_api.py -v
uv run pytest tests/test_browser.py -v
uv run pytest tests/test_mode_roundtrip.py -v
uv run pytest tests/test_command_reachability.py -v
uv run python tests/visual_smoke.py
```

## Commands

- `uv run pytest tests/<target>`
- `uv run arrayview <file>`
- `uv build`
