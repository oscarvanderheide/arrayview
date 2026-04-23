---
name: viewer-ui-checklist
description: Use when the user explicitly asks to sync UI docs/help/test coverage, or when preparing a UI change for release. Ensures visual_smoke.py, help overlay, and docs/ stay in sync.
---

# ArrayView UI Checklist

## Rule

This is the release/explicit-sync checklist for UI work. Do not invoke it by default for every UI edit during development. Use it when the user explicitly asks for the full sync, or when preparing a UI change for release.

`README.md` is intentionally minimal and stable — do **not** add per-feature shortcuts or behavior to it. User-facing docs live in `docs/` and are organized by topic (`display.md`, `viewing.md`, `comparing.md`, `loading.md`, `measurement.md`, `remote.md`, `configuration.md`).

## What counts as a UI change

- New keyboard shortcut
- Changed keyboard shortcut behavior
- New view mode or display mode
- Layout changes (canvas sizing, colorbar position, overlays)
- New overlay, dialog, or panel

## Steps (when this checklist is invoked)

1. **Update coverage table** at the top of `visual_smoke.py`
   - If shortcut is now testable: change `✗` to `✓ NN` with scenario number
   - If new shortcut: add a row
   - Mark untestable shortcuts with `✗ (reason)`

2. **Add or update scenario** in `run_smoke()` with a numbered section comment
   - Follow the existing pattern: `# ── NN: description ──────`
   - Capture at least one screenshot with `_shot(page, "NN_descriptive_name")`

3. **Run the smoke test** to verify the new scenario works:
   ```
   uv run python tests/visual_smoke.py
   ```

4. **Open and review** the new screenshots in `tests/smoke_output/`

5. **Update `GUIDE_TABS`** in `_viewer.html` if you added, removed, or changed a keybinding
   - Add/update the entry in the appropriate tab and section
   - Include a `hint` for non-obvious shortcuts
   - Use the `docs-style` skill for formatting rules

6. **Update the matching `docs/*.md` page** for any user-facing change. Pick the page by topic:
   - `display.md` — colormaps, range, log scale, themes, masking, layout toggles
   - `viewing.md` — navigation, zoom, slice/dim shortcuts, multiview, projections
   - `comparing.md` — compare mode, overlay, diff, registration
   - `loading.md` — file formats, CLI, Python/Julia/MATLAB API
   - `measurement.md` — ROI, hover, pixel info
   - `remote.md` — VS Code Remote, tunneling
   - `configuration.md` — settings, env vars, persistence
   - Use the `docs-style` skill for formatting rules
   - Do **not** touch `README.md` for per-feature changes — it's intentionally stable

## Red flags — STOP

- "The shortcut is too simple to need a smoke test" → ALL shortcuts need entries
- "I'll add the test later" → add it in the same task
- "The coverage table says ✗, that's fine" → only fine if you document WHY (requires dialog, etc.)

## Stability check pattern

When verifying a key causes no visual jumps:

```python
def _check_no_jump(page, key, selectors, shot_name):
    before = {s: page.locator(s).bounding_box() for s in selectors}
    _press(page, key)
    after = {s: page.locator(s).bounding_box() for s in selectors}
    _shot(page, shot_name)
    for s in selectors:
        b, a = before[s], after[s]
        if b and a:
            dx = abs(b["x"] - a["x"]); dy = abs(b["y"] - a["y"])
            if dx > 2 or dy > 2:
                print(f"  JUMP: {s} moved {dx:.0f}px/{dy:.0f}px after {key}")
```

Key selectors to check: `#canvas-wrap`, `#slim-cb-wrap`, `#info`
