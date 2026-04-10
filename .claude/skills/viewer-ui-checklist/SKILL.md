---
name: viewer-ui-checklist
description: Use when adding keyboard shortcuts, changing layout, or making any UI change to arrayview. Ensures visual_smoke.py, help overlay, and README stay in sync.
---

# ArrayView UI Checklist

## Rule

Every UI change to arrayview MUST be reflected in `tests/visual_smoke.py`, the help overlay, and (when user-facing) `README.md` before the task is complete.

## What counts as a UI change

- New keyboard shortcut
- Changed keyboard shortcut behavior
- New view mode or display mode
- Layout changes (canvas sizing, colorbar position, overlays)
- New overlay, dialog, or panel

## Steps (mandatory, in order)

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

6. **Update `README.md`** if the change is user-facing (new CLI flag, new API, new mode)
   - Use the `docs-style` skill for formatting rules

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
