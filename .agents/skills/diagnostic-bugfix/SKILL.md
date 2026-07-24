# Skill: diagnostic-bugfix

# Diagnostic Change Making

## Rule

When making any change — bugfix, behavior tweak, or feature — don't stop at the first code path. Find the root cause and check everywhere the change should apply. Propose the larger fix to the user before implementing.

## Use This When

- The user reports a bug OR asks for a behavior change in any specific mode, feature, or code path
- A change touches interaction logic (keyboard, mouse, events) that runs in multiple modes
- A change touches shared state that is read or written from multiple contexts

Skip for trivial isolated changes (typo, single-element CSS tweak, error message text).

## Workflow

After reproducing the immediate bug and before writing the fix:

1. **Map the state.** What variables/objects are involved? Who reads them? Who writes them? Use `grep` to find every access site.

2. **Find parallel paths.** Does the same behavior exist in other modes (normal, compare, multiview, qMRI, mosaic, projection, zen)? If the bug is in keyboard handling, check every keyboard handler. If it's in a collapse/close function, check every close path.

3. **Check for duplication.** Is the same logic copy-pasted across modes with different object references (e.g., `primaryCb._expanded` vs `view._colorBar._expanded`)? If so, the fix needs to handle all variants, not just one.

4. **Identify the abstraction gap.** Could this be solved once instead of per-mode? Is there a missing helper, a base class method, or a shared query function that would eliminate the duplication?

5. **Report findings to the user.** Before implementing, say: "This bug also affects X, Y, Z modes because `<reason>`. The root cause is `<cause>`. Option A: fix each instance. Option B: `<abstraction fix>`. I recommend `<choice>`."

6. **Fix the root cause**, not just the reported symptom.

## Red Flags

- The fix only touches one mode but the code pattern appears in 3+ places
- A function takes `primaryCb` as a hardcoded reference instead of querying the active one
- The fix adds a guard in one function when the same guard is missing from sibling functions
- "This works in normal mode" without checking the other modes listed in the Mode Map

## Mode Map

| Mode | Colorbar owner | Keyboard guard |
|------|---------------|----------------|
| Normal | `primaryCb` | `_histPickerActive` |
| Compare | `primaryCb` + `_diffCenterCb` | `_histPickerActive` |
| qMRI | per-pane `view._colorBar` / `view._mosaicColorBar` | `_histPickerActive` |
| MultiView | `_mvColorBar` (synced to `primaryCb`) | `_histPickerActive` |
| Projection | `primaryCb` | `_histPickerActive` |
| Mosaic | per-pane `view._mosaicColorBar` | `_histPickerActive` |
| Zen | `primaryCb` | `_histPickerActive` |

When a fix touches one cell in this table, verify all cells in the same column.

## Example

**Reported:** "Arrow keys don't close histogram mode in qMRI."

**Shallow fix:** Add arrow-key close handler to qMRI keyboard path.

**Diagnostic approach:**
1. Found that both single-view and qMRI use the same `_histPickerActive` flag and `_histPickerKey` handler — so the keyboard dispatch already works for both.
2. Found that the close path (`_histPickerClose` → `_collapseHistogramImmediately`) hardcodes `primaryCb._expanded` and returns early if it's not expanded.
3. In qMRI, the expanded colorbar is `view._colorBar` (per-pane), so the collapse function does nothing.
4. Root cause: `_collapseHistogramImmediately` assumes only `primaryCb` can be expanded, but per-pane colorbars and `_diffCenterCb` can also be expanded independently.
