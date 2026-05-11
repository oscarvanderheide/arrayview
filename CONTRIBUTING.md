# Contributing to arrayview

Thanks for your interest. This guide keeps things consistent as more people
contribute.

## Proposing changes

For anything user-facing (new shortcut, overlay, layout change), **open an
issue first**. Include:

- What it does, in one sentence
- Which key triggers it (if any)
- Which modes it affects (Normal, Multi-view, Compare, Diff, Registration, qMRI)
- A rough sketch or description of how it looks

Bug fixes and internal refactors can go straight to a PR.

## Design principles

1. **Array fills the screen.** Minimize chrome. UI elements stay hidden or
   dimmed until the user hovers or presses a key, then fade back out.

2. **Monospace only.** All text uses the system monospace stack
   (`'SF Mono', ui-monospace, 'Cascadia Code', 'JetBrains Mono', monospace`).
   Never use sans-serif.

3. **Colors via CSS custom properties.** Use `var(--surface)`, `var(--text)`,
    `var(--active-dim)`, etc. Never hardcode hex values. The viewer ships two
    themes (dark, light) and both must work.

4. **Yellow for active state.** `--active-dim` (#f5c842 in dark theme) marks
   the currently active element. Don't introduce new accent colors.

5. **All six modes.** Every visual feature must be tested across Normal,
   Multi-view (V/v), Compare (B/P), Diff (X), Registration (R), and qMRI (q).
   If your feature only applies to some modes, add explicit mode guards.

## Keyboard shortcuts

- Check the existing shortcut table (press `?` in the viewer) before picking a
  key. Conflicts will be caught in review but save yourself the round-trip.
- Single lowercase letters are scarce. Prefer Shift+key or a modifier for new
  features.
- If a shortcut only makes sense in certain modes, guard it:
  ```js
  if (currentMode !== 'compare') return;
  ```
- Add the new shortcut to `GUIDE_TABS` in `_viewer.html` — the help overlay renders from that data structure at runtime, do not edit the overlay HTML directly.

## Popup menus and overlays

Several proposals involve popup/context menus. To keep them visually
consistent:

- Background: `var(--surface)`, border: `1px solid var(--border)`,
  border-radius: `var(--radius-lg)`.
- Dismiss on **Escape** and on clicking outside the popup.
- No permanent visibility -- show on trigger, hide when done.
- Keep text small (12-13px) and monospace.
- Use `var(--active-dim)` for the selected/hovered item, `var(--text)` for
  normal items, `var(--muted)` for secondary info.
- Animate in with a short opacity+scale transition, not an instant pop.

Look at `#uni-picker-box` in `_viewer.html` for a reference implementation.

## Testing checklist

Before submitting a PR:

- [ ] `uv run pytest tests/test_api.py -x` passes
- [ ] `uv run python tests/visual_smoke.py` passes
- [ ] If animation code changed (GSAP, rAF, CSS transitions), run
      `uv run python tests/capture_v_animation.py` and verify frame captures
- [ ] If you added UI, update `tests/visual_smoke.py` to cover it
- [ ] Manually verify in all affected modes (at minimum: Normal + one
      multi-pane mode)
- [ ] New shortcuts are documented in the help overlay

## Dev setup

```bash
git clone <repo-url>
cd arrayview
uv sync
uv run arrayview tests/  # launch with test data
```

## Style notes

- The frontend lives in a single file: `src/arrayview/_viewer.html`.
  HTML, CSS, and JS are all in there. Keep it that way.
- Python backend uses `uv` for package management.
- Commit messages follow conventional commits (`feat:`, `fix:`, `refactor:`,
  etc.).
