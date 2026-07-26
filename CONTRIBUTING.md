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

### Making Explorer clicks start fast

In a workspace with no `.venv`, a click falls through to `uv run --with
arrayview`, which builds an ephemeral environment. That environment is cached,
but **publishing a new arrayview invalidates it**, so the next click pays a full
rebuild — measured at 10.4 s to resolve and install 110 packages, plus ~6 s more
compiling bytecode on first import. Roughly 19 of a 21 s cold launch.

Install arrayview as a uv tool once and the extension prefers it (a stable venv,
bytecode already cached — 1.6 s from spawn to serving URL):

```bash
uv tool install arrayview
# one-time: uv does not compile bytecode on install, and the first launch
# otherwise pays for it
"$(uv tool dir)/arrayview/bin/python" -m compileall -q -j 8 \
    "$(uv tool dir)/arrayview/lib/python3.12/site-packages"
```

**Updating is now explicit** — this is the tradeoff. `uv run --with arrayview`
picked up new releases on its own (at the cost above); a tool install stays on
the version you installed until you say otherwise:

```bash
uv tool upgrade arrayview   # after publishing a release
```

Re-run `compileall` after upgrading. `curl -s localhost:8000/ping` reports
`package_version` if you need to confirm what a click actually ran.

The extension skips the tool environment whenever `arrayview.packageSpec` points
at a checkout, so it cannot shadow the working tree you are testing.

### Testing backend changes without cutting a release

Clicking an array in the VS Code Explorer launches ArrayView through
`uv run … --with arrayview`, which resolves the **released** package from PyPI.
That makes iterating on `src/` painful: every change would need a release.

The `arrayview.packageSpec` setting overrides that. Point it at an absolute
path and the extension switches to `--with-editable <path>`, so the launch runs
your working tree live — edits apply on the next launch, no reinstall, no
release:

```jsonc
// ~/.vscode-server/data/Machine/settings.json   (remote/tunnel)
// ~/.config/Code/User/settings.json             (local)
"arrayview.packageSpec": "/path/to/your/arrayview"
```

Set it back to `"arrayview"` to test the real released package again.

Three things to know, all of which have cost debugging time:

- **The setting is `machine-overridable` and Machine settings are per-host, not
  per-window.** Setting it while developing silently applies to *every* window
  on that host — including unrelated workspaces you thought were testing the
  release. If a "clean" window behaves like your checkout, check this first.
- **A running daemon outranks the setting.** If port 8000 already has a daemon,
  a click takes the fast-load path and hands the file to that existing process,
  whatever version it is. Changing `packageSpec` has no effect until the daemon
  is gone. `curl -s localhost:8000/ping` reports `package_version` — check it
  before concluding anything about which code just ran.
- **No window reload is needed.** The setting is read per launch. The daemon,
  however, must actually be killed.

This only affects the Python package. The extension itself is a separate
artifact: rebuild and reinstall the VSIX to change extension behaviour (see
`vscode-extension/AGENTS.md`).

## Style notes

- The frontend lives in a single file: `src/arrayview/_viewer.html`.
  HTML, CSS, and JS are all in there. Keep it that way.
- Python backend uses `uv` for package management.
- Commit messages follow conventional commits (`feat:`, `fix:`, `refactor:`,
  etc.).
