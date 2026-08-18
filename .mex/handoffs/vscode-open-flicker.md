---
name: handoff-vscode-open-flicker
description: Handoff for VS Code tunnel launch flicker/retries. Opener 0.15.53 is active and passed real-host launch, multi-tab, scoped-close, and final-cleanup validation.
last_updated: 2026-08-18
---

# Handoff: VS Code tunnel launch flicker/retries

## Current state — read this first

Opener **0.15.53 is built, installed, and active in both live Tunnel windows**.
Its installed source exactly matches the bundled and working source. After the
user reloaded, eleven real-host launches rendered on navigation attempt 0. Two
controlled five-launch batches each produced five distinct tabs with no retry,
replacement, or cross-window claim. In the second batch, closing the middle tab
released exactly that array while the other four stayed connected; closing the
rest released every session and the temporary server shut down automatically.

Rows 1, 3, 23–26, and 29 now have 2026-08-18 `real host` evidence. Row 32's
same-tab recovery remains `component`: no post-reload launch happened to lose
its first navigation, so the retry path was not induced on the real host. Idle,
network-mount, Explorer, collection, Remote SSH, and local VS Code rows were not
exercised by this validation and must not be claimed from it.

The prior diagnosis ("VS Code drops the request") combined two different
failures and was too vague:

1. In the two newest real flickers (`918c60cf…`, `8a8e5a08…`), attempts 0 and
   1 reached and resolved the short viewer route in 31–69 ms. The opener then
   waited about 1.6 s for `script-loaded`, which sits in the large inline viewer
   script, declared the page blank, and closed it. That was ArrayView replacing
   a page whose navigation had already begun. 0.15.53 reports
   `navigation-arrived` from a tiny script at the start of `<head>` and gives
   the full script a fresh bounded startup budget.
2. A separate VS Code startup race can genuinely lose the first navigation.
   `workbench.action.browser.open` resolves when the editor is created, while
   the integrated browser starts `loadURL` without awaiting a load result. The
   remote-browser proxy arrives asynchronously; an early localhost navigation
   is not replayed when that proxy later becomes ready. VS Code exposes neither
   proxy readiness nor browser load events to extensions.

ArrayView can still make the second case invisible. VS Code supports navigating
an existing integrated-browser tab when `reuseUrlFilter` is a matching glob.
The previous filter, `/_av/<tab-key>/`, did not match the actual
`/_av/<tab-key>/<navigation-key>` path, so the documented 0.15.35 "reuse does
not work" conclusion was invalid. 0.15.53 uses
`/_av/<tab-key>/**` and keeps the tab open. A retry now navigates that exact tab
in place; every separate array gets a fresh tab key and therefore a separate
tab. Component tests emulate VS Code's path matching and prove one physical tab
across retries and two physical tabs across two launches.

Full narrative, in order, is in `plans/tunnel/LOG.md` — this file is a map to
that, not a replacement. The sections from 2026-08-14 onward are the ones
that matter; everything before is settled or superseded.

## The user's ask (plain language)

Click an array in Explorer, or run `arrayview <file>` in a VS Code tunnel
terminal — it opens. No visible retry, no flicker, no fallback surface of any
kind (a browser-popup fallback was tried and explicitly rejected). Also,
**unchanged and non-negotiable**: opening several arrays must give several
tabs, one per array, never fewer. See `plans/tunnel/REQUIREMENTS.md` R5 — it
is short and load-bearing; read it before designing anything here.

## What's actually wrong, confirmed with hard evidence

`~/.arrayview/launch-trace.jsonl` (opt-in tracing, already enabled on this
host) distinguishes `page.route_prepared` (extension asked to navigate) from
`page.route_entered` (the page's GET actually reached the backend). Scoped to
the last 48h as of 2026-08-15: **~20-27% of individual page-load attempts
never arrive at all**, in bursts (a burst can outlast the whole ~10s retry
ladder). This is a VS Code / Dev Tunnels transport characteristic, not an
ArrayView bug — two research subagents independently confirmed no lever
inside this codebase reduces it. It is *not* about lost status reports —
directly falsified: the trace shows the page's own GET never arriving at all
on a real failed launch, so there was no loaded page to fail to report
anything in the first place.

## Dead ends — do not re-propose these

- **A softer fallback** (open the array in the system browser if the retry
  ladder exhausts). Built, tested, explicitly rejected by the user: no
  fallback surface of any kind, not even a gentler one. Full writeup in
  `plans/tunnel/LOG.md`, 2026-08-15 ("0.15.49: stop escalating a fully-dead
  ladder").
- **Hiding the retry ladder** (open the tab non-visibly until confirmed, then
  reveal it). No VS Code API supports a non-visible tab for
  `workbench.action.browser.open`; the only "hidden" surface is a webview
  panel, which is the settled-decision-forbidden architecture (see
  `CLAUDE.md`). Confirmed by a research subagent, 2026-08-15.
- **Warming the path from a webview/hidden panel before the real tab opens.**
  Already tried and killed on 2026-08-06 (see `plans/tunnel/LOG.md`) — a
  webview's requests cross a different channel than the real browser tab's,
  so it can't test or predict the thing it's standing in for. Do not retry
  this shape of idea.

## What broke on 2026-08-16 — read before touching this again

Built "in-place array switching": an already-open, already-connected viewer
tab can be retargeted to a different array over its own live connection
instead of the extension opening a fresh, drop-prone navigation. Mechanism,
two real dormant bugs found and fixed via actual Playwright testing (not
guessed — a `View.destroy()` bug that permanently deleted the canvas on a
second normal-mode load, and a metadata promise that only ever resolved
once), and a real concurrency bug an independent review caught and a fix
landed for — all real, all genuinely verified. Full detail in
`plans/tunnel/LOG.md`'s "0.15.49: in-place array switching" section.

**The fix was never gated on whether the user already has more than one
ArrayView tab open.** It reused *any* remembered live tab on the same port
for *any* subsequent launch — so opening a second array while a first was
already open silently retargeted the first tab instead of opening a second
one, directly violating R5. None of the testing performed caught this
(Playwright tests, extension unit tests, two rounds of subagent review, a
full regression suite) — all of it exercised single-tab sequences only. It
was caught by the user asking "what if I want two tabs open" *after* the
build was already installed and the window had already been reloaded — i.e.
it was live in the user's real session for a period. **Reverted in full**:
`extension.js`, the two new backend routes (`_routes_websocket.py`), and the
`_viewer.html` client changes are all back to the pre-2026-08-16 state;
0.15.50 ships that reverted code under a new version number solely to
supersede the broken 0.15.49 registration that had already activated.

**If this is attempted again**: the retarget must only ever apply when the
user currently has exactly one ArrayView tab open — the moment there are two
or more, every further launch must fall through to a normal, separate-tab
navigation exactly as today. That heuristic has a known gap (one tab open,
user wants a deliberate *second* one — it will guess wrong and reuse instead
of opening new). Get explicit sign-off on the exact rule, including that gap,
before writing any code — not after building and installing it.

## Rebuild / install recipe (exact active host)

- Bump `vscode-extension/package.json` **and** `_VSCODE_EXT_VERSION` in
  `src/arrayview/_vscode_extension.py` together.
- `cd vscode-extension && npx --yes @vscode/vsce package -o
  ../src/arrayview/arrayview-opener.vsix`
- Install via:
  `uv run python -c "from arrayview._vscode_extension import _ensure_vscode_extension; _ensure_vscode_extension(is_remote=True)"`
  — the tunnel host loads `~/.vscode-server/extensions`; a plain `code
  --install-extension` run from an unrelated shell writes the wrong directory
  (`~/.vscode`, the desktop location). `is_remote=True` is required when
  running from a shell that is not itself a live VS Code terminal (no IPC
  hook to auto-detect it).
- The user's own window reload is the only step that cannot be done from
  here — never attempt to trigger it programmatically; warn and wait.

## How to test

- `node --check vscode-extension/extension.js`; every
  `vscode-extension/test_*.js` file passes via plain `node <file>.js`.
- `uv run pytest tests/test_lifecycle_contract.py -k "vsix or version"`,
  `tests/test_vscode_ack_protocol.py`, `tests/test_api.py`.
- Real-browser verification (not mocks) is available and load-bearing for
  anything touching `_viewer.html`'s load/render lifecycle:
  `tests/conftest.py`'s `page` / `server_url` / `client` fixtures spin up a
  real local server and real Chromium via Playwright. Static reading alone
  missed two real bugs here that only showed up when actually run — don't
  skip this step for viewer.html changes.
- One pre-existing, unrelated failure on the clean tree:
  `tests/test_lifecycle_contract.py::test_integrated_launch_cleanup_is_scoped_per_request_token`.
  Confirm via `git stash` before assuming anything you touched caused a
  failure — this one and a matching set in `test_mode_matrix.py` /
  `test_mode_consistency.py` / `test_colorbar_hover_highlight.py` /
  `test_mode_roundtrip.py` (20 total) fail identically with and without any
  change made here.
