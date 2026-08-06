# ArrayView

Consult `.mex/ROUTER.md` for task routing, project state, and context loading.

## How to talk to me

I am the user of this tool, not its implementer. I do not know the internals and
do not want to learn them. Write for someone who knows what ArrayView *does* and
nothing about how it works.

**Rules, in priority order:**

1. **Answer first, in one sentence.** What is broken, or what you did. Then stop.
   Add detail only if I ask.
2. **Never name an internal thing in a sentence I have to act on.** No function
   names, file names, variable names, transports, proxies, handlers, sockets,
   sessions, tokens, or phase names. If I must choose between options, describe
   them by *what I will see happen*, not by what the code does.
3. **No walls of text.** A few short sentences. No multi-section reports with
   headers unless I ask for a writeup.
4. **No tables of internal measurements** unless I asked for numbers.
5. **Say "I don't know" and "I broke it" plainly.** Do not soften, do not
   explain at length, do not list what you tried.

**Bad:** "Skip blank-tab recovery when the transport is the direct loopback
proxy, and keep it for the relay path it was written for."

**Good:** "The viewer gives up too early and reopens the tab. I can make it wait
longer so that stops."

**Bad:** "panel_opened → script-loaded exceeded preScriptTimeoutMs."

**Good:** "Big arrays take a few seconds to open, and something was treating
that as a failure."

If you catch yourself writing a sentence I would have to ask you to translate,
delete it and write the plain version instead.

## Skills

Load the relevant skill before touching the corresponding area.

| Skill | When |
|-------|------|
| `frontend-designer` | Styling/layout changes to `_viewer.html` |
| `visual-bug-fixing` | A real visual bug, layout glitch, or rendering artifact |
| `modes-consistency` | Canvas, zoom, colorbars, shortcuts, or layout across modes |
| `invocation-consistency` | Server startup, display-opening, env detection |
| `ui-consistency-audit` | Explicit full visual audit or pre-release validation |
| `viewer-ui-checklist` | Release prep — syncing smoke/help/docs |
| `diagnostic-bugfix` | Any change — find root cause, check all modes, propose abstraction fix |
| `todo-workflow` | Batches of items — enforces commit-per-item and collateral updates |
| `playwright-cli` | Driving a real browser to verify behaviour |

Claude Code additionally loads `.claude/skills/vscode-extension` for opener/IPC
work and `.claude/skills/iterative-debug`.

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

### Settled decisions — do not reopen these

Both were decided by measurement and both keep getting proposed again by people
who have not read this. If you are about to suggest either, you are going
backwards.

**The viewer is delivered over a forwarded port with a WebSocket. Not a webview.**
The extension-owned webview panel was tried and is **slower**. It also cannot
resolve a URL to load in the tunnel setup and fails outright with "Failed to
resolve remote viewer URL". Do not propose moving the viewer back into a webview
panel, an iframe wrapper inside one, or any variant of "retry inside our own tab"
that depends on that surface. If the built-in browser tab needs to behave
differently, fix the browser path.

**Private ports are fine. Stop trying to make ports public.** A long series of
tricks existed to force a public/relay-visible port, and they are all obsolete —
a way was found to use private ports. Do not add port-promotion, relay
publication, public-visibility fallbacks, or "the port must be public for the
client to reach it" reasoning. If the client cannot reach the port, the forward
is missing or stale, which is a different problem with a different fix.

### Launch and display changes

`LAUNCH-MATRIX.md` lists every way ArrayView can be launched and whether that way
is known to work. Before changing launch, display, or server code, name the rows
you are changing and the rows you could break; after the change, re-check those
rows and update their status with the date and evidence label. A row is about the
last time it was *seen* working, not about whether the code looks right.

- Before editing launch code, run the smallest affected **public command** in
  the real target environment when that environment is available. Record the
  command and the observable failure.
- Run that same real command after each meaningful fix. Do not replace it with
  a mocked extension, fake opener, or helper-level test.
- A process starting, port responding, tab opening, or WebSocket connecting is
  progress, not success. Display success requires the requested array's first
  rendered frame. `--window none` succeeds only after registration completes.
- Verify caller behavior and cleanup too: prompt/return timing, repeat launch,
  display close, session release, and owned-process shutdown.
- Label evidence honestly as `real host`, `real process`, `component`, or
  `unavailable`. Never describe component evidence as proof of an unavailable
  host boundary.
- If the simplest public gate still fails, diagnose that exact boundary before
  broad refactoring or adding more abstraction.
- Warn the user before opening or reloading IDE windows, installing an
  extension into their active profile, or launching any GUI. Never create a
  temporary VS Code profile/window without explicit permission.
- Follow `.mex/patterns/validate-launch-path.md`; for VS Code delivery also use
  `.mex/patterns/debug-vscode-extension-python.md`.
- Errors during display opening are shown as clean user messages by default.
  Run with `--trace` or set `ARRAYVIEW_TRACE=1` to see full Python tracebacks.

Validate against a served session on `http://localhost:<port>/`. Never open
`src/arrayview/_viewer.html` as a file link: `file://.../_viewer.html` has no
backend session behind it, so the viewer cannot work and proves nothing.

If `uv run arrayview --serve --port <port>` reports success but `localhost:<port>`
refuses connections, start the empty server directly in one terminal, leave it
running, and load the file from a second terminal:

```bash
uv run python -c "from arrayview._launcher import _serve_empty; _serve_empty(8000)"
uv run arrayview <file> --window browser --port 8000
```

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
For startup/display work, "narrowly" includes the affected real launch gate
defined above; automated tests alone are not sufficient.

```bash
uv run pytest tests/test_api.py -v
uv run pytest tests/test_browser.py -v
uv run pytest tests/test_mode_roundtrip.py -v
uv run python tests/visual_smoke.py
```
