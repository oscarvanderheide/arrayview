# Launch matrix

Every way ArrayView can be asked to show something, and whether that way is
actually known to work. **Most rows say "never verified". That is the point of
this file** — it marks where any claim about ArrayView is a guess. Rows get
verified as work reaches them; an unverified row is not a task, it is a warning.

**Current priorities: the VS Code rows — CLI (rows 1-4) and Explorer clicks
(rows 19-20).**

## How to use it

Before changing launch, display, or server code:

1. Name the rows you are changing.
2. Name the rows you could plausibly break (same call site, same window target,
   or same environment).
3. After the change, re-check those rows and update the Status column with the
   date and the evidence label.

Evidence labels, as in `AGENTS.md`: `real host` (the actual target machine),
`real process` (a real launch, not the target environment), `component` (test
doubles), `unavailable` (could not be tested). **Never upgrade a label without
re-running the check.** A row's status is about the *last time it was seen
working*, not about whether the code looks right.

Display success means the requested array's **first rendered frame**. A process
starting, a port answering, a tab opening, or a socket connecting is not success.

## The dimensions

These multiply into thousands of combinations. The table below lists the ones
that actually occur, not the cross-product.

- **Call site** — CLI (`arrayview` console script), Python (`arrayview.view()`),
  VS Code Explorer click (`arrayview.openFile` / `arrayview.openFolder`),
  empty server (`--serve`).
- **Window target** — `vscode` (built-in browser tab over a forwarded port),
  `native` (PyWebView desktop window), `browser` (system browser), inline
  (Jupyter IFrame), `none` (returns a URL, opens nothing).
- **Environment** — local desktop, VS Code remote tunnel, VS Code Remote SSH,
  plain SSH, Jupyter local, Jupyter remote.
- **OS** — Linux, macOS, Windows.
- **Storage** — fast local disk, slow network mount (NFS, SMB).
- **Array** — small, large (progress reporting path), directory/collection.
- **Lifecycle** — cold start (no server yet), warm repeat, several viewers at
  once, close one, close all, shutdown with no orphans.

`window` resolution, from `_resolve_window_mode`: explicit `--window` wins, then
`--browser`, then config, then `vscode` if launched from a VS Code terminal;
`native` is rewritten to `vscode` under VS Code remote.

## The table

Status is **`never verified`** unless a dated entry says otherwise.

### CLI

| # | Environment | Window | Array / storage | Status |
|---|-------------|--------|-----------------|--------|
| 1 | VS Code tunnel, Linux | vscode tab | small, fast local | **verified 2026-08-06 `real host`** — clean when a viewer is already open; see row 2 |
| 2 | VS Code tunnel, Linux | vscode tab | small, fast local, **first open after ~1 min idle** | **verified 2026-08-06 `real host`** — 6/6 clean at 90 s idle, no flicker, after the server-driven idle nudge |
| 3 | VS Code tunnel, Linux | vscode tab | large (88 MB+), fast local | never verified — believed to share row 1/2 behaviour once the page loads |
| 4 | VS Code tunnel, Linux | vscode tab | slow network mount (NFS/SMB) | never verified |
| 5 | VS Code tunnel, Linux | browser | any | never verified |
| 6 | VS Code Remote SSH | vscode tab | small, fast local | never verified this session; resolves the URL differently (`asExternalUri`) |
| 7 | plain SSH, no VS Code | browser | small | never verified |
| 8 | local desktop Linux | native window | small | never verified |
| 9 | local desktop macOS | native window | small | never verified |
| 10 | local desktop Windows | native window | small | never verified |
| 11 | any | none | any | never verified — succeeds only after registration completes |

### Python `arrayview.view()`

| # | Environment | Window | Array / storage | Status |
|---|-------------|--------|-----------------|--------|
| 12 | Jupyter local | inline IFrame | small | never verified |
| 13 | Jupyter remote | inline IFrame | small | never verified — a colleague hit a failure here (fixed in `e92a92c`, not re-verified) |
| 14 | Jupyter any | native / browser | small | never verified |
| 15 | plain Python, local desktop | native (auto) | small | never verified |
| 16 | plain Python, VS Code tunnel terminal | vscode tab | small | never verified — auto-resolves to `vscode` |
| 17 | any | `window=False` | any | never verified — must return a URL and open nothing |
| 18 | any | multiple arrays (2-4 handles) | any | never verified |

### VS Code Explorer click

| # | Environment | Window | Array / storage | Status |
|---|-------------|--------|-----------------|--------|
| 19 | VS Code tunnel, Linux | vscode tab | small, fast local | **verified 2026-08-06 `real host`** — but shares row 2's idle failure |
| 20 | VS Code tunnel | vscode tab | folder / collection (`openFolder`) | never verified |
| 21 | VS Code Remote SSH | vscode tab | small | never verified |
| 22 | VS Code local (no remote) | vscode tab | small | never verified |

### Lifecycle (cuts across every row above)

| # | Case | Status |
|---|------|--------|
| 23 | Cold start — no server running yet | partially — first open with no viewer running is still unprotected; succeeded first try 2026-08-06 but has no nudge behind it |
| 24 | Warm repeat — second array within ~10 s | **verified 2026-08-06 `real host`** — always first-try, ~0.4 s |
| 25 | Several viewers open at once | **broken 2026-08-06 `real host`** — at 18 open viewer tabs, launches fail outright with no viewer; nothing closes tabs automatically |
| 26 | Close one viewer, others keep working | never verified |
| 27 | `--kill` / shutdown leaves no orphan processes | never verified this session |
| 28 | Repeat launch after `--kill` | **broken 2026-08-06 `real host`** — see row 23 |
| 29 | Two VS Code windows open, launch claimed by the right one | never verified — three windows were live during 2026-08-05 failures |

## Known-bad rows, in priority order

1. **Row 25** — viewer tabs accumulate and nothing removes them. At 18 open
   tabs, launches stop working entirely. Identified 2026-08-04, never built.
   With the idle path fixed this is the top user-facing defect.
2. **Row 23** — the very first open, with no viewer running, has nothing keeping
   its path warm and is still exposed.
3. A blank tab has two unrelated causes — a stale path and the tab ceiling — and
   they look identical. Check `browserTabsOpen=` in the opener log before
   theorising about either.
4. Everything marked `never verified` is a claim nobody has checked.

## Where the work is logged

This file is **not** a log. It records only the current status of each row.

Everything tried — hypotheses, experiments, measurements, dead ends, and
reverted changes — goes in `plans/tunnel/LOG.md`, newest at the end. Read it
before forming a theory about launch or display behaviour; it already contains
several that were tested and disproven, and re-deriving them is how the same
wrong fix keeps getting proposed. Record failures there as carefully as
successes, with the evidence label.
