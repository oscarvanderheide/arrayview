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
| 3 | VS Code tunnel, Linux | vscode tab | large (88 MB, 4-D), fast local | **verified 2026-08-06 `real host`** — every flicker measurement this session used this file; renders in ~0.4-0.9 s once the page loads |
| 4 | VS Code tunnel, Linux | vscode tab | slow network mount (NFS/SMB) | never verified |
| 5 | VS Code tunnel, Linux | browser | any | never verified |
| 6 | VS Code Remote SSH | vscode tab | small, fast local | never verified this session; resolves the URL differently (`asExternalUri`) |
| 7 | plain SSH, no VS Code | browser | small | never verified |
| 8 | local desktop Linux | native window | small | never verified |
| 9 | local desktop macOS | native window | small | never verified |
| 10 | local desktop Windows | native window | small | never verified |
| 11 | any | none | any | **verified 2026-08-06 `real host`** — but note what it *does*: it loads the array, waits for the session to register, then **releases it again** and exits, so no server is left behind. It now says so on success. This is deliberate and **pinned by tests** (`test_source_safety.py`: the daemon must not keep its port or stay alive), because the mode exists to load a source — including one on a network share — prove it registers, and leave nothing holding it. It is a safe loader, not "give me a URL to open later" |

### Python `arrayview.view()`

| # | Environment | Window | Array / storage | Status |
|---|-------------|--------|-----------------|--------|
| 12 | Jupyter local | inline IFrame | small | never verified |
| 13 | Jupyter remote | inline IFrame | small | never verified — a colleague hit a failure here (fixed in `e92a92c`, not re-verified) |
| 14 | Jupyter any | native / browser | small | never verified |
| 15 | plain Python, local desktop | native (auto) | small | never verified |
| 16 | plain Python, VS Code tunnel terminal | vscode tab | small | never verified — auto-resolves to `vscode` |
| 17 | any | `window=False` | any | never verified — must return a URL and open nothing |
| 18 | any | multiple arrays (2-4 handles) | any | **verified 2026-08-06 `real host`** — `view(a, b, c)` returns 3 handles and opens one tab holding all three as a compare group, first try |

### VS Code Explorer click

| # | Environment | Window | Array / storage | Status |
|---|-------------|--------|-----------------|--------|
| 19 | VS Code tunnel, Linux | vscode tab | small, fast local | **verified 2026-08-06 `real host`, by the user** — clean cold start via Explorer click, served from a freshly forwarded port |
| 20 | VS Code tunnel | vscode tab | folder / collection | **broken 2026-08-06 `real host`** — opening a *directory* from the CLI never displays. The session it signals to VS Code is not registered on the server (`POST /viewer-phase/<sid>/…` returns 404 Session not found), so the opener cannot prepare its readiness journal and the launch fails. Interleaved control: single files from the same directory rendered 2/2 while the folder failed 2/2. The Explorer `openFolder` click is a separate path and still untested |
| 21 | VS Code Remote SSH | vscode tab | small | never verified |
| 22 | VS Code local (no remote) | vscode tab | small | never verified |

### Lifecycle (cuts across every row above)

| # | Case | Status |
|---|------|--------|
| 23 | Cold start — no server running yet | **verified 2026-08-06 `real host`** — 5/5 clean, served from a freshly forwarded port |
| 24 | Warm repeat — second array within ~10 s | **verified 2026-08-06 `real host`** — always first-try, ~0.4 s |
| 25 | Several viewers open at once | **verified 2026-08-06 `real host`** — 24 viewers open, 24/24 clean opens, no ceiling reached; the 2026-08-04 wall predates the duplicate-socket fix |
| 26 | Close one viewer, others keep working | never verified |
| 27 | `--kill` / shutdown leaves no orphan processes | **verified 2026-08-06 `real host`** — after `--kill`: no daemon processes, no listening ports (cold-start ports included), no stale claims, registry empty. A normal launch creates exactly one server. **But**: the server is `persist`ent by design, so it outlives the command that started it — including one interrupted mid-launch — and is only stopped explicitly. It stays discoverable via `arrayview instances` and killable, so it is a managed server rather than an orphan; a server from an *earlier session* was still running today and was found and stopped the same way |
| 28 | Repeat launch after `--kill` | **verified 2026-08-06 `real host`** — 5/5 clean, this is the cold-start path |
| 29 | Two+ windows open, terminal launch opens in **the window the terminal is in** | never verified end to end — every request is targeted at one window by name and 544 dispatches show **0** cross-window claims, so misrouting is not the risk; the risk is a **refusal** when the terminal outlives ~8 window reloads or under tmux |
| 30 | Two+ windows open, Explorer click opens in **the window clicked in** | partially verified 2026-08-06 `real host` — clicks landed in the clicking window throughout; not tested with two windows side by side. Separately: sound on this host (the click writes to its own window's file), but on any platform with no IPC hook (documented: local macOS) the click writes a filename **no window watches**, costing ~12 s per click before it falls back |
| 31 | Two+ windows, an unfocused window must not claim another's launch | **not reachable** — the focus guard protects a broadcast path that current Python never writes; safety comes from every request being targeted, not from focus |
| 32 | Two+ windows, one alive but **not responding** | **partially fixed 2026-08-06** — the window now detects itself after two consecutive stalled display commands and offers a one-click reload, and the CLI names the cause instead of suggesting a plain retry (`component` evidence: notice fires on the 2nd stall, once only). Still true underneath: liveness is pid-based, there is no heartbeat and no takeover, so a launch already claimed by that window is still lost |
| 33 | Two windows, each with its own server on its own port | never verified — and **not a state the code maintains**: the instance registry has no concept of a window, so a second window silently reuses the first window's server unless `--port` is passed explicitly |

## Known-bad rows, in priority order

1. **Row 32** — a wedged window now *announces itself* and the CLI explains it,
   but nothing takes over the launch it already claimed, so that one launch is
   still lost. A takeover needs a heartbeat in the claim journal; the detection
   landed first because it is what the user actually lacked.
2. **Row 25** — no ceiling was found at 24 open viewers. Nothing still closes
   viewer tabs, so a ceiling presumably exists somewhere higher; it has not been
   observed and should not be designed against until it is.
3. A blank tab has three unrelated causes — a stale forward, a forward that does
   not exist yet, and the tab ceiling — and they look identical from outside.
   Check `browserTabsOpen=` and whether any viewer is connected before
   theorising about any of them.
4. Everything marked `never verified` is a claim nobody has checked.

## Risks the current fixes introduce

Both 2026-08-06 fixes are narrow, but neither is free. Re-check these rows before
trusting them.

**The idle nudge** (server asks one connected viewer to make a few connections
while nothing is being opened):

- Rows 12-15 (Jupyter, native, plain browser): the nudge is sent to whichever
  viewer the server picks, and only a VS Code integrated-browser viewer acts on
  it — everything else ignores the message. Harmless, but **untested against a
  Jupyter or native viewer**, and it does mean a non-VS-Code deployment sends a
  message every 15 s that nobody uses.
- **Known gap, mixed environments**: the server stops at the first viewer that
  *accepts* the message, not the first that can act on it. With a Jupyter viewer
  and a VS Code viewer on one server, the nudge may keep going to the Jupyter one
  and the VS Code path would silently stop being protected — the exact failure
  shape as the oldest-socket bug, which took a five-run measurement to catch.
  Not built: the scenario is rare, and it should be fixed by having the viewer
  declare it can act rather than by guessing.
- Row 25: adds a small burst of short-lived connections. It is one burst total
  regardless of tab count, which is why it was built this way, but it is still
  connections against a ceiling that has not been located.

**The cold-start port** (running server binds one more port for a launch with no
viewer connected):

- Explorer clicks against an already-running server take the opener's own fast
  path (`extension.js` `_fastLoadViaDaemon`) rather than the Python signal
  writer. That path now asks for a cold-start port itself, tunnel only —
  **observed firing correctly 2026-08-06** (fresh port, rendered in under a
  second). What is *not* verified is that same path after a long idle, because
  the state it needs — server running, no viewer open, connection gone cold — is
  hard to produce: the server tends to stop when its last viewer closes. Likely
  rare in normal use; do not spend much on it.
- That fast path still **hardcodes port 8000** and silently declines when the
  daemon is on any other port. Unfixed, and unrelated to the flicker.
- Rows 6 and 21 (Remote SSH): **guarded 2026-08-06** — the swap is now tunnel
  only. Under SSH the main port is forwarded because it is printed to the
  terminal and given port attributes; an ephemeral port gets neither, so it
  would have taken new risk for a benefit never measured there.
- Rows 11 and 17 (`--window none`, `window=False`): unaffected — the swap only
  happens on the VS Code signal path, and those return the original URL.
- Rows 5, 7, 8-10, 12-15 (browser, native, Jupyter): unaffected, same reason.
- Row 27 (orphans): extra ports live in the existing process and die with it, so
  they cannot outlive the server. The release path is only *observed* holding one
  port while a viewer is open; it has not been watched letting go on the real
  host, because that needs every viewer tab closed.
- `doctor`, `instances`, `--kill` and the registry all still record the main
  port only. A viewer served from an extra port is still owned by that process,
  so stopping works — but any tool that assumes "the port in the registry is the
  port the viewer uses" is now wrong.

## Where the work is logged

This file is **not** a log. It records only the current status of each row.

Everything tried — hypotheses, experiments, measurements, dead ends, and
reverted changes — goes in `plans/tunnel/LOG.md`, newest at the end. Read it
before forming a theory about launch or display behaviour; it already contains
several that were tested and disproven, and re-deriving them is how the same
wrong fix keeps getting proposed. Record failures there as carefully as
successes, with the evidence label.
