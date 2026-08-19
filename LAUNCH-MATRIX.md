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
| 1 | VS Code tunnel, Linux | vscode tab | small, fast local | **0.15.53: verified 2026-08-18 `real host`** — repeated public CLI launches rendered on their first navigation with no tab replacement |
| 2 | VS Code tunnel, Linux | vscode tab | small, fast local, **first open after ~1 min idle** | **0.15.53: 2026-08-18 `component`; the live idle case was not exercised**. Previous design passed 6/6 on 2026-08-06 `real host` |
| 3 | VS Code tunnel, Linux | vscode tab | large (88 MB, 4-D), fast local | **0.15.53: verified 2026-08-18 `real host`** — four large-array launches across two five-launch batches rendered on their first navigation; a ~3.25 s load was retained without replacement |
| 4 | VS Code tunnel, Linux | vscode tab | slow network mount (NFS/SMB) | **0.15.53: 2026-08-18 `component`; no network-mount launch was exercised**. Previous design verified 2026-08-11 `real host`, by the user |
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
| 13 | Jupyter remote (VS Code tunnel) | inline IFrame | small | **2026-08-19 `real host`: works — the array renders in the cell.** Root cause: a cell's output renders in a webview on the VS Code *client*, which has no listener on this host's `localhost` and no devtunnel cookie, so a private forward answers it with a GitHub sign-in redirect and the cell stays black. Measured directly: `curl` of the forwarded address from this host returns `302` to `.../auth/github`. The viewer *tab* is immune because VS Code fetches that page itself, remote-side, over the private route — which is why the tab kept working throughout and why three earlier sessions concluded "webview sandbox blocks it". It is not a sandbox: nothing is blocked, the address simply does not resolve to the backend. Fix: inline requests (only) ask the opener to publish that one port, restoring the pre-2026-07-29 `remote.tunnel.privacypublic` promotion verbatim (`resolvePublicTunnelBase` in `extension.js`, opener v0.15.57), over a dedicated `public-port-request-*.json` signal that does not touch the tab pipeline. Two obstacles found on the way, both fixed: (1) the `jupyter-server-proxy` relative-URL route was picked inside VS Code's own notebook, which has no Jupyter page to resolve it against — the page now picks the route from its own origin, because gating it in Python on `_in_vscode_terminal()` regressed a `jupyter lab` started from a VS Code terminal and opened in a browser (row 34); (2) `_viewer_port_url` / `_server_id_for_url` probed the loading-page stub instead of the backend on cold starts (row 16). Standing hazard: publishing needs a free forwarded-port slot, and VS Code auto-forwards every listening port it notices (MATLAB, kernel zmq, its own server), which exhausted this tunnel's `PortsPerTunnel` limit and produced a silent 26s black cell. Measured with Microsoft's `devtunnel` CLI: the cap is **10 ports per tunnel**, and 6 of the 10 were orphaned registrations no VS Code window showed (`devtunnel port list <tunnel>` names them; `ss` says nothing listens). That limit message is now shown in the cell instead of swallowed, and `remote.autoForwardPortsSource` is `output` on this host so process-scan forwarding stops refilling the tunnel |
| 14 | Jupyter any | native / browser | small | never verified |
| 15 | plain Python, local desktop | native (auto) | small | never verified |
| 16 | plain Python, VS Code tunnel terminal | vscode tab | small | **2026-08-19 `real host`, fixed** — the correlated port lease (`_viewer_port_url`, `_server_id_for_url`) was probing the loading-page stub instead of the real backend on every cold start over a tunnel: the stub answers any path including `/ping` with placeholder HTML, so the lease's JSON parse failed (or, once that was patched, `_server_id_for_url` silently returned `None` from the same stub-probing bug, making `expectedServerId` empty and the lease 400). Reproduced 100% on a cold kernel/process start; fixed by unwrapping the loading-page URL before probing. Confirmed against the user's live tunnel kernel, cold start, 4/4 |
| 17 | any | `window=False` | any | never verified — must return a URL and open nothing |
| 18 | any | multiple arrays (2-4 handles) | any | **verified 2026-08-06 `real host`** — `view(a, b, c)` returns 3 handles and opens one tab holding all three as a compare group, first try |

### VS Code Explorer click

| # | Environment | Window | Array / storage | Status |
|---|-------------|--------|-----------------|--------|
| 19 | VS Code tunnel, Linux | vscode tab | small, fast local | **0.15.53: 2026-08-18 `component`; no Explorer-click launch was exercised**. Previous design verified 2026-08-06 `real host`, by the user |
| 20 | VS Code tunnel | vscode tab | folder / collection | **current handoff change: 2026-08-14 `component`; real-host revalidation unavailable**. A valid collection passed on the previous design 2026-08-06 `real host`; misleading invalid-collection errors remain separate |
| 21 | VS Code Remote SSH | vscode tab | small | never verified |
| 22 | VS Code local (no remote) | vscode tab | small | never verified |

### Lifecycle (cuts across every row above)

| # | Case | Status |
|---|------|--------|
| 23 | Cold start — no server running yet | **0.15.53: verified 2026-08-18 `real host`** — the second five-launch batch began from no ArrayView server and rendered normally |
| 24 | Warm repeat — second array within ~10 s | **0.15.53: verified 2026-08-18 `real host`** — ten public CLI launches across two batches rendered without reloads, manual retries, or replacement tabs |
| 25 | Several viewers open at once | **0.15.53: verified 2026-08-18 `real host`** — five simultaneous viewers remained in five distinct tabs in each of two batches |
| 26 | Close one viewer, others keep working | **0.15.53: verified 2026-08-18 `real host`** — closing the middle of five released exactly that array while the same window and the other four viewers stayed alive; closing the rest released every session and the temporary server shut down automatically |
| 27 | `--kill` / shutdown leaves no orphan processes | **verified 2026-08-06 `real host`** — after `--kill`: no daemon processes, no listening ports (cold-start ports included), no stale claims, registry empty. A normal launch creates exactly one server. **But**: the server is `persist`ent by design, so it outlives the command that started it — including one interrupted mid-launch — and is only stopped explicitly. It stays discoverable via `arrayview instances` and killable, so it is a managed server rather than an orphan; a server from an *earlier session* was still running today and was found and stopped the same way |
| 28 | Repeat launch after `--kill` | **verified 2026-08-06 `real host`** — 5/5 clean, this is the cold-start path |
| 29 | Two+ windows open, terminal launch opens in **the window the terminal is in** | **0.15.53: verified 2026-08-18 `real host`** — with two Tunnel windows registered, all ten test launches were claimed only by the initiating window and rendered there in distinct tabs |
| 30 | Two+ windows open, Explorer click opens in **the window clicked in** | partially verified 2026-08-06 `real host` — clicks landed in the clicking window throughout; not tested with two windows side by side. Separately: sound on this host (the click writes to its own window's file), but on any platform with no IPC hook (documented: local macOS) the click writes a filename **no window watches**, costing ~12 s per click before it falls back |
| 31 | Two+ windows, an unfocused window must not claim another's launch | **not reachable** — the focus guard protects a broadcast path that current Python never writes; safety comes from every request being targeted, not from focus |
| 32 | Two+ windows, one alive but **not responding** | **failed 2026-08-11 `real host`; 0.15.53 same-tab recovery has 2026-08-18 `component` evidence** — ordinary real-host launches are clean, but none triggered recovery, so re-navigation inside the same tab is not yet induced-failure `real host` evidence; guided reload remains the bounded final fallback |
| 33 | Two windows, each with its own server on its own port | never verified — and **not a state the code maintains**: the instance registry has no concept of a window, so a second window silently reuses the first window's server unless `--port` is passed explicitly |
| 34 | classic Jupyter server on a remote (SSH), opened in the user's own browser | inline IFrame via `jupyter-server-proxy` | small | **2026-08-19 `component`** — the page picks the relative `/proxy/<port>/` route from its own `http` origin. Regressed and fixed the same day: a Python-side gate on `_in_vscode_terminal()` also fired for this row, because `jupyter lab` started from a VS Code terminal has VS Code in its process ancestry and is indistinguishable from a VS Code notebook kernel from Python. Verified by rendering the HTML for both origins, not against a real remote browser — an actual user runs this daily, so a `real host` check is worth doing. `ARRAYVIEW_JUPYTER_PROXY=1\|0` overrides |

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

**The private viewer port** (the running server leases one extra port to tunnel
launches):

- Explorer clicks and terminal launches now acquire the same port lease using
  the same request identity. Port acquisition failure stops with an explicit
  message instead of silently returning to the known-stale main port.
- That fast path still **hardcodes port 8000** and silently declines when the
  daemon is on any other port. Unfixed, and unrelated to the flicker.
- Rows 6 and 21 (Remote SSH): **guarded 2026-08-06** — the swap is now tunnel
  only. Under SSH the main port is forwarded because it is printed to the
  terminal and given port attributes; an ephemeral port gets neither, so it
  would have taken new risk for a benefit never measured there.
- Rows 11 and 17 (`--window none`, `window=False`): unaffected — the swap only
  happens on the VS Code signal path, and those return the original URL.
- Rows 5, 7, 8-10, 12-15 (browser, native, Jupyter): unaffected, same reason.
- Rows 23-26: the singleton, overlapping handoffs, exact lease consumption,
  abandoned-launch expiry, close-during-handoff, and reconnect grace have
  component coverage. Their real tunnel boundary remains open.
- Row 27 (orphans): the extra port lives in the existing process and dies with
  it. Component coverage proves bounded abandoned-lease cleanup; real-host
  final cleanup has not been re-observed for this design.
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
