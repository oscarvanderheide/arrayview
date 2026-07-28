# Handoff — ArrayView tunnel launch latency (2026-07-28)

**Read `.mex/ROUTER.md` and `plans/tunnel/LOG.md` before touching anything.**
This file is the short version of the LOG's 0.14.97–0.14.99 entries. The user
has asked for this problem ~40 times across many sessions; re-deriving what is
already written down is the specific failure mode that keeps producing bandaids.

Branch: `fix/relay-probe-hedging`, 6 commits, installed and verified as
extension **0.14.99**. Working tree clean.

---

## The user's experience, unchanged

Launch a file → fast. Close the tab, launch again → slow. Repeatedly, all day.
Latest measurement (0.14.99, after a window reload):

| launch | signal → first frame |
|---|---|
| 14:54:58 `large_array.npy` | **2.6 s** |
| 14:55:15 `coil_sensitivities.npy` | **20.1 s** |

Work so far took the bad case from ~32 s to ~20 s. **That is not enough to
change his experience and he is right to be annoyed.**

---

## The one durable finding — do not re-derive this

Measured directly against the live devtunnel (`v54z0psh-8000.euw.devtunnels.ms`),
30 fresh connections to `/ping`, backend healthy on loopback throughout, no VS
Code involved:

```
answered  25/30   median 185 ms   p90 233 ms   max 639 ms
stalled    5/30   no response, no reset, no error, to a 15 s budget
```

**There is nothing between 639 ms and never.** ~1 connection in 5 is accepted by
the relay and silently dropped. Stalls are largely independent per connection
but arrive in correlated bad windows lasting tens of seconds.

Three plausible theories were killed by measurement — don't revisit them:

- **Not a cold relay.** Across all 81 probe episodes since 07-26, idle time is
  uncorrelated with outcome (OK after 35,636 s idle = 0.5 s; UNVERIFIED after
  10.7 s idle = 13.7 s).
- **Not Node's keep-alive socket pool.** The stall reproduces with
  `reusedSocket=false`, on a brand-new connection.
- **Not saturation by open viewers** (the n=1 theory in the LOG). Stalls occur
  with zero viewers open.

Reproduce in two minutes with plain `node`/`curl` against the tunnel URL.
Scripts are gone with the session, but they are ~30 lines; see the LOG.

## What follows from it

Any relay request must use **short hedged attempts on fresh connections**, never
escalating waits. A budget wider than ~1 s buys literally nothing. Loopback
keeps sequential retry — it has no black-hole mode.

---

## What is fixed and verified working

1. **`hedgedProbeStatus`** — overlapping probes, first verdict wins. Replaced a
   1.5/4/8 s ladder that cost a flat 13.6 s on a stall. Confirmed in the field:
   the route check now costs 3.5 s worst case instead of ~20 s.
2. **A no-answer no longer discards a cached route** (`_usableCachedTunnelBase`).
   Measured 0/6: every route ever discarded on a stall was re-derived to the
   byte-identical URL, once past a downstream deadline into a spurious
   "Backend stopped answering". Only `PROBE_DEAD` discards now.
3. **`PROBE_RELAY_DOWN`** split from `PROBE_UNKNOWN`. A relay 502 (~200 ms,
   connector detached) is *information* and port promotion cures it; a stall is
   an absence of information. Do not re-merge these.
4. **Warmup budget 12 s → 2 s**, and it no longer blocks navigation beyond that.
   Confirmed in the field (`transport-warmup-failed` at exactly 2.0 s).

---

## What FAILED — revert or gate this first

**`portMapping` page delivery (commit `1c80bb1`) does not work in a tunnel
window.** This was my last hypothesis and the log disproves it:

```
14:54:58.702  wrapper-started
14:54:58.759  iframe-loaded          ← 57 ms: something loaded
14:55:00.327  delivery-fallback      ← but the viewer never booted
14:55:00.654  script-loaded          ← via the relay, as before
```

`delivery-fallback` fired on **both** launches. The mapped navigation to
`http://localhost:8000/?sid=…&data_origin=…` fires `load` almost instantly with
content that is not the viewer (most likely a VS Code error page — it was never
inspected), the boot watchdog fires 1.5 s later, and the panel falls back to the
relay.

Net effect: **+1.5 s of pure overhead on every single launch.** The fallback
worked exactly as designed, so nothing is broken — but the change is currently
net-negative and should be reverted or gated behind a probe of the mapped URL
before anyone builds on it.

Before discarding the idea entirely, one cheap check: open the webview
developer tools (`Developer: Open Webview Developer Tools`) on a launch and see
what the mapped iframe actually loaded. If VS Code is refusing the mapping
outright, the idea is dead. If it is a 404/redirect, it may be fixable.

The **server-side half is sound and worth keeping either way**: `GET /` accepts
`?data_origin=`, injects `window.__ARRAYVIEW_SERVER_ORIGIN__`, is validated
against `^https?://host(:port)?/?$`, and is provably inert when absent (served
bytes hash identically). It lets the page be delivered over one path while its
data uses another — whatever that first path turns out to be.

---

## The actual remaining problem

Launch 2's 20.1 s decomposes as:

```
 3.5 s  cached-route probe — all 3 hedged attempts black-holed
 1.5 s  failed portMapping delivery      ← regression, see above
 8.4 s  the 1.96 MB viewer page crossing the relay
 6.0 s  WebSocket open — ws-open-timeout, ws-error, then ws-open
```

Two things are now on the critical path, both on the relay:

### A. The 1.96 MB page, on every launch

- It is **byte-identical every time**. `md5` of `/?sid=A` and `/?sid=B` match
  exactly; only the no-sid variant differs (it embeds the latest session id).
  `get_ui` sets `query_val = "null"` whenever a sid is present.
- It is served `Cache-Control: no-store`, at a URL whose query changes per
  launch, so it is re-downloaded in full, every time, over the flakiest link.

Untried ideas, roughly in order of promise:

1. **Make it cacheable.** Serve the sid-present variant with a strong `ETag` +
   `immutable`, keyed on the arrayview version, and move the sid out of the
   query (fragment, or `__ARRAYVIEW_QUERY__` injection) so the URL is stable
   across launches. With `immutable`, Chromium serves it with *no network
   request at all*. Load-bearing unknown: whether the VS Code webview's HTTP
   cache persists across panel instances. **Verify that before building it.**
2. **Shrink it.** 1.96 MB of single-file HTML is the root cause of the cost.
   Nobody has profiled what dominates it.
3. **A local delivery channel** — extension host fetches over loopback (4 ms)
   and hands the HTML to the webview via `postMessage` + `srcdoc`. Needs CSP
   work and a query-injection seam, and `reportParent` no-ops when
   `window.parent === window`, so it must stay in an iframe. Both viewer seams
   (`__ARRAYVIEW_SERVER_ORIGIN__`, `__ARRAYVIEW_QUERY__`) already exist.
   Note: the `get-viewer-html` / stdio bridge described in older notes **no
   longer exists in `extension.js`** — grep before trusting any doc about it.

### B. The WebSocket is black-holed too

New in this log and not yet investigated: `ws-open-timeout` → `ws-error` →
`ws-open`, costing 6 s. The bounded-open logic (6 s × 3) is doing its job, but
the socket is subject to the same ~20% drop rate. **Fixing the page alone will
not make launches reliably fast.** VS Code does not remap WebSocket ports
(`_backendPortMapping`'s comment), so the socket cannot trivially move off the
relay — this may be the harder half.

---

## Traps that cost time today (all in the LOG)

- **`uv tool install --force --reinstall .` is not enough on its own** and the
  tool env is **not** at the default path — this box sets
  `UV_TOOL_DIR=/home/oheide/localscratch/.cache/uv-tools`, and a *stale* env at
  `~/.local/share/uv/tools/arrayview` sits at 0.14.89. Resolve with `uv tool
  dir`, and verify with a **code marker from your change** inside the bundled
  VSIX, not just a version string.
- **`git stash push -- <path>` on an already-committed file is a silent no-op.**
  It produced a "baseline" that was the same build compared against itself. To
  baseline, `git checkout <rev> -- <file>` and *prove* the revert landed with
  `git diff HEAD --stat`.
- **There is an unrelated pre-existing stash** at `stash@{0}`
  ("final-baseline", loading-screen work from another worktree, includes
  untracked files). Do not pop it. It was accidentally popped and restored
  intact today.
- **`tests/test_lifecycle_contract.py` hard-codes port 8000** and fails whenever
  a real ArrayView session is running. Not a real failure.
- **`tests/test_browser.py` has 34 pre-existing failures** on this machine
  (verified against a true baseline). Compare sets, not counts, and expect
  `TestKeyboard::test_initial_scoped_volume_range_stays_fixed_without_opening_histogram`
  to be order-flaky.

## Method note

Twice in a row the diagnosed cause was real, the fix was correct and verified,
and the user saw no improvement — because the same pattern existed at another
layer. **Decompose the entire observed wall time before claiming a latency fix
works**, and check what now holds the largest share. Do not report a fix as a
win on the strength of the component it improved.
