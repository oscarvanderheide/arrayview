# Handoff for review: opener 0.15.19, hedge token rotation

**Status: unreviewed, uncommitted, not installed. Do not treat the fix as
correct — the author's own confidence dropped while writing this document.**

Reviewer: assume the diagnosis is wrong until you have re-derived it. The
specific reason to distrust it is in "The claim, and why it is probably too
weak" below, written by the author against their own change.

## The user's report

An Explorer click on `1226601.nii.gz` in a VS Code desktop-tunnel window
"bounced a few times" before the array appeared. The user's requirement, stated
plainly: opening an array from the CLI or by clicking should just work —
one tab, no retries, no reload, immediately.

## Observed real-host evidence

`~/.arrayview/extension.log:11097-11135`, window `0bf6eeb4`, opener 0.15.18.
Both live tunnel windows ran 0.15.18. The click path ran
`/home/oheide/localscratch/projects/anne_ruckert/detection/.venv/bin/python`,
which is an editable install of the working tree at 0.42.0 — **no version
skew**, this is current code.

```
13:12:04.9   click; FASTLOAD: no daemon on port 8000; fresh Python launch
13:12:06.97  integrated browser tab opened
13:12:08.68  closed exact blank tab; retrying navigation attempt=1
13:12:10.41  closed exact blank tab; retrying navigation attempt=2
13:12:12.15  closed exact blank tab; retrying navigation attempt=3
13:12:12.62  script-loaded
13:12:12.85  frame-rendered
```

~8 s click-to-frame, three visible tab close/reopen cycles. A second episode the
same day at 09:48 recovered on hedge 2. These are the **first two firings of the
multi-hedge ladder** since 0.15.9, which the LOG recorded as never having run.

Two comparison launches minutes earlier (`0078709_0000.nii.gz`,
`0078709.nii.gz`) went straight through in ~1.5 s and ~0.4 s, zero retries. The
label files are the same size (~49 KB), so array size is not the variable.

## What was changed

Four files. `git diff` is the authority; this is the summary.

**`src/arrayview/_routes_query.py`** — the substantive change.
- The journal keeps every token a request has issued (`tokens`, capped at 8)
  instead of only the newest.
- `record_viewer_phase` and `get_viewer_phases` accept any token in that list
  rather than only `journal["token"]`.
- On a hedge's `launch-prepared`, `phases`, `viewer_instance_ids` and
  `related_sids` now **carry forward** instead of being reset to empty.
- Inheritance requires the same `window_id`; a different window still gets a
  fresh journal. Foreign windows and unknown tokens still 409.
- `launch-prepared` now returns the carried state rather than empty lists.

**`src/arrayview/_server.py`** — instrumentation only.
- `journal["document_requests"]` counts page requests actually served for a
  launch, incremented in `get_short_viewer_ui`.

**`vscode-extension/extension.js`**
- Logs `document_requests` as it changes.
- The duplicate-viewer-instance guard now fails the launch only when
  `navigationAttempt === 0`; after a hedge it logs instead of failing.

**Version markers**: `vscode-extension/package.json` and `_VSCODE_EXT_VERSION`
both 0.15.19. VSIX rebuilt, `extension.js` byte-identical to the working tree,
sha256 `20ad137cfd1d641f5271cb29ba661c5c927aadd4754148caaf52361a7e504da0`.

## The claim, and why it is probably too weak

**The claim.** Every hedge POSTs `launch-prepared`, which replaced the journal
with a fresh token. A delivered page reports readiness with the token baked into
its own URL. `record_viewer_phase` rejected any non-current token with 409, and
`reportParent` in `_viewer.html:5910` swallows the failure (`.catch(() => {})`).
So a page from a superseded attempt could never report readiness.

**This part is verified.** See "Evidence" below — on unmodified HEAD the
superseded report is rejected with `409 Viewer phase owner changed` and the
phase list stays empty.

**Why it probably does not explain the observed bouncing.** In
`extension.js`, the retry callback closes the tab *before* calling
`prepareNavigation`, and `prepareNavigation` is what POSTs `launch-prepared` and
rotates the token:

```js
const closed = await tabGroups.close(requestTab, true);
...
return prepareNavigation(navigationAttempt, deadline);
```

So by the time the token rotates, the previous attempt's tab is already
disposed and its page is dead. A dead page does not POST. The 409 can therefore
only bite in the few-millisecond race between the readiness poll and the tab
close — far too narrow to explain a three-bounce, 8-second episode.

**Conclusion the author reached too late:** the 409 defect is real but is
probably not the cause of what the user saw. The root cause of the initial blank
navigation remains **unidentified**. The author told the user "that's your
bouncing" before checking the ordering. That statement was wrong.

## Regression risk introduced — review this first

Carrying `phases` forward across hedges may be actively harmful.

The hedge is gated on `!scriptLoaded`. If a superseded attempt records
`script-loaded` and that attempt's tab is then closed, the phase now persists
into the next attempt's journal. The opener sees `scriptLoaded === true`,
**stops hedging**, and waits for a `frame-rendered` that a dead tab will never
produce. The launch then fails at the inactivity deadline instead of recovering.

Under the old reset-per-attempt behaviour this could not happen.

The same narrow race that limits the benefit also limits this risk — both live
in the poll-to-close window. But the risk is a converted failure mode, not a
slower success, so it is not symmetric with the benefit. **A reviewer should
decide whether carrying `phases` forward is justified at all, or whether
accepting superseded tokens without carrying phases is the correct subset.**

Second, smaller: relaxing the duplicate-instance guard after a hedge removes a
check that previously caught one request producing two live tabs. The
justification is that a superseded attempt reporting late from a closed tab is
expected once phases carry forward. If phases stop carrying forward, revisit
this too.

## Evidence

**`real host`** — the two bounce episodes, the version state, and the absence of
skew. Read from `~/.arrayview/extension.log` (append-only; snapshot byte size
and read only the delta).

**`component`** — the launch handshake driven directly against the real route
table via `starlette.testclient` against `arrayview._server.app`, using a
pending SID (`launch-prepared` and `script-loaded` are control-plane phases that
do not wait for the data session). Probe:
`/tmp/claude-1885/-localscratch-oheide-projects-arrayview/c1c5b4f3-8366-4ae1-ad27-de6ace5108d7/scratchpad/probe.py`

On unmodified HEAD:

```
script-loaded via superseded token0 -> 409
journal on token1 -> phases=[]
document_requests -> None
```

With the change:

```
script-loaded via superseded token0 -> 200
journal on token1 -> phases=['script-loaded']
document_requests -> 1
foreign window -> 409
unknown token -> 409
```

The baseline was taken by `git checkout HEAD -- <files>` and confirming an empty
`git diff HEAD --stat`, per the stash trap recorded in the 0.14.99 LOG entry.

**Test suites**: 18/18 `vscode-extension/test_*.js`, 17/17
`tests/test_vscode_ack_protocol.py`. One test,
`test_integrated_browser_readiness.js`, was failing on the duplicate-instance
assertion and **the extension was changed to accommodate it** — verify that was
legitimate and not the test being bent to fit.

**`unavailable`** — no real-host launch with 0.15.19. Not installed, windows not
reloaded. It is not known whether this removes the bouncing.

## Discarded mid-work — do not retry without reading why

A small bootstrap shell serving the ~2 MB viewer body as a separate in-page
fetch, to move retries off the tab. Built, then deleted. Two LOG entries kill
it:

- **0.15.3** measured silent drops *independently* for a small probe, a large
  document, metadata, and a WebSocket upgrade. A smaller document is not a safer
  one.
- **0.15.7** put this path on `remoteProxy=true`, so the document does not cross
  the devtunnel at all. Today's log confirms it: `REMOTE: desktop
  integrated-browser proxy uses backend URL directly`.

Also discarded: the argument that "four consecutive drops at ~20% is only 0.5%
likely, so it must be a new bug." **0.15.6** documents correlated bad windows
where ten candidates went completely silent for 46 s. Correlated failure is
established behaviour here, and that reasoning was invalid.

## Required reading before reviewing

`plans/tunnel/LOG.md` — particularly 0.14.99, 0.15.3, 0.15.6-0.15.7, 0.15.9,
0.15.10, and the new 0.15.19 entry. This investigation has five prior versions
of retry-ladder tuning with the root cause marked unconfirmed throughout. The
dominant failure mode of this workstream is re-deriving a plausible mechanism
and shipping a tuned ladder on top of it. Treat 0.15.19 as a candidate for
exactly that mistake.

`plans/tunnel/PLAN.md` for the development loop and evidence discipline.
`.mex/patterns/validate-launch-path.md` for the acceptance contract.

## Questions the reviewer should answer

1. Is carrying `phases` forward correct, or should the change be narrowed to
   token acceptance only? Argue from the poll-to-close race window.
2. Can a dead attempt's `script-loaded` suppress hedging and convert a
   recoverable stall into a deadline failure? Construct the interleaving or
   prove it cannot happen.
3. Was relaxing the duplicate-instance guard justified, or was a real check
   weakened to make a test pass?
4. If the 409 is too narrow to explain the observation, what does explain a
   three-bounce 8-second stall on a `remoteProxy=true` loopback path where the
   document never touches the devtunnel? The `document_requests` counter is
   intended to answer this on the next stall; check that it is sound and that
   the count is not reset by the hedge it is meant to measure.
5. Is the 1.5 s hedge cadence (`preScriptTimeoutMs * 0.15`) defensible at all
   for a path whose successful loads have never been slower than 1.5 s — i.e.
   is the cadence selecting the outcome it then measures?

## State of the working tree

Modified, uncommitted:

```
plans/tunnel/LOG.md
src/arrayview/_routes_query.py
src/arrayview/_server.py
src/arrayview/_vscode_extension.py
src/arrayview/arrayview-opener.vsix
vscode-extension/extension.js
vscode-extension/package.json
```

No orphan processes; the test server on port 8123 was stopped and confirmed
down. The user's profile was not modified and no window was reloaded.

Note: during testing the author ran `arrayview --window browser`, which routed
into the integrated browser and may have opened a tab in the user's window.
