# VS Code tunnel launch repair — plan

Requirements: `plans/tunnel/REQUIREMENTS.md`
Log: `plans/tunnel/LOG.md`
Skill: `.claude/skills/iterative-debug` (one hypothesis per round, no
self-declared success)

## Goal

Every row of the acceptance matrix in `plans/tunnel/REQUIREMENTS.md` passes in a
real tunnel window, repeatedly, without manual intervention.

## Development loop

Debugging runs the **working tree**, not PyPI. The extension's uv invocation is
unchanged in shape; only the package spec is redirected:

- `arrayview.packageSpec` (extension setting, `machine-overridable`) is set to
  `/localscratch/oheide/projects/arrayview` in
  `~/.vscode-server/data/Machine/settings.json`.
- An absolute path makes the extension use `--with-editable <path>`, so Python
  edits apply on the next launch with no reinstall.
- Measured cost: `--with arrayview` 0.25 s warm; `--with-editable <repo>`
  0.42 s. The extra ~0.17 s does not change timing behaviour materially.
- `uvx`/PyPI is out of the loop until the final release, which is row T12.

Extension changes still require: version bump in
`vscode-extension/package.json` **and** `_VSCODE_EXT_VERSION` in
`src/arrayview/_vscode_extension.py`, VSIX rebuild, reinstall, and a window
reload per window.

## Evidence discipline

- `~/.arrayview/extension.log` is append-only across days. Snapshot its byte
  size before each test and read only the delta. Never read the whole file.
- Terminal ACK evidence: newest `~/.arrayview/open-ack-v0100-*.json`.
- A `PASS` requires a rendered frame (R9). A tab, a port, a socket, or a
  `backend_ready` without a visible array is not a pass.

## Ordered hypotheses

Worked one at a time, most-blocking first. Each becomes a numbered attempt in
the log.

**H1 — the processing lock has no failure path.**
Evidence: 2026-07-25 19:40:48 → 19:42:18. Request `51fef3af` failed with
"Viewer did not render a frame before timeout". Request `45e5a0bb` then set
`isProcessingSignal=true` and the log shows nothing but `SKIP:
isProcessingSignal=true` once per second for ~90 s, until the 240 s expiry timer
released it. Request `4bbaa131` was queued behind it and expired unserved at
`ageMs=215046`. Violates R7 directly, and converts any single failure into a
dead window — the likely cause of "works the first time, not the third".

**H2 — first-frame readiness is the underlying failure.**
Three distinct terminal states in six minutes: "Viewer did not render a frame
before timeout", "Viewer panel closed before its first frame rendered", "Signal
expired during extension-host recovery". H1 explains why one failure cascades;
it does not explain the first failure. Root-cause this only after H1, so that
each test round produces one clean signal instead of a cascade.

**H3 — cross-window cleanup is not owner-fenced.**
Window `910d7fbf` was repeatedly logging `CLEANUP: retained active claim` over
window `1bdd1ffc`'s claim files. Retention was correct here, but a second window
inspecting and acting on another window's in-flight claims is a live race
surface. Violates R4. Confirm whether any cleanup path can remove or expire a
peer's live claim.

**H4 — readiness deadlines are fixed, not scaled to load cost.**
R8. Deadlines observed: 190 s and 240 s signal expiry, plus a shorter frame
deadline. A large array on `/smb` may legitimately exceed the frame deadline.
Not yet evidenced — do not act on this before H1 and H2 produce clean traces.

## Ruled out

- **uv resolution latency.** Hypothesised that `uv run --directory /tmp
  --no-project --python 3.12 --with arrayview` re-resolves against the network
  on each Explorer click. Measured 2026-07-25: 0.65 s cold, 0.25 s warm, 0.19 s
  with `--offline`. Not a contributor under current conditions. May still matter
  on a cold uv cache or a degraded network; not the current cause.

## Baseline established 2026-07-25

- Extension directories reduced to one live build; `.obsolete` and both
  `extensions.json` registries reconciled.
- `~/.vscode/extensions/extensions.json` referenced
  `arrayview.arrayview-opener-0.14.70` while the directory on disk was named
  `...-0.14.70.broken`. That dangling reference is the silent no-load failure
  documented in `plans/webview/LOG.md`. Registration removed, directory deleted.
- Stale `open-ack-*.json`, `window-*.json`, and `tunnel-routes.json` cleared
  (`tunnel-routes.json` held 10 window ids, none of them current).
- `extension.log` rotated to `~/.arrayview/archive/`.
- Backups: `~/.arrayview-cleanup-backup-20260725/`.
- Test arrays on bulk storage:
  `/smb/.../securo/arrayview-tests/{small_64,vol_128}.npy`.
