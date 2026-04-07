# ArrayView UI Maturity Strategy

> **Authored:** 2026-04-07. This is a *strategy* document, not a TDD-style implementation plan. Once you pick a phase to start, a concrete implementation plan can be written for that specific phase. Do not try to execute the whole thing in one go — that is the second-system trap, and it would be the wrong response to the kind of pain you're describing.
>
> **Source reports** (read these if you want the raw evidence):
> - `dev/plans/ui-maturity-codebase-report.md` — codebase deep-dive: mode transition map, code-path duplication inventory, state topology, keyboard handler audit, reconciler audit, commit-history bug pattern analysis, test coverage gaps, ranked root causes
> - `dev/plans/ui-maturity-external-research.md` — external research: napari, 3D Slicer, ImageJ2, tldraw, Excalidraw, VS Code, Blender, plus 9 patterns with fit analysis and 4 concrete test recipes

---

## TL;DR

The bugs you keep hitting are not random. They are four specific architectural shapes, all of which the codebase has *already started to fix* and then stopped at ~10–30% completion. The strategy is **not to introduce new architecture; it is to finish the things you already started, in a specific order, with a test safety net under them.**

The four shapes, in priority order:

1. **State save/restore is asymmetric and incomplete.** `collectStateSnapshot()` exists but only captures ~25 fields out of ~180 globals. Half of mode-exit handlers don't call it. That single fact directly explains your "enter MIP, change colorbar, exit, change is gone" complaint. (~35% of historical bugs.)
2. **Code paths fork on N=1 vs N≥2 (and on mode), then drift.** Five near-identical scale functions. Six near-identical colorbar drawers. The "I changed the colorbar but it didn't take effect with two arrays" complaint is this exact bug. (~25%.)
3. **Keybinds dispatch through a 1300-line switch with hand-rolled per-key mode guards.** New keybinds silently fail in modes the author forgot. There is no registry, no `when` clause, no "is this command reachable from this mode?" question you can ask. (~10–15%.)
4. **The reconciler pattern is the right idea but only ~30% covered.** `_reconcileUI()` handles container visibility, compare sub-mode classes, and colorbar visibility — but not eggs, projection mode, MIP, mosaic, or anything keyboard-driven. It's called from ~30 sites instead of from one place after every state change. (~15%.)

**These four explain ~85% of the bugs you've been swatting.** None of them require throwing anything away, none require a JS framework, none require a rewrite. They require finishing the existing patterns, in a specific order, with tests written *first* as a safety net.

The **one** thing you should do right now (before any refactor) is write **a round-trip pytest matrix** that exercises every mode by entering it, mutating something, exiting, and asserting state is unchanged. ~50–100 lines of test code, no production-code change. It's both a bug detector and a regression net for everything that follows.

**Do not** install XState. **Do not** rewrite the renderer. **Do not** model `compare > diff > wipe` as a hierarchical statechart. These are second-system traps that the external research explicitly flags as failure modes for projects this size. The whole strategy is incremental.

---

## Part 1 — The bug classes, plain language

You gave four examples. Let me restate them in terms of what's actually happening so the strategy maps cleanly:

| Your phrasing | What's actually broken | Architectural shape |
|---|---|---|
| "Changed colorbar, doesn't work with two arrays — different code path" | Single-array goes through `drawSlimColorbar` / `scaleCanvas`; compare goes through `drawComparePaneCb` / `compareScaleCanvases`. The two implementations diverged. The fix you applied to one didn't propagate. | Code-path duplication (Class 2) |
| "In some mode, doesn't enter immersive view when it should" | The `F` / `K` keybind for fullscreen/zen is checked in some modes but not all. There's no registry that can answer "is fullscreen reachable from compare+qMRI?" without grepping. | Keybind dispatch (Class 3) |
| "Enter 3D MIP, exit, doesn't return to previous state" | `enterMultiView` saves zoom but not `mvDims`. `exitMultiView` doesn't call `_reconcileUI`. `mipActive` isn't even checked in the colorbar reconciler. The exit half of the round-trip is missing. | Save/restore asymmetry (Class 1) + reconciler gap (Class 4) |
| "Updated colorbar in some mode, exited, change forgotten" | The field you changed isn't in `collectStateSnapshot()`. The snapshot captures ~25 fields out of ~180 globals; the field you touched is one of the ~155 it doesn't track. | Save/restore asymmetry (Class 1) |

You said you suspect "combinatorial explosion of features × modes × keybinds." That's accurate but slightly misleading: the number of *valid* states is large but tractable. The problem is that the *implementation* duplicates state-handling logic across many places, so each new state requires touching N files instead of one. Reduce N to 1 and the explosion goes away.

You also said "it feels like guesswork/patchwork a lot of the time." That's because it *is*: the codebase doesn't make invariants checkable. There's no place to ask "show me everything that depends on `compareActive`" — instead, `compareActive` is read in 100+ inline conditions throughout a 15kloc file. The strategy below is about making invariants checkable and bugs unrepresentable, not about adding more checks.

---

## Part 2 — Diagnosis (converged from both reports)

This section is dense; skim if you want and go to Part 3.

### What already exists and is correct (don't throw away)

| Thing | Where | Status |
|---|---|---|
| `_reconcileUI()` and 3 sub-reconcilers | `_viewer.html:14043–14147` | **~30% complete.** Handles layout container visibility, compare sub-mode classes, colorbar visibility. Called from ~30 sites, not all. |
| `collectStateSnapshot` / `applyStateSnapshot` | `_viewer.html:6486–6637` | **~14% complete** (25 fields out of ~180 globals). Captures display state well. Misses orthogonal features (ROI, ruler, FFT, mask, RGB, lebesgue), modal flags (zen, compact, fullscreen), per-mode toggles (qmriCompact, mipActive), drag positions, animation state. |
| `compareCenterMode` enum + `_setCompareCenterMode` | `_viewer.html:2900-onwards` | **Done well.** This is the model the rest of the codebase should follow: one enum, one setter, derived flags. The other top-level mode flags (`compareActive`, `multiViewActive`, `qmriActive`, …) should follow suit but currently do not. |
| `ColorBar` class | `_viewer.html:~3800` | **~40% complete.** Used for primary colorbar and diff colorbars. Multi-view and qMRI still use raw inline canvas loops. Per the project memory, the migration was started and stalled. |
| `ModeRegistry.scaleAll()` | called from mode exits | **Exists** but is parallel to `_reconcileUI` instead of unified with it. Two separate "post-mode-change" hooks. |
| `tests/ui_audit.py` with R1–R35 + DOM assertions + runtime `_validateUIState` | tests/ + `_viewer.html:13884` | **Good infrastructure.** Catches visibility/positioning regressions. Does **not** catch round-trip state loss, cross-mode equivalence, or unguarded keybinds. |
| `tests/test_mode_consistency.py`, `tests/test_mode_matrix.py` | tests/ | **Good coverage of static mode states.** Same gap as ui_audit: no round-trip, no cross-mode equivalence. |
| `dev/mode_matrix.md` | dev/ | **Authoritative reference.** Use it as the source of truth for which modes exist. |
| `dev/lessons_learned.md` (Reconciler Pattern section) | dev/ | **Already prescribes the right pattern.** It just isn't applied everywhere. |

### What's broken — by class

**Class 1 — Asymmetric state save/restore (~35% of historical bugs)**

- `enterMultiView` saves zoom; `exitMultiView` doesn't restore `mvDims`. Doesn't call `_reconcileUI`.
- `enterQmri` saves zoom + qmriDim; `exitQmri` doesn't restore the `qmriCompact` toggle (because `qmriCompact` is not in `collectStateSnapshot`). Doesn't call `_reconcileUI`.
- `enterCompareMode` is the only mode that round-trips correctly via `applyStateSnapshot` — but it skips reconcilers on exit.
- Projection mode, MIP, mosaic toggle directly with no save/restore at all.
- `_savedQmriCompact` doesn't exist as a variable. `mipActive` is not in any snapshot. Drag positions in immersive mode aren't reset on exit (rule R30 was added precisely because of this bug class).
- Animation state (`_mvCrosshairAnim`, `_complexFadeTimer`, `_cmvCrosshairAnim`) persists across mode exits. Timers can fire after the panes they target are gone.

**Class 2 — Code-path duplication (~25% of historical bugs)**

- Five scale functions, ~85% identical: `scaleCanvas`, `mvScaleAllCanvases`, `compareScaleCanvases`, `qvScaleAllCanvases`, `compareMvScaleAllCanvases`, plus `compareQmriScaleAllCanvases`. Roughly **~1800 redundant SLOC**. Bug fixes apply to one and not the others — git log has 8+ commits where a scale fix in one mode was later replicated to another.
- Six colorbar drawers. The ColorBar class started to consolidate them; multi-view and qMRI never migrated.
- Backend rendering endpoints (`/slice`, `/diff`, `/oblique`, `/grid`, `/gif`, `/exploded`) share ~60% of their logic by inline copy.
- Mouse listener attachment is per-canvas-class with similar pan/zoom logic duplicated.
- Colormap / dynamic-range read-paths know their own variable names (`currentVmin`, `cmpManualVmin[i]`, `_diffManualVmin`, per-map vmin).

The shape of this class is the **N=1 vs N≥2 fork**. The single-array path was written first; the multi-array path was added later and copy-modified; subsequent fixes to the single-array path were not always mirrored.

**Class 3 — Keybind dispatch (~10–15% of bugs)**

- The keydown handler at `_viewer.html:8288` is a **1309-line giant switch** with inline mode guards. Sample of the actual structure (from grep):

```js
if (compareActive || multiViewActive || qmriActive) { … }
if (!compareActive) { showStatus('compare: off (press B to enter)'); return; }
if (compareActive && _flickerActive) { … }
} else if (compareActive && _checkerActive) { … }
} else if (compareActive && _wipeActive) { … }
} else if (compareActive && registrationMode) { … }
} else if (hasVectorfield && !overlay_sid && !compareActive) { … }
if (multiViewActive) { _mvVfieldCache.clear(); mvDrawAllVectorOverlays(); }
if (qmriActive || compareQmriActive) { showStatus('FFT: not available in qMRI mode'); return; }
```

- Six unguarded keybinds where the code blindly mutates state without checking applicable modes (`v` in compare, `p` in multi-view, `q` in compare, `z` in multi-view, etc.).
- No `when` clause. No registry. No way to ask "is `c` (cycle colormap) reachable in qMRI?" without grepping.

**Class 4 — Reconciler gaps (~15% of bugs)**

- `_reconcileUI` is called from ~30 sites, not from one place after every state mutation.
- `_reconcileCbVisibility` doesn't check `mipActive`, `projectionMode`, or `qmriMosaicActive`.
- Mode entry/exit functions sometimes call `ModeRegistry.scaleAll()`, sometimes call `_reconcileUI()`, sometimes call neither.
- There is no rule that says "every state change must call the reconciler." The pattern is documented in `dev/lessons_learned.md` but not enforced.

### Existing infrastructure that the strategy will leverage

You don't need to build:
- A test framework (you have pytest + Playwright + visual_smoke + ui_audit)
- A reconciler primitive (`_reconcileUI` is the seed)
- A state snapshot mechanism (`collectStateSnapshot` is the seed)
- An enum-based mode model (`compareCenterMode` is the seed)
- A class-based colorbar (`ColorBar` is the seed)
- A mode reference doc (`dev/mode_matrix.md`)
- A consistency skill (`ui-consistency-audit`, `modes-consistency`)

You need to *finish* each of these and *connect* them under one rule.

---

## Part 3 — The strategy: four pillars

The whole strategy is summarized in one sentence:

> **One store, one reducer, one render, one command registry — and one round-trip test that proves it.**

Each pillar finishes a pattern that's already partially in the codebase. None of them is novel.

### Pillar A — One store, one snapshot, one round-trip rule

**Finish `collectStateSnapshot`. Make it the *only* place state lives that needs to survive a mode round-trip. Make every mode enter/exit go through it.**

Currently the snapshot covers display state well but misses ~85% of the state space. The fix is mechanical:

1. **Audit every `let`/`var`/`const` at module scope in `_viewer.html`** — there are ~180. Classify each as: (a) display state (in snapshot), (b) orthogonal feature state (ROI, ruler, FFT, RGB, mask, lebesgue, vfield density, etc.), (c) per-mode toggle (qmriCompact, mipActive, mosaic), (d) modal/transient (animation timers, drag positions, render pipeline), (e) immutable session config.
2. **Move every (b), (c), (d non-transient) field into the snapshot.** Add it to both `collectStateSnapshot` and `applyStateSnapshot`. Where the field has constraints (range, type), put the validation in `applyStateSnapshot`.
3. **Establish the rule:** every `enterX` and `exitX` function calls `collectStateSnapshot()` on entry, stores the result on a per-mode `_savedX` variable, and calls `applyStateSnapshot(_savedX)` on exit. No exceptions. The compare-mode enter/exit pair is the model.
4. **Establish a second rule:** every `exitX` ends with `_reconcileUI()`. Same for every `enterX` after the snapshot is restored.
5. **Cancel pending animations on mode exit.** Add a `_cancelAllPendingAnimations()` helper that clears all known timer/RAF handles, called from every mode exit.

This is the highest-impact change because it doesn't require restructuring anything. It's pure additive. You can do it one mode at a time, ship each one, and watch the bug class shrink.

**The point of this pillar:** "did the user change something that should round-trip?" becomes a *type-system* question: if it's in the snapshot, it round-trips; if it's not in the snapshot, it explicitly doesn't. No more "I forgot to save that field." The list of fields IS the contract.

**Photoshop variant (optional, Phase 5):** instead of a single global snapshot, store per-mode UI state on the mode object itself (compare's wipe position, qMRI's compact toggle). Switching modes preserves *each mode's* state automatically. This is the napari-layer-options model. Defer until Pillars A/B/C are done.

### Pillar B — One renderer, always-a-list

**Replace the N=1 / N≥2 fork with a list-of-arrays model. The render path becomes a loop. The N=1 case is `arrays.length === 1`.**

This is napari's model and it directly answers your "I changed the colorbar but it didn't take with two arrays" complaint. From the external research: napari has *no* single-image code path because every operation is a `LayerList` operation; the active layer is just `layers.selection.active`. There's no place for the two paths to drift because there's only one path.

Concretely for arrayview:

1. **Single state slice for all arrays.** Instead of `sid` (single) plus `compareSidList` (multi), have one `arrays: [{sid, name, vmin, vmax, colormap, …}]` list. Length 1 is the normal case. Length 2+ is compare. The `active` index is just `state.activeIdx`.
2. **Render becomes `for (const a of state.arrays) renderPane(a)`**. No "if compare, draw N panes; else draw 1 pane." The N=1 case is one iteration.
3. **The five scale functions collapse into one parameterized function.** It takes a list of panes, a layout (1×1, 1×N, M×N grid, multi-view 3-plane), and a viewport. The mode-specific layouts become *layout strategies* the function dispatches over. ~1800 SLOC become ~400.
4. **The six colorbar drawers collapse into the existing `ColorBar` class.** Multi-view becomes "three ColorBars in three slots." qMRI becomes "N ColorBars in N slots." Compare becomes "one ColorBar per pane plus a shared one in non-diff modes." Each pane owns a `ColorBar` instance; nothing is drawn inline.
5. **Commands that genuinely need ≥2 arrays declare it via a `when` clause** (Pillar C). Diff/wipe/registration become commands gated by `arrays.length >= 2`. There's no other N-gate anywhere in the codebase.
6. **Backend endpoints unify:** `/slice`, `/diff`, `/oblique`, `/grid` share their setup helpers via the dependency you already added (`get_session_or_404`) and a new `RenderRequest` model. The 60% of common logic moves into a `_render_pane(session, params)` helper.

**The point of this pillar:** drift between code paths becomes structurally impossible because there's only one path. The N=1 path is the same code as the N=4 path.

**This is the biggest refactor in the strategy.** It's also the one that pays off the most over the long term. It must be done *behind* a green test suite (Pillar D), not naked. Plan it as a multi-week effort and ship it pane-by-pane.

### Pillar C — One command registry with `when` clauses

**Every user action becomes a command object. Keybinds are entries in a table that points at command IDs. Commands declare when they're enabled. The 1309-line keydown switch becomes a 50-line dispatcher.**

This is the VS Code / Blender / ImageJ2 model. From the external research: VS Code's `when` clauses are the gold standard for "the same command works correctly across editor/notebook/diff/terminal" — exactly your "feature in one mode bypasses another mode" problem.

Concretely:

```js
// commands.js — sketch
const commands = {
  'colormap.cycleNext': {
    title: 'Cycle colormap',
    when: '!qmriActive && !compareQmriActive && !rgbMode',
    run: (ctx) => { /* the body that's currently inline in the switch */ },
  },
  'compare.toggleDiff': {
    title: 'Toggle diff sub-mode',
    when: 'compareActive && arrays.length >= 2',
    run: (ctx) => { … },
  },
  'mode.toggleZen': {
    title: 'Toggle zen / immersive',
    when: 'true',  // works everywhere
    run: (ctx) => { … },
  },
  …
};

const keybinds = [
  { key: 'c', command: 'colormap.cycleNext' },
  { key: 'X', command: 'compare.toggleDiff' },
  { key: 'F', command: 'mode.toggleZen' },
  { key: 'K', command: 'mode.toggleZen' },  // alias
  …
];

// dispatcher
function onKeyDown(e) {
  const ctx = makeContext(state);  // booleans derived from state
  const binds = keybinds.filter(b => b.key === e.key);
  for (const b of binds) {
    const cmd = commands[b.command];
    if (evalWhen(cmd.when, ctx)) {
      cmd.run(ctx);
      return;
    }
  }
  // no match — log it for diagnostic, don't silently swallow
}
```

The `when` evaluator does **not** need to be a parser. A simplest-thing-that-works version is an array of required context keys: `when: ['compareActive', 'arrays.length>=2']`, evaluated against the context bag. You can upgrade later if you want operators.

**Why this is essential:**

1. **No more silent keybind drops.** If a key didn't fire, it's because no command's `when` matched — and the dispatcher logs that fact. You can answer "why did `p` do nothing in multi-view?" by looking at the dispatcher log, not by grepping.
2. **Help overlay is generated, not hand-written.** Iterate the registry, group by category, print `key — title`. Forevermore in sync.
3. **Command palette for free.** A `/`-dialog that lists every command, shows which are enabled in the current state, and lets the user run them by name. This is the single best UX upgrade you can give power users, and it's free once the registry exists.
4. **"Is this command reachable from this mode?" is a unit test.** `pytest.mark.parametrize` over modes × commands. The matrix lives in the test, not in your head.
5. **Every command has one place to find its body.** Refactoring the colormap-cycle is one file edit, not "find the right `case 'c':` in a 1300-line switch."

The migration is gradual. You can keep the existing switch as a fallback while you move commands one at a time, and delete the switch when it's empty. ~30–50 commands total based on the keybind table in `dev/mode_matrix.md`.

### Pillar D — One test that proves it: round-trip + hypothesis

**Before any refactor, write a round-trip pytest matrix. After Pillar B, layer Hypothesis stateful tests on top. Keep Playwright snapshots small and intentional.**

Three layers, in order:

#### D1 — Round-trip pytest matrix (write this first, ~50 lines)

```python
# tests/test_mode_roundtrip.py — sketch
from itertools import product

ALL_MODES = ['normal', 'multiview', 'compare', 'diff', 'wipe', 'flicker',
             'checker', 'registration', 'qmri', 'qmri_compact',
             'compare_mv', 'compare_qmri', 'mip', 'projection', 'mosaic',
             'zen', 'fullscreen', 'compact']

ALL_PERTURBATIONS = [
    'cycle_colormap', 'cycle_dynamic_range', 'toggle_log', 'toggle_rgb',
    'toggle_complex', 'enter_roi_rect', 'enter_ruler', 'toggle_vfield',
    'toggle_pixel_info', 'toggle_fft', 'change_slice', 'manual_window',
]

@pytest.mark.parametrize('mode,perturb', product(ALL_MODES, ALL_PERTURBATIONS))
def test_round_trip_preserves_state(browser_session, mode, perturb):
    """Enter mode, perturb, exit, assert state is identical to before-enter."""
    page = browser_session
    snapshot_before = page.evaluate('collectStateSnapshot()')
    enter_mode(page, mode)
    apply_perturbation(page, perturb)
    exit_mode(page, mode)
    snapshot_after = page.evaluate('collectStateSnapshot()')
    diff = snapshot_diff(snapshot_before, snapshot_after,
                        ignore=KNOWN_INTENTIONAL_CHANGES.get((mode, perturb), set()))
    assert not diff, f"round-trip {mode} × {perturb} corrupted: {diff}"
```

This is roughly 18 modes × 12 perturbations = 216 tests, all driven through your existing browser test harness. It will probably fail on day one and tell you exactly which fields aren't being saved. **That's the point** — write it first as a diagnostic, fix Pillar A as the failures get triaged.

`KNOWN_INTENTIONAL_CHANGES` is your escape hatch for legitimate exceptions — every entry must be documented with a comment explaining *why*. This forces you to make exceptions explicit instead of accidental.

#### D2 — Cross-mode command-reachability matrix (~30 lines)

After Pillar C exists:

```python
@pytest.mark.parametrize('mode,cmd', product(ALL_MODES, ESSENTIAL_COMMANDS))
def test_essential_command_reachable_in_mode(browser_session, mode, cmd):
    page = browser_session
    enter_mode(page, mode)
    enabled = page.evaluate(f'commands["{cmd}"].when_eval(makeContext(state))')
    expected = INTENTIONAL_DISABLES.get((mode, cmd), True)
    assert enabled == expected, \
        f"{cmd} expected {expected} in mode {mode}, got {enabled}"
```

Same idea: the matrix lives in the test, exceptions are documented, drift is mechanical to detect.

#### D3 — Hypothesis stateful tests (~100 lines)

After Pillars A/B/C are largely done. Use Hypothesis `RuleBasedStateMachine` to randomly fire keybinds (via the command registry) and assert invariants after every step:

```python
class ViewerStateMachine(RuleBasedStateMachine):
    @rule(key=st.sampled_from(ALL_KEYBINDS))
    def press(self, key):
        result = dispatch_key(self.page, key)
        # if no command matched, that's fine; if one matched, run it

    @rule(mode=st.sampled_from(ALL_MODES))
    def goto_mode(self, mode):
        enter_mode(self.page, mode)

    @invariant()
    def renders_without_error(self):
        # render must complete without throwing
        assert self.page.evaluate('lastRenderError') is None

    @invariant()
    def reconcile_is_idempotent(self):
        before = self.page.evaluate('snapshotDom()')
        self.page.evaluate('_reconcileUI(); _reconcileUI();')
        after = self.page.evaluate('snapshotDom()')
        assert before == after  # second call must be a no-op

    @invariant()
    def every_mode_flag_has_a_visible_indicator(self):
        # if compareActive, an .compare-egg must exist; etc.
        ...
```

Hypothesis will generate random sequences and shrink any failure to a minimal counterexample. This is what catches the "I never thought to try this combination" bugs.

#### D4 — Visual snapshots: small, intentional, ~20 max

Keep your existing `tests/ui_audit.py` rules R1–R35 — they are doing the right thing. Do **not** expand them into a combinatorial matrix; that direction leads to flaky-test hell. The matrix is in D1/D2/D3; pixel snapshots are for "this mode looks right" not "this mode behaves right."

**Why this testing layering matters:** the round-trip test (D1) is your *safety net for the refactor*. You write it first, against the *current* code, and it fails. Each fix to Pillar A makes more tests pass. By the time you start Pillar B (the big scale-function refactor), you have a green test suite that tells you instantly if a regression happened.

---

## Part 4 — Phased rollout

Each phase ships independently and is independently valuable. Each phase ends with a green test suite. **Do not start a phase before the previous one is green.**

> **Note on time estimates:** I'm not giving day/week estimates. The user-rules are clear about not predicting durations, and the right unit here is "phase complete + tests green," not calendar time.

### Phase 0 — Baseline diagnostic (no production-code changes)

**Goal:** know what you're up against before changing anything.

- Write `tests/test_mode_roundtrip.py` (Recipe D1) against the *current* code. Expect many failures.
- Run it. Triage the failures into a failure inventory in `dev/plans/ui-maturity-roundtrip-failures.md`. Each row: mode, perturbation, what got corrupted, hypothesis for which class (A/B/C/D) it falls into.
- Add a `dev/plans/ui-maturity-progress.md` scoreboard tracking: round-trip pass rate, snapshot field count vs total field count, command-registry migration progress, scale-function consolidation progress.

**Exit criteria:** the round-trip test runs end-to-end and produces a triage table. No production code touched. No regression risk.

**Why this first:** you cannot run the rest of the strategy without a safety net, and the failure inventory tells you which sub-tasks of Phase 1 to prioritize.

### Phase 1 — Pillar A: finish the snapshot, finish the reconciler

**Goal:** every mode round-trips. Every mutation calls the reconciler.

- For each field in the failure inventory: add it to `collectStateSnapshot` + `applyStateSnapshot`, with validation. Re-run round-trip tests. Watch them turn green.
- Audit every `enterX` / `exitX` for the standard pattern: enter saves snapshot to `_savedX`, exit calls `applyStateSnapshot(_savedX)` then `_reconcileUI()`. Fix the asymmetric ones.
- Add the missing reconciler conditions: `mipActive` in `_reconcileCbVisibility`, `projectionMode` in `_reconcileLayout`, `qmriMosaicActive` wherever it's relevant.
- Add `_cancelAllPendingAnimations()` helper, call from every mode exit.
- Establish the rule `_validateUIState` already enforces, but extend to: "after every state mutation, `_reconcileUI` must have been called." Document in CLAUDE.md / a new dev rule file.

**Exit criteria:** round-trip test pass rate goes from baseline to ≥95% (with documented exceptions for the remaining ~5%). The four "your example" complaints about MIP/colorbar/immersive/state-loss should all be fixed as a side effect. No new architecture introduced.

**Risk:** low. Pure additive changes. The main pitfall is over-broad snapshot fields that include transient state and break rendering on restore — `applyStateSnapshot` validation prevents this.

### Phase 2 — Pillar C: command registry + `when` clauses

**Goal:** the 1309-line keydown switch shrinks to a 50-line dispatcher. All keybinds are in a table with mode contexts.

- Create `_viewer.html` `commands` object and `keybinds` table. Start with three commands as proof-of-concept (e.g., `colormap.cycleNext`, `mode.toggleZen`, `compare.toggleDiff`).
- Create `makeContext(state)` that produces the boolean bag.
- Create `evalWhen(when, ctx)` — start with array-of-required-keys form, upgrade later if needed.
- Create the dispatcher. Wire it as a *prefix* to the existing switch — if a command matches, dispatcher handles it; otherwise fall through.
- Migrate commands one at a time. After each: regenerate the help overlay from the registry, run the round-trip + reachability tests, commit.
- When the switch is empty, delete it.
- Add the command palette UI (`/` opens a fuzzy-finder over the registry). This is small and high-value.
- Add Recipe D2 (reachability matrix test).

**Exit criteria:** zero `case 'X':` clauses left in the keydown handler. Help overlay is auto-generated. Command palette works. Reachability test green for all essential commands. The "feature silently ignored in mode X" class is structurally eliminated.

**Risk:** low–medium. Per-command. Each commit is reversible. You can pause indefinitely between commands.

### Phase 3 — Pillar B step 1: unify colorbars under `ColorBar` class

**Goal:** every colorbar in every mode is a `ColorBar` instance. Inline canvas loops are gone.

- Audit `ColorBar` for missing capabilities (per-pane in compare, per-map in qMRI, multi-view 3-pane). Add the missing capabilities.
- Migrate `drawMvCbs` to use `ColorBar`. Test multi-view round-trip. Commit.
- Migrate `qvRender` colorbar code to use `ColorBar`. Test qMRI round-trip. Commit.
- Migrate `drawComparePaneCb`, `drawDiffPaneCb`, `drawRegBlendCb` to use `ColorBar` (some may already partially do this). Commit.
- Delete the inline gradient-rendering code blocks (the 6 duplications).

**Exit criteria:** grep for `createLinearGradient` returns 1 site (inside `ColorBar.draw`) instead of 6. The "I changed the colorbar and it didn't take in compare" bug is structurally impossible. Round-trip tests stay green.

**Risk:** medium. Visual regression possible — rely on `ui_audit.py` rules R15, R16, R19, R21, R26, R29, R33 (which all guard colorbar layout invariants) to catch them.

### Phase 4 — Pillar B step 2: unify scale functions

**Goal:** one parameterized scale function. The `1800 SLOC of duplication` shrinks to ~400.

This is the largest single piece of work. Do it last, behind a green test suite, and incrementally:

- Extract `_scalePane(pane, viewport, opts)` from `scaleCanvas`. Use it inside `scaleCanvas` first (no behavior change). Commit.
- Use `_scalePane` inside `mvScaleAllCanvases`. Test multi-view. Commit.
- Use `_scalePane` inside `compareScaleCanvases`. Test compare. Commit.
- Same for the other three. Each commit shrinks the codebase.
- Once `_scalePane` is the universal primitive, the wrappers become layout strategies (`layoutMv`, `layoutCompare`, `layoutQmri`, `layoutCompareMv`, `layoutCompareQmri`) — much smaller, mostly geometry. Maybe 100 lines each, vs 200–300 currently.
- Optional: introduce `state.arrays` list and `state.activeIdx`, replacing the scattered `sid`/`compareSidList` model. This is the big napari-style refactor; only do it if Phase 3 has been stable for a while and the round-trip test catches everything.

**Exit criteria:** scale-function SLOC drops by ~70%. Round-trip + visual tests green. New scale bug fixes propagate to all modes for free.

**Risk:** medium–high without the test net, low–medium with it. This is the phase where the test investment from Phase 0 pays off most.

### Phase 5 — Pillar D step 3: Hypothesis stateful tests + Photoshop variant

**Goal:** the test suite drives random sequences and proves global invariants. Per-mode UI options live on the mode object, not in global state.

- Write Hypothesis `RuleBasedStateMachine` per Recipe D3. Drive it through the command registry.
- Define ~5 invariants: render-without-throw, reconcile-is-idempotent, every-mode-flag-has-an-indicator, snapshot-roundtrip, no-orphan-fs-overlays.
- Triage what Hypothesis finds. These will be the weird edge cases.
- (Optional) Photoshop variant: instead of one global snapshot per mode, store per-mode UI options on the mode object. Compare wipe position lives on the compare-mode struct; entering compare a second time picks it up automatically.

**Exit criteria:** a Hypothesis run of 1000 sequences finds zero invariant violations. Per-mode options persist across enter/exit naturally.

**Risk:** low. Pure test additions. The risk is Hypothesis finds 30 bugs that existed all along — that's a feature.

### What's explicitly out of scope

These are the things to **not** do, even though they sound appealing:

- **Adopt React / Lit / Solid / any framework.** The codebase is vanilla JS for a reason (single-file deployability, no build step). The reducer + render pattern works fine without a framework — see the external research, Topic 3 pattern 1.
- **Adopt XState or any statechart library.** Per the external research: foreign mental model, second-system trap, and arrayview's modes are peers not a hierarchy. A flat enum + reducer covers it.
- **Split `_viewer.html` into modules with a build step.** This would multiply the risk of every refactor by adding a build pipeline. Defer until the codebase is stable. (You're already deferring this in `dev/TODO.md` for `_server.py`; same logic.)
- **Rewrite the canvas/WebGL renderer.** The renderer isn't the bug. The state layout is.
- **Hierarchical statechart for compare > diff > wipe.** They're peers under "compare-family," not a hierarchy. The existing `compareCenterMode` enum is correct.
- **Split `_server.py` into domain modules** (the deferred item #5 in `dev/TODO.md`). It's not contributing to the bug class you're describing.
- **One mega-snapshot test on every commit.** Pixel diffs in zoomed/panned states are inherently noisy (per `dev/lessons_learned.md`). Keep snapshots curated.

---

## Part 5 — How to brief Claude (or any agent) on UI work going forward

You said you wonder if you've "been giving the right prompts." Here's the honest answer: the prompts have been fine, but the *codebase* makes it impossible for any agent (or human) to give a small change without missing modes — because the modes are scattered. Once Pillar A is done, the agent guidance also gets simpler. Until then, here's a prompt template that forces the right invariants.

### Template: "I want to change X"

```
I want to change <thing> in arrayview. Before any code:

1. Identify all modes affected. Use the modes-consistency skill, walk
   through dev/mode_matrix.md, and list every primary mode + orthogonal
   feature combination this could touch. If you're unsure, list it.

2. For each affected mode, prescribe the behavior. Note where it's the
   same as normal and where it differs and WHY.

3. Identify the state fields you'll touch. For each: is it in
   collectStateSnapshot()? If not, this change must add it (or document
   why it shouldn't survive a round-trip).

4. Identify the keybinds you'll add or change. For each: which when-clause
   should it have? Which modes should it be reachable from?

5. Plan the test additions. Round-trip test for any state change. Reachability
   test for any new keybind. Visual smoke for any layout change.

Only after I approve the plan, write the code.
```

### Template: "There's a bug where mode X does Y wrong"

```
Bug: in mode X, action Y produces Z instead of W.

Before any fix:

1. Use systematic-debugging skill. Read the relevant code paths first.
2. Classify the bug: state save/restore (Class 1)? code-path duplication
   (Class 2)? keybind dispatch (Class 3)? reconciler gap (Class 4)? other?
3. Find the equivalent code path in the OTHER modes. Check if they have
   the same bug or if one is right and the other is wrong.
4. Propose a fix that addresses the class, not just the symptom. If the fix
   is "add mvDims to collectStateSnapshot," that's better than "in
   exitMultiView, also restore mvDims."
5. Add a regression test in tests/test_mode_roundtrip.py that would have
   caught this.
```

### Template: "I'm starting a UI refactor"

```
I'm refactoring <area>. Hard rules:

- Round-trip test must be green before I start and after I finish.
- I will write the test for the refactor's expected outcome FIRST.
- I will commit per file or per logical step, not as one giant commit.
- For each commit: I run round-trip + ui_audit + visual_smoke, paste the
  results in the commit body.
- I will not introduce new architecture beyond what's prescribed in
  dev/plans/ui-maturity-strategy.md without an explicit approval round.
- If a refactor touches >5 files at once, I split it.
```

### Things you do NOT need to ask about explicitly

The following should become automatic Claude-side, per the existing skills (and in the post-Pillar-A world they'll be enforced by tests):

- Multi-array consistency (modes-consistency skill catches this when invoked)
- Visual regression (ui_audit + visual_smoke run by default per viewer-ui-checklist skill)
- Help overlay sync (auto-generated from command registry post-Pillar C)
- Round-trip preservation (caught by round-trip tests post-Pillar A)
- Keybind reachability (caught by reachability tests post-Pillar C)

In other words: the strategy moves invariants from *human discipline* to *code-and-tests*. After it lands, your prompts can be terser, not more elaborate.

---

## Part 6 — Open decisions for the user

These are things I deliberately did *not* decide for you. When you wake up, decide them and the implementation can start:

1. **Phase 0 first?** I strongly recommend starting with the round-trip test diagnostic before any production-code change. Confirm yes/no.
2. **Single-mode vs all-modes Phase 1?** Pillar A can be done one mode at a time (low risk, slow) or in a batch (faster, more risk). My recommendation: do MIP and qMRI first (highest impact per your examples), then sweep the rest.
3. **Command palette in Phase 2?** It's not strictly necessary for the bug-class fix, but it's a high-value UX feature with marginal additional cost once the registry exists. My recommendation: yes, ship it.
4. **Pillar B as one phase or two?** I split it into Phase 3 (colorbars) and Phase 4 (scale functions) because Phase 3 is a contained, lower-risk warmup that proves the Phase 4 approach. You could also do them together if you're feeling brave, but I don't recommend it.
5. **Photoshop variant (per-mode state on mode object)?** Optional, defer to Phase 5 or beyond. Not required for the bug class.
6. **Should the strategy live in a worktree?** Per `superpowers:using-git-worktrees`: this is multi-week work that should not block ongoing fixes. A `ui-maturity` worktree would let you merge phases as they ripen without disturbing main. My recommendation: yes, create the worktree at the start of Phase 1.
7. **Who runs Phase 0?** This can be a single subagent task — `tests/test_mode_roundtrip.py` is small and self-contained. Alternatively, a subagent-driven plan per `superpowers:subagent-driven-development` can run the whole thing with a checkpoint between phases.

---

## Part 7 — Success criteria (how you'll know it worked)

Concrete, measurable, observable from the outside:

| Metric | Today (estimated) | Target after strategy |
|---|---|---|
| Round-trip test pass rate (modes × perturbations) | ~50–60% | ≥98% |
| Snapshot field count vs total state field count | ~25 / ~180 | All non-transient fields |
| Lines in keydown handler | 1309 | ≤100 (dispatcher only) |
| Distinct scale functions | 5 + 2 wrappers | 1 + N small layout strategies |
| Distinct colorbar drawers | 4 + inline | 1 (`ColorBar` class) |
| `_reconcileUI` call sites needing manual reasoning | "every mode exit, hopefully" | Enforced by `_validateUIState` |
| Help overlay sync | Manual, drifts | Auto-generated from registry |
| New UI bugs per month from this class | (high — driving this whole effort) | ≤1 |
| Time from "user reports bug" to "I know which class it's in" | ~30–60 min grep | <5 min via failing test |

A weaker but more important success criterion: **the next time you change something visual, you should not have to ask "did I break compare mode?" because the test suite answers it.**

---

## Appendix A — File map

Files this strategy will touch, in order of phase:

| Phase | New files | Modified files |
|---|---|---|
| 0 | `tests/test_mode_roundtrip.py`, `dev/plans/ui-maturity-roundtrip-failures.md`, `dev/plans/ui-maturity-progress.md` | none |
| 1 | (none) | `_viewer.html` (collectStateSnapshot, applyStateSnapshot, every enter/exit, _reconcileCbVisibility, _validateUIState) |
| 2 | `_viewer.html` (new commands.js section), help overlay generator, command palette dialog. `tests/test_command_reachability.py` | `_viewer.html` (delete keydown switch incrementally) |
| 3 | (none) | `_viewer.html` (ColorBar class extension, drawMvCbs, qvRender colorbar, drawComparePaneCb, drawDiffPaneCb, drawRegBlendCb) |
| 4 | (none) | `_viewer.html` (scale functions consolidation), possibly `_server.py` (RenderRequest helper) |
| 5 | `tests/test_mode_hypothesis.py` | `_viewer.html` (per-mode option storage, optional) |

## Appendix B — Source reports

- `dev/plans/ui-maturity-codebase-report.md` — full codebase analysis with file:line refs, mode transition map, ranked root causes
- `dev/plans/ui-maturity-external-research.md` — full external research with napari/Slicer/ImageJ2/tldraw/VS Code/Blender deep-dives, 9 patterns, 4 test recipes, 8 anti-patterns, recommendation matrix
- `dev/lessons_learned.md` — prior wisdom; the "UI Reconciler Pattern" section is the seed of Pillar A
- `dev/mode_matrix.md` — authoritative mode/feature reference
- `.claude/skills/ui-consistency-audit/SKILL.md` — current consistency checking discipline (will be partially auto-enforced post-strategy)
- `.claude/skills/modes-consistency/SKILL.md` — current per-mode discipline (same)

## Appendix C — Why this strategy and not another

A few alternatives I considered and rejected. Brief because the external research goes deep on each:

| Alternative | Why rejected |
|---|---|
| Install React or Lit, port the viewer to declarative components | Build pipeline cost, deployment cost, single-file breakage, no payoff over a 200-line vanilla reducer |
| Install XState, model every mode as a state node | Foreign mental model, second-system trap, modes are peers not hierarchy. External research explicitly flags this for projects this size |
| Rewrite the canvas/WebGL renderer to be more flexible | The renderer isn't the bug; state layout is. Rewriting without fixing state would just create new bugs |
| Add type checking via TypeScript | High cost, doesn't address the runtime mode-mismatch class (the bugs aren't type errors, they're semantic errors) |
| Split `_viewer.html` into ES modules | Doesn't fix the bug class; just moves code around. Defer until Pillar A/C are done and the seams are clearer |
| One giant Playwright snapshot matrix | External research and your own `lessons_learned.md` agree this becomes a flaky-test hellscape. The matrix belongs in model-level tests, not pixel diffs |
| Just write more `ui_audit.py` rules | Diminishing returns. The rules catch *visibility* bugs but not *state-loss* or *reachability* bugs. The strategy attacks the latter directly |

---

*End of strategy. This document is intentionally long for a single read; future references to it should be by phase number ("we're doing Phase 1") rather than by re-reading the whole thing.*
