# External Research: Taming Mode/Feature Combinatorics

Research target: a ~15 kloc vanilla-JS / Python scientific image viewer (`arrayview`) suffering the classic combinatorial-explosion bugs: features × viewing modes × keybinds × array-counts. This report is opinionated. Sources are linked inline; a consolidated list is at the end.

---

## Executive summary

After surveying napari, 3D Slicer, ImageJ2, tldraw, Excalidraw, VS Code, and Blender, plus the state-management / testing literature, the maintainer should focus on **four** things and explicitly **ignore** the rest:

1. **Unify the N=1 and N≥2 code paths by modeling state as "always a list of layers."** This is what napari actually does: there is no "single-layer" special case because the viewer's only public surface is `LayerList`, and every operation iterates it. This is the single highest-leverage change. It is a *data-model* fix, not a framework rewrite. (See Topic 4.)

2. **Adopt a command registry with `when`-clause contexts, VS Code-style.** Every user action becomes `{id, run(), enabled(ctx)}`. Keybinds resolve through the registry. This gives you: auto-generated help, a command palette for free, automated "is feature F reachable from mode M?" tests, and the ability to answer "why didn't that key work?" in one grep. This is the single highest-leverage change for the keybind-silently-ignored class of bug. (See Topics 2 and 3, pattern 4/5.)

3. **Add a tiny reducer + explicit `render(state)` in vanilla JS.** You already have a partial reconciler. Finish it. One mutable `state` object, one `dispatch(action)`, one `render()` called after every dispatch. No React, no Redux, no XState — ~150 lines. This makes "entering then exiting a mode doesn't restore prior state" mechanically impossible for anything held in the store. (See Topic 3, pattern 1+2.)

4. **Write Hypothesis `RuleBasedStateMachine` tests that randomly fire keybinds and assert invariants** (e.g., "every enter→exit returns to the start state", "every command in the registry is enableable in at least one reachable context"). Driving a *model* rather than the real DOM is cheap and finds the round-trip bugs the maintainer is describing. Supplement with a *small* number of Playwright screenshot tests for the visually load-bearing states only. (See Topic 5.)

**Things to explicitly NOT do** (Topic 6): do not adopt XState or a formal statechart library; do not attempt a "rewrite the renderer" as a first step; do not try to model nested modes (compare > diff > wipe) as a hierarchical statechart before the flat command/context approach is in place. Every one of those is a known second-system trap for a codebase this size.

---

## Topic 1 — Scientific viewers

### napari (most directly relevant prior art)

- **State layout.** `ViewerModel` is a pydantic-style model that owns a `LayerList` (a frozen field) plus `Dims`, `Camera`, `Cursor`, `GridCanvas`. Operations on "the image" don't exist — operations go through `viewer.layers`, which is always a list, and the active layer is just `viewer.layers.selection.active`. The single-layer case is literally `len(layers) == 1`; there is no separate code path. See `napari/components/viewer_model.py` on the napari repo. ([viewer_model source](https://github.com/napari/napari/blob/main/src/napari/components/viewer_model.py))
- **Events.** Every attribute on every model is wrapped in a psygnal `EventedModel`. Mode changes emit `layer.events.mode`. The Qt layer listens and re-renders; nothing in the core model knows Qt exists. ([events reference](https://napari.org/stable/guides/events_reference.html))
- **Per-layer mode.** Each layer class (`Labels`, `Points`, `Shapes`) has its own `Mode` enum (`PAN_ZOOM`, `PAINT`, `FILL`, `PICK`, `TRANSFORM`, …). The base `Layer.mode` setter in `napari/layers/base/base.py` does a uniform dance for every layer: validate the enum, force `PAN_ZOOM` if the layer isn't editable/visible, swap in the right callback lists from `_drag_modes`, `_move_modes`, `_double_click_modes`, update cursor, update help text. **The mode setter is the single place mode transitions happen.** That's the "one place to break" invariant worth stealing.
- **Keybinds.** `Layer` inherits from `KeymapProvider`. Keybindings are registered on the *class*, resolved against the active layer. Same key (e.g. `M` for pick-label) works in different layer contexts because the lookup is class-scoped; keys that aren't defined on the active layer's class simply fall through.
- **1 vs N.** There isn't one. Everything in napari is a LayerList operation. Adding a channel is `layers.extend(...)`; adding one image is `layers.append(...)`. This is the direct answer to arrayview's Topic 4 pain.
- **Tests.** napari's test suite heavily parametrizes over layer types and modes; the pattern is `@pytest.mark.parametrize('mode', list(Mode))` + assert that `layer.mode = m; layer.mode = 'pan_zoom'` leaves no dangling callbacks.

### 3D Slicer (MRML)

- **Everything is a node in a scene graph.** `vtkMRMLScene` owns `vtkMRMLNode` instances — volumes, segmentations, views, *and* two singleton nodes `vtkMRMLSelectionNode` and `vtkMRMLInteractionNode` that hold global "what's selected" and "what's the current interaction mode (view/place/transform)" state. ([MRML overview](https://slicer.readthedocs.io/en/latest/developer_guide/mrml_overview.html))
- **Mode = a value on a singleton node.** The interaction mode is not a property of a module or the viewer — it's a value on the singleton InteractionNode. Modules *observe* it. This is a clean way to ensure "every mode-aware thing sees the same truth."
- **Modules guard themselves.** Each module's logic subscribes to the scene and decides whether it applies. This is the MVC shape arrayview should *not* literally adopt (it's heavy C++/Qt), but the idea — "current mode lives in exactly one place, everything reads it" — is the key takeaway.

### ImageJ2 / SciJava

- **Command / Op / Tool plugin trinity.** Every user-visible action is a `Command` (a SciJava plugin). A `Tool` is "a collection of rules binding user input to actions." Tools are *also* plugins. This means keybinds, menu items, and scripts all dispatch through the same registry. ([ImageJ2 paper](https://link.springer.com/article/10.1186/s12859-017-1934-z), [architecture](https://imagej.net/develop/architecture))
- **Headless-capable because GUI is decoupled.** Commands declare their inputs/outputs as annotated fields; the UI is generated. Same command runs from menu, keybind, script, or macro.
- **Takeaway for arrayview:** make every user action a command object with a name, a `run`, and an `enabledWhen`. Don't build a plugin system — just the registry.

### ITK-SNAP / MITK

- Less directly useful for a 15 kloc viewer. MITK uses a DataStorage + rendering-manager pattern that's essentially "MRML with a different name." ITK-SNAP uses a global `IRISApplication` + per-tool classes with `OnMouseDown/OnKey` virtuals — the tldraw StateNode pattern, in C++. Not enough novel content to justify a deep dive.

---

## Topic 2 — General UI apps

### VS Code — the gold standard for this exact problem

- **ContextKeyService.** A key-value store of boolean/string contexts: `editorTextFocus`, `inDebugMode`, `resourceExtname`, `activeViewlet`, etc. Any component can `setContext('foo', true)`. ([when-clause docs](https://code.visualstudio.com/api/references/when-clause-contexts))
- **Command registry.** Every action is `{id, handler, title?}`. Commands can declare an `enablement` when-clause. The palette, keybindings, and menus all look up commands from the same registry.
- **Keybindings.** Each keybind is `{key, command, when}`. When a key is pressed, VS Code looks up all matching bindings, filters by `when` evaluated against the current ContextKey snapshot, and dispatches the first match. **This is the thing that makes "the same command works in editor, notebook, and diff" work:** the contexts are set by the view that has focus, so `editor.action.formatDocument` is enabled wherever there's a formattable editor, regardless of which view type hosts it.
- **Why this matters for arrayview.** Your "keybind silently ignored in modes where it should work" bug goes away if you have: (a) a single registry, (b) explicit `when` clauses, (c) a single key-dispatch function. Then "is this key supposed to work here?" is `registry.resolve(key, ctx)` — printable, testable, greppable.

### Excalidraw and tldraw

- **Excalidraw** uses a big `AppState` object with booleans/enums (`viewModeEnabled`, `zenModeEnabled`, `activeTool`, `bindMode`). Transitions are `setState` calls scattered across handlers. They are on record about this becoming a source of bugs, and the community periodically discusses moving to XState — but hasn't. Takeaway: "big mutable AppState" scales further than you think if you have a single render function, but distributes state-transition logic and makes bugs harder to localize.
- **tldraw** is the opposite: a true statechart. Tools are `StateNode`s in a hierarchy; input events are routed to the currently-active leaf state; each StateNode has `onEnter`, `onExit`, `onPointerDown`, etc. Child states handle sub-interactions (idle → pointing → dragging). ([tldraw tools docs](https://tldraw.dev/docs/tools), [child-states example](https://tldraw.dev/examples/shapes/tools/tool-with-child-states)) This is the cleanest real-world statechart in a shipping product; the cost is that new contributors have to learn the StateNode mental model.

### Blender

- **Operators with `poll()`.** Every user action is a `bpy.types.Operator` with a `poll(cls, context)` classmethod that returns True iff the operator can run in the current context. The operator's `execute(context)` is only called if `poll` passes. ([Blender docs](https://docs.blender.org/api/current/bpy.ops.html))
- **This is the VS Code when-clause pattern, Python-flavored.** Keybinds, menus, and scripting all go through the same `bpy.ops.*` registry; `poll()` is the universal gate.

### Photoshop / Krita

- Tool options are stored *per tool*, not on the canvas. Switching tools and switching back restores the prior option state automatically. The equivalent for arrayview: when you exit "compare" mode, the compare-specific settings (wipe position, diff threshold) should be stored *on the compare-mode object*, not on the global viewer state, so entering compare later picks them back up for free.

---

## Topic 3 — Patterns and tradeoffs

### 1. Single source of truth / unidirectional data flow

**What.** One `state` object, one `dispatch(action)`, a reducer function that returns the next state, subscribers notified. No partial updates — the whole state gets replaced (or explicitly patched in one place). **Vanilla-JS recipe:**
```js
let state = initialState;
const listeners = new Set();
function dispatch(action) {
  state = reduce(state, action);
  for (const l of listeners) l(state);
}
```
That's it. ~20 lines. ([vanilla redux example](https://ramonvictor.github.io/tic-tac-toe-js/))

**Who does it well.** Redux/Elm in React land; napari's EventedModel; Slicer's MRML scene (subscribers observe a scene).

**When it's the wrong choice.** When state has huge non-serializable blobs (canvas contexts, WebGL objects) and you end up storing references *outside* the store anyway, defeating the point. Mitigation: keep GPU/DOM handles out of the store, store only the parameters that produced them, let render() rebuild or cache.

**Fit for arrayview.** Strong. You already have a partial reconciler; this is the last mile. The thing it will fix: "exiting a mode doesn't restore prior state." If it isn't in the store, the round-trip is unrepresentable.

### 2. Reconciler / declarative render

**What.** `render(state) → DOM`, called after every `dispatch`. No imperative "when the user clicks X, also update Y and Z." ([Restate Your UI](https://www.cognitect.com/blog/2017/5/22/restate-your-ui-using-state-machines-to-simplify-user-interface-development))

**Who does it well.** React, Lit, Solid. Also napari's Qt layer, which re-reads the model on every event.

**When it's the wrong choice.** Perf-critical hot paths (per-pixel canvas updates). You solve this by splitting into a *shell* render (cheap, runs on every dispatch) and a *canvas* render (debounced, only runs when the parameters it depends on actually change). This is how napari's vispy canvas works.

**Fit for arrayview.** Strong. The HTML shell (toolbars, overlays, help text, colorbar labels) should be fully declarative. The WebGL/canvas layer should read from the same store but memoize on the relevant slice.

### 3. Finite state machine / statecharts

**What.** States, events, transitions, guards, entry/exit actions, optionally hierarchy (compare > diff > wipe) and parallelism. Harel statecharts (1987) are the canonical formalism; XState is the popular JS impl. ([statecharts.dev](https://statecharts.dev/), [xstate](https://xstate.js.org/))

**Who does it well.** tldraw (native StateNode), embedded systems, Figma's interaction model.

**When it's the wrong choice.** This is the pattern most likely to seduce you and most likely to hurt. Shevlin's guidelines note that guards often indicate a mismodeled hierarchy, that setup cost is real, and that small systems are often clearer with a switch+enum. ([Shevlin](https://kyleshevlin.com/guidelines-for-state-machines-and-xstate/)) The statecharts.dev site itself lists "usually a very foreign way of coding" and "code expansion with smaller implementations" as drawbacks. For a 15 kloc viewer with a handful of modes, a flat `mode: 'normal' | 'compare' | 'diff' | 'wipe'` plus per-mode sub-state inside the reducer will get you 90% of the way there with 5% of the ceremony.

**Fit for arrayview.** Weak as a *library adoption*. Strong as a *discipline*: draw the state diagram on paper, identify which transitions are legal, encode them as assertions in the reducer. Do not install XState.

### 4. Command pattern + command registry

**What.** Every user action is `{id, title, run(ctx), enablement}`. Keybinds, menus, palette, help, and tests all resolve through the registry. ([VS Code commands](https://code.visualstudio.com/api/extension-guides/command), [ImageJ2](https://link.springer.com/article/10.1186/s12859-017-1934-z))

**Who does it well.** VS Code, Sublime, Emacs, ImageJ2, Blender (operators).

**When it's the wrong choice.** Basically never, once you've got more than ~20 actions. The upfront cost is one file. The payoff is enormous.

**Fit for arrayview.** **This is the single biggest win.** It directly addresses "keybinds silently ignored" and "feature in one mode bypasses another mode." Write it once, migrate the existing keybind table into it, and every future feature slots in as a command.

### 5. Capability-based contexts / when-clauses

**What.** A flat context bag (`{ compareMode: true, hasMultipleArrays: true, helpOpen: false, ... }`) plus tiny boolean-expression evaluator. Each command/keybind declares `enablementWhen: 'compareMode && !helpOpen'`. Auto-generating the (keybind × context) matrix is literally `for cmd in registry: print(cmd.id, cmd.when)`. ([VS Code when clauses](https://code.visualstudio.com/api/references/when-clause-contexts))

**Fit for arrayview.** Direct, essential complement to pattern 4. Don't build a DSL parser — a tagged-template or array-of-required-keys is enough.

### 6. State snapshot / mode stack with push/pop

**What.** On mode enter, push a snapshot of the state slice you're about to clobber; on exit, pop. Used in vim, browser history, modal dialogs.

**When it's the wrong choice.** When "what to restore on exit" depends on what happened *during* the mode (e.g., entering compare, loading a new array, exiting — do you restore the old array or keep the new one?). Stacks encode the wrong intent.

**Fit for arrayview.** Useful for *modal* interactions (help overlay, command palette, temporary tool). Wrong for *viewing modes* like compare/diff — those should be first-class store fields with a reducer that owns the transitions. Don't use the same mechanism for both.

### 7. Property-based / model-based testing

**What.** Hypothesis `RuleBasedStateMachine` (Python) or fast-check (JS). You declare rules ("press key X", "enter compare mode", "load array"), and Hypothesis generates random sequences, checking `@invariant`s after every step. ([Hypothesis stateful](https://hypothesis.readthedocs.io/en/latest/stateful.html))

**Fit for arrayview.** Excellent, *if you drive the Python model or a headless JS model* rather than the real DOM. See Topic 5 for a concrete sketch.

### 8. Visual regression testing

**What.** Percy/Chromatic/Playwright `toHaveScreenshot`. Baseline image + diff per commit. ([Playwright snapshot guide](https://www.browserstack.com/guide/playwright-snapshot-testing), [fixing flaky playwright visual tests](https://www.houseful.blog/posts/2023/fix-flaky-playwright-visual-regression-tests/))

**When it's the wrong choice.** If every diff is noise, you have no tests. The field's practical advice: mask timestamps, disable animations/fonts with stable loading, target elements not full pages, and keep the number of snapshots small and intentional.

**Fit for arrayview.** Small, curated set (~10–20 snapshots: one per mode × representative array count). Not a 500-snapshot matrix. The matrix belongs in model-based tests (pattern 7).

### 9. Interaction / story testing

**What.** Storybook play functions or Playwright scenarios: "given mode M, press key K, assert state S." Good unit for "feature F works in mode M" assertions.

**Fit for arrayview.** Medium. Probably better to do this at the *model* level (Python side + JS store) than with a browser driver, until you have a stable store.

---

## Topic 4 — 1-vs-N code-path unification

The maintainer's complaint: "1 array goes through one path, 2 arrays goes through another, and they drift." This is the single most common architectural mistake in scientific viewers, and it has a known fix.

### napari's answer: there is no N=1 case

napari's `ViewerModel` holds a `LayerList`, full stop. The public API is `viewer.layers.append(...)`, `viewer.layers[i]`, `viewer.layers.selection`. You cannot ask the viewer "what's the image?" — only "give me your layers." Operations like "set contrast limits" go through `layer.contrast_limits` on the *active* layer, or are broadcast to the selection. The rendering pipeline composes layers bottom-up with per-layer blending; the N=1 case is a one-iteration loop. ([LayerList API](https://napari.org/stable/api/napari.components.LayerList.html), [layers guide](https://napari.org/stable/guides/layers.html))

**What this buys napari:**
- No "if single image, do A; if multiple, do B" branches. The branch would live in the rendering pipeline, and napari's pipeline is always a loop.
- Adding/removing the second image triggers the same `layers.events.inserted` event as the first — there is no "transition to multi-image mode" code path that can drift.
- Any feature that works for one layer works for N, because it's implemented as `for layer in layers: ...` or as `layers.selection.active.something`.

### 3D Slicer's answer: same shape, different name

Slicer's scene graph treats "one volume" as a scene with one volume node. The `Red/Green/Yellow` slice viewers always render `scene.GetNodesByClass('vtkMRMLVolumeNode')`. One volume is just `len == 1`.

### The "scalar as length-1 list" pattern

**When it succeeds.** When the list semantics are coherent for N=1: "iterate and render", "iterate and compute", "active selection is one element". The cost of the list wrapper is zero; the payoff is no branches.

**When it fails.** When the operation is *genuinely* binary — e.g., "diff mode" requires exactly 2 arrays. The list pattern doesn't hurt here, it just means you write `if len(arrays) != 2: disable diff command via when-clause`. The `when`-clause approach (pattern 5) is how you gate the genuinely-binary case without re-introducing a branch in the render path.

**Concrete recommendation for arrayview.**
1. Introduce a single `arrays` list in the store. Replace all `array` / `array2` / `is_compare_mode` scalars with `arrays: Array[]` and `compareMode: null | {...}`.
2. The renderer becomes "iterate `arrays`, apply the compare overlay if `compareMode` is set." The N=1 case becomes `arrays.length === 1` with no special code path.
3. Commands that require N≥2 (diff, wipe, side-by-side) declare `enablementWhen: 'arrays.length >= 2'`. That's the only N-gate in the codebase.
4. Keybinds for those commands automatically become no-ops with N=1, and the command palette shows them disabled with the reason — no silent ignoring.

This is a refactor, not a rewrite. You do it layer by layer: first the store shape, then the render, then migrate each feature.

---

## Topic 5 — Cross-mode invariant testing

### Recipe A — Hypothesis RuleBasedStateMachine (Python, drives model)

```python
# sketch — do not run
from hypothesis.stateful import RuleBasedStateMachine, rule, invariant, precondition

class ViewerModel(RuleBasedStateMachine):
    def __init__(self):
        super().__init__()
        self.store = make_initial_store()

    @rule(n=st.integers(1, 4))
    def load_arrays(self, n):
        dispatch(self.store, {'type': 'load', 'arrays': fake_arrays(n)})

    @rule(mode=st.sampled_from(['normal', 'compare', 'diff', 'wipe']))
    def set_mode(self, mode):
        dispatch(self.store, {'type': 'set_mode', 'mode': mode})

    @rule(key=st.sampled_from(ALL_KEYBINDS))
    def press(self, key):
        cmd = resolve_keybind(self.store, key)
        if cmd and cmd.enabled(self.store):
            dispatch(self.store, cmd.action)

    @invariant()
    def every_mode_has_valid_render(self):
        # state must always be renderable without throwing
        render_dry_run(self.store)

    @invariant()
    def exiting_returns_to_normal(self):
        # after set_mode('normal'), no compare/diff/wipe residue
        if self.store['mode'] == 'normal':
            assert self.store.get('compareOverlay') is None

TestViewer = ViewerModel.TestCase
```

Hypothesis will randomly sequence rules and shrink any failure to a minimal counterexample. This is the single highest-leverage test you can write against mode-explosion bugs. ([Hypothesis stateful docs](https://hypothesis.readthedocs.io/en/latest/stateful.html), [rule-based stateful article](https://hypothesis.works/articles/rule-based-stateful-testing/))

### Recipe B — Round-trip matrix (plain pytest)

```python
# sketch
@pytest.mark.parametrize('mode', ALL_MODES)
def test_mode_round_trip_restores_state(mode):
    s0 = make_store_with_two_arrays()
    snapshot = deepcopy(s0)
    dispatch(s0, {'type': 'set_mode', 'mode': mode})
    dispatch(s0, {'type': 'set_mode', 'mode': 'normal'})
    assert s0 == snapshot, f"round-trip through {mode} corrupted state"

@pytest.mark.parametrize('mode', ALL_MODES)
@pytest.mark.parametrize('cmd', ESSENTIAL_COMMANDS)
def test_essential_command_reachable_in_mode(mode, cmd):
    s = make_store_with_two_arrays()
    dispatch(s, {'type': 'set_mode', 'mode': mode})
    assert cmd.enabled(s) or cmd.intentionally_disabled_in.get(mode), \
        f"{cmd.id} unexpectedly disabled in mode {mode}"
```

This is the round-trip strategy from the FSM testing literature — [round-trip paths](https://www.researchgate.net/publication/4000992_A_case_study_using_the_round-trip_strategy_for_state-based_class_testing). The `intentionally_disabled_in` field forces you to *document* every exception.

### Recipe C — napari-style: parametrize over the Mode enum

napari does this pervasively. Example pattern:

```python
@pytest.mark.parametrize('mode', list(Labels.Mode))
def test_mode_has_callbacks(mode):
    layer = Labels(data)
    layer.mode = mode
    assert layer.mouse_drag_callbacks  # setter must wire callbacks
    layer.mode = 'pan_zoom'  # exit round-trips
    assert layer.mouse_drag_callbacks == initial_callbacks
```

### Recipe D — Playwright, but small and targeted

```js
// sketch
for (const mode of ['normal','compare','diff','wipe']) {
  test(`${mode}: help overlay opens and closes`, async ({page}) => {
    await setMode(page, mode);
    const before = await page.screenshot();
    await page.keyboard.press('h');
    await page.keyboard.press('h');
    const after = await page.screenshot();
    expect(after).toEqual(before); // round-trip
  });
}
```

Keep it to ~20 tests. The matrix belongs in Recipe A. ([Playwright visual regression](https://www.browserstack.com/guide/visual-regression-testing-using-playwright))

---

## Topic 6 — Anti-patterns

Do not do any of these:

- **Install XState first.** You will spend a week modeling and discover your state model is wrong once you start wiring render. Draw the diagram on paper, encode it as a plain reducer, *then* decide if you need a library (you won't).
- **Rewrite the renderer as the first step.** The renderer isn't the bug — the bug is that state is scattered. Fix the state layout first; the renderer change falls out for free.
- **Hierarchical statechart for compare > diff > wipe.** These are three peers under "compare-family." Model as `mode: 'normal' | 'compare'` + `compareKind: 'overlay' | 'diff' | 'wipe'`. Hierarchy is the thing that makes statecharts hard to debug. ([Harel pitfalls discussion](https://news.ycombinator.com/item?id=35328995))
- **Refactor without tests.** Write Recipe B (the plain pytest matrix) *before* touching the store. It's ~50 lines and will catch every regression during the refactor.
- **One giant screenshot matrix.** 500 Playwright snapshots = 500 flaky tests = tests muted = no tests. Keep visual regression small and intentional; rely on model-level property tests for the combinatorics. ([flaky playwright postmortem](https://www.houseful.blog/posts/2023/fix-flaky-playwright-visual-regression-tests/))
- **Premature abstraction of "viewer backends."** The moment you generalize over "canvas mode" vs "WebGL mode" before you have a unified store, you're building the second system.
- **Distributing mode-transition logic across handlers.** Excalidraw's AppState is the cautionary tale. A single reducer file is dramatically easier to debug than 40 `setState` call sites.
- **Silent keybind drop.** Never `if (!mode.supports(key)) return;` silently. Either the command is in the registry and its `when` clause is false (palette explains why), or the key is unbound (explicit). No secret third option.

---

## Recommendation matrix

| Pattern | Effort | Payoff | Risk | Fits arrayview today? |
|---|---|---|---|---|
| **1. Always-a-list data model (N=1 is N)** | Medium (refactor store shape + renderer loop) | **Very high** — eliminates the 1-vs-N drift class entirely | Low if done behind a single reducer | **Yes — do first** |
| **4. Command registry + `when` clauses** | Low–Medium (one file + migration of existing keybinds) | **Very high** — eliminates silent keybind drops, enables palette, enables test matrix | Very low — purely additive | **Yes — do first** |
| **2. Reducer + `render(state)` finished** | Medium (finish the partial reconciler) | High — round-trip bugs for store-held state become unrepresentable | Low if you keep GPU handles out of the store | **Yes — do second** |
| **7. Hypothesis stateful tests + round-trip matrix** | Low (drives the model, not the DOM) | High — finds the bugs the maintainer is describing, automatically | Low | **Yes — do alongside** |
| **8. Small curated Playwright snapshots** | Low | Medium — catches visual regressions the model can't see | Medium — flakiness if you over-do it | Yes, **~20 snapshots max** |
| **3. XState / formal statechart library** | High | Medium | **High** — foreign mental model, team cost, second-system risk | **No** |
| **5. Hierarchical statechart for nested modes** | High | Low for your scale | High | **No** |
| **6. Mode stack with push/pop** | Low | Low — solves modal dialogs, not viewing modes | Low | Only for help/palette/transient overlays |
| **"Rewrite the renderer"** | Very high | Unknown | **Very high** | **No** |

---

## Final opinionated ordering

1. Week 1 — Write the round-trip pytest matrix (Recipe B) against the *current* code. This is your safety net.
2. Week 1–2 — Build the command registry with `when` clauses. Migrate keybinds into it. This alone will kill most of the keybind-silently-ignored bugs and give you a printable table of (key × command × context).
3. Week 2–3 — Refactor the store to "always a list" (Topic 4). Introduce the reducer. The renderer loop becomes `for array of state.arrays`.
4. Week 3–4 — Finish the reconciler so the HTML shell is fully derived from state.
5. Week 4+ — Add Hypothesis stateful tests (Recipe A) driving the store. Add ~20 Playwright snapshots for the visual-critical mode states.

Skip XState. Skip the statechart library. Skip the renderer rewrite. The combinatorics problem is a *data-layout* problem, not an algorithmic one.

---

## Consolidated sources

- napari: [viewer_model.py](https://github.com/napari/napari/blob/main/src/napari/components/viewer_model.py), [layers base](https://github.com/napari/napari/blob/main/src/napari/layers/base/base.py), [events reference](https://napari.org/stable/guides/events_reference.html), [layers guide](https://napari.org/stable/guides/layers.html), [LayerList API](https://napari.org/stable/api/napari.components.LayerList.html), [NAP-9 multi-canvas](https://napari.org/stable/naps/9-multiple-canvases.html)
- 3D Slicer: [MRML overview](https://slicer.readthedocs.io/en/latest/developer_guide/mrml_overview.html), [module overview](https://slicer.readthedocs.io/en/latest/developer_guide/module_overview.html)
- ImageJ2: [paper](https://link.springer.com/article/10.1186/s12859-017-1934-z), [architecture](https://imagej.net/develop/architecture), [SciJava](https://imagej.net/libs/scijava)
- VS Code: [when-clause contexts](https://code.visualstudio.com/api/references/when-clause-contexts), [commands guide](https://code.visualstudio.com/api/extension-guides/command)
- tldraw: [tools docs](https://tldraw.dev/docs/tools), [custom tool with child states](https://tldraw.dev/examples/shapes/tools/tool-with-child-states), [editor](https://tldraw.dev/docs/editor)
- Excalidraw: [App.tsx](https://github.com/excalidraw/excalidraw/blob/master/packages/excalidraw/components/App.tsx)
- Blender: [bpy.ops](https://docs.blender.org/api/current/bpy.ops.html), [Operator type](https://docs.blender.org/api/current/bpy.types.Operator.html)
- Statecharts / XState: [statecharts.dev](https://statecharts.dev/), [XState](https://xstate.js.org/), [Kyle Shevlin guidelines](https://kyleshevlin.com/guidelines-for-state-machines-and-xstate/), [Tim Deschryver love letter](https://timdeschryver.dev/blog/my-love-letter-to-xstate-and-statecharts)
- Testing: [Hypothesis stateful](https://hypothesis.readthedocs.io/en/latest/stateful.html), [Hypothesis rule-based article](https://hypothesis.works/articles/rule-based-stateful-testing/), [round-trip FSM testing](https://www.researchgate.net/publication/4000992_A_case_study_using_the_round-trip_strategy_for_state-based_class_testing), [Playwright visual regression (BrowserStack)](https://www.browserstack.com/guide/visual-regression-testing-using-playwright), [fixing flaky Playwright visual tests](https://www.houseful.blog/posts/2023/fix-flaky-playwright-visual-regression-tests/)
- State management essays: [Restate Your UI (Cognitect)](https://www.cognitect.com/blog/2017/5/22/restate-your-ui-using-state-machines-to-simplify-user-interface-development), [vanilla Redux example](https://ramonvictor.github.io/tic-tac-toe-js/)
- Second-system: [Wikipedia](https://en.wikipedia.org/wiki/Second-system_effect)
