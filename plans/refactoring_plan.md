# ArrayView Refactoring Plan

## Phase 1: Dead Code + Deduplication

**Risk: Low | Files touched: ~8**

### 1.1 Remove dead TOML parser fallback

`_config.py` lines 120-142: hand-rolled parser for Python 3.10. The project requires `>=3.12` (`pyproject.toml:10`). Dead code.

- Delete `_parse_toml` lines 126-142 (the `else` branch handling `sys.version_info < (3,11)`)
- `_parse_toml` becomes just: `import tomllib; return tomllib.loads(text)`

### 1.2 Extract shared `_ensure_pil()` to `_imaging.py`

Identical ~10-line lazy import pattern in 5 files:

```python
def _ensure_pil():
    try:
        from PIL import Image
        return Image
    except ImportError:
        ...
```

- Create `src/arrayview/_imaging.py` with `_ensure_pil()` and `_pil_image(data, w, h, mode)` helper (also duplicated)
- Replace in: `_diff.py`, `_server.py`, `_routes_rendering.py`, `_routes_websocket.py`, `_stdio_server.py`
- Tests: `test_backend_shared.py` (PIL usage is checked through rendering paths)

### 1.3 Extract `_evict_lru()` in `_render.py`

While-loop repeated 3 times (raw cache, rgba cache, mosaic cache):

```python
while _raw_bytes > budget:
    oldest = cache.popitem(last=False)
    _raw_bytes -= len(oldest[1])
```

- Extract to: `_evict_lru(cache, bytes_attr, budget, get_bytes_fn)`
- Replace 3 call sites in `_render.py`

### 1.4 Extract mosaic grid helper

Gap=2, row/col division, NaN-mask mosaic grid duplicated in:
- `_render.render_mosaic()`
- `_diff._render_normalized_mosaic()`
- `_routes_rendering.get_grid()`

- Create `_build_mosaic_grid()` in `_render.py` (it's the hub module)
- Call from the other two files

---

## Phase 2: Split `_vscode.py`

**Risk: Medium | New files: 4**

`_vscode.py` is 1,333 lines. Split path:

| Current function(s) | → New file | Lines |
|---|---|---|
| `_ensure_vscode_extension()`, `_get_extension_version()`, port preview config | `_vscode_extension.py` | ~300 |
| `_write_vscode_signal()`, signal-file helpers, PID/tmux targeting | `_vscode_signal.py` | ~500 |
| `_open_direct_via_shm()`, SHM transport | `_vscode_shm.py` | ~120 |
| `_open_browser()`, SSH guidance, `_print_viewer_location()` | `_vscode_browser.py` | ~130 |

`_vscode.py` becomes a facade that re-exports from submodules. Importers (`_launcher`, `_routes_loading`, `_app`) get the same API.

No circular risk — `_vscode` currently imports only `_session` and `_platform`, both leaf modules.

- Tests: extend `test_api.py` VS Code integration tests to exercise each submodule's boundary

---

## Phase 3: Split `_launcher.py`

**Risk: Medium | New files: 4**

`_launcher.py` is 3,570 lines. Split path:

| New module | Contents | Lines | Layer |
|---|---|---|---|
| `_webview.py` | `_get_icon_png_path()`, `_open_webview()`, `_open_webview_with_fallback()`, `_open_webview_cli()` | ~260 | Leaf |
| `_loading_server.py` | `_server_ready_event`, `_LOADING_HTML`, `_LoadingHandler`, `_run_loading_server()`, `_serve_background()` | ~160 | Leaf |
| `_url_builder.py` | `_viewer_query()`, `_viewer_path()`, `_viewer_url()`, `_shell_url()`, `_jupyter_base_url_prefix()`, `_should_use_jupyter_proxy_inline()`, `_make_jupyter_proxy_inline_html()`, `_with_loading()`, `_join_query_values()`, `_OVERLAY_PALETTE`, `_CLI_DAEMON_CONNECT_TIMEOUT_SECONDS`, `_CLI_DAEMON_IDLE_SECONDS` | ~200 | Leaf |
| `_cli_paths.py` | `_resolve_cli_window_mode()`, `_plan_cli_port_strategy()`, `_resolve_view_port()`, `_select_arrayview_launch_path()`, `_normalize_view_window_request()`, `_load_compare_sids()`, `_open_cli_existing_server_view()`, `_register_cli_session_with_existing_server()`, `_handle_cli_existing_server()`, `_handle_cli_spawned_daemon()`, `_open_cli_spawned_view()`, `_parse_dims_spec()` | ~600 | Mid-layer |

`_launcher.py` retains: `_LazyMod`, lazy proxies (`_vscode_mod()`, `_server_mod()`, `_uvicorn()`), `ViewHandle`, `view()`, `arrayview()`, `_handle_config_command()`, `_make_demo_array()`, `_start_watch_thread()`, `_serve_empty()`, `_serve_daemon()`, `_wait_for_viewer_close()`, `_is_script_mode()`, `_stop_server_when_viewer_closes()`, server port utilities, display resolution helpers, Julia/subprocess paths, `_vprint()`.

Result: `_launcher.py` → ~1,900 lines (from 3,570).

No circular risk — `_launcher` already lazy-loads `_session`, `_server`, `_vscode` with `_LazyMod` / inline imports.

- Tests: existing `test_api.py`, `test_cli.py`, `test_loading_server.py` cover these paths

---

## Phase 4: Frontend Selective Cleanup

**Risk: Medium | File: `_viewer.html` only**

### 4.1 Fix 3 stub LayoutStrategies

`CompareMvLayout`, `CompareQmriLayout`, `MipLayout` currently return `[]` from `getViews()`. Fix them to correctly populate `currentViews` by wrapping the existing DOM canvases like the other layouts do. No rendering changes — just mirror what `CompareLayout`, `QmriLayout` already do.

### 4.2 Group `window.*` globals into namespace

233 `window.*` assignments pollute global scope. Introduce a single `App` object:

```js
const App = {
  compareActive: false,
  multiViewActive: false,
  qmriActive: false,
  // ... other window-level properties
};
```

Replace `window.compareActive` → `App.compareActive`, etc. This is mechanical, search-and-replace, low-risk.

### 4.3 Break up `showUnifiedPicker` (435 lines)

Extract nested closures into top-level helper functions:
- `_unifiedPickerPopulate()` — list rendering
- `_unifiedPickerToggleSel()` — selection logic
- `_unifiedPickerCommit()` — file loading + session creation

The 12 nested closures inside the function make it hard to follow. Lifting them out improves readability without changing behavior.

### 4.4 Add `_app.py` re-export test

`_app.py` is a 182-line compat shim with no test ensuring its re-exports stay current. Add to `test_api.py`:

```python
def test_app_shim_exports():
    import arrayview._app as appmod
    expected = ["view", "ViewHandle", "arrayview", "Session", ...]
    for name in expected:
        assert hasattr(appmod, name), f"_app.py missing {name}"
```

---

## Execution Order

```
Phase 1 (1-2 sessions): Dead code + deduplication — immediate payoff, low risk
Phase 2 (1 session):    Split _vscode.py — contained, medium risk
Phase 3 (2 sessions):   Split _launcher.py — larger surface, medium risk
Phase 4 (2-3 sessions): Frontend cleanup — targeted, medium risk
```

**At each phase boundary**: run `uv run pytest tests/ -x -q`, `uv run python tests/visual_smoke.py`, verify no regression.

---

## What We're NOT Doing

- Full VCS rendering migration — multi-week effort, high regression surface
- Splitting `_viewer.html` into multiple files — violates "no build step" constraint
- Removing `_app.py` — backward compatibility break
- `_io.load_data()` dispatch refactor — ugly but battle-tested
- Extracting slice response building across transports — glue code resists clean extraction
- Replacing `_LazyMod` — works, fast-path critical, one instance only
- `_segmentation.py` module-level globals — no multi-session segmentation in practice

---

## Completion Notes (2026-05-05)

### Phase 1: ✓ Done (commit `08b818f`)
- Removed dead TOML parser fallback
- Created `_imaging.py` with shared lazy PIL accessors (replaced in 5 files)
- Extracted `_evict_lru()` cache helper in `_render.py`
- Extracted `_build_mosaic_grid()` shared mosaic builder
- Net: -90 lines

### Phase 2: ✓ Done (commit `526cf09`)
- Split `_vscode.py` into 4 submodules + facade
- `_vscode_extension.py` (346 lines), `_vscode_signal.py` (822 lines), `_vscode_shm.py` (77 lines), `_vscode_browser.py` (176 lines)
- `_vscode.py` → 30-line facade re-exporting all public symbols
- Updated tests for new module structure

### Phase 3: ✗ Reverted
- `_launcher.py` split reverted — 11 test failures due to tight monkeypatching
- Extracted submodules deleted (`_port_utils`, `_loading_server`, `_url_builder`, `_webview`, `_cli_paths`, `_lazy`)
- Requires test refactoring before attempting again

### Phase 4: ✓ Partially done (commit `de96bb2`)
- Added `_app.py` re-export verification test (`test_app_shim_re_exports`)
- Updated `.mex/context/architecture.md` and `project-state.md` for new file layout
- Skipped frontend VCS/LayoutStrategy/App namespace changes — too high-risk in 26K-line file
