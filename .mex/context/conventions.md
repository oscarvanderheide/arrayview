---
name: conventions
description: How code is written in arrayview — naming, structure, lazy imports, and frontend organization. Load when writing new code or reviewing existing code.
triggers:
  - "convention"
  - "pattern"
  - "naming"
  - "style"
  - "how should I"
  - "what's the right way"
  - "lazy import"
edges:
  - target: context/architecture.md
    condition: when a convention depends on understanding the system structure
  - target: context/stack.md
    condition: when the convention is library-specific
  - target: patterns/frontend-change.md
    condition: when writing frontend (HTML/CSS/JS) code
last_updated: 2026-04-13
---

# Conventions

## Naming

- **Module files** — all private, prefixed with `_` (`_launcher.py`, `_render.py`, `_server.py`). Only `__init__.py` and `__main__.py` are public entry points.
- **Functions** — `snake_case`, internal helpers prefixed with `_` (e.g. `_compute_vmin_vmax`, `_in_jupyter`)
- **Constants** — `SCREAMING_SNAKE_CASE` for module-level constants (`SESSIONS`, `LUTS`, `COLORMAPS`, `HEAVY_OP_LIMIT_BYTES`)
- **Classes** — `PascalCase` (`Session`, `ViewHandle`, `TrainingMonitor`, `_LazyMod`)
- **Frontend section separators** — `/* ── Section Name ── */` in CSS, `// ── Section Name ──` in JS (double-dash with spaces, em-dashes)

## Structure

- **Lazy imports everywhere** — any import that costs >10 ms must be deferred. Pattern: module-level `_cache = None` + accessor function (see `_server_mod()`, `_uvicorn()`, `_nib()`, `_init_luts()`). The CLI fast path (server already running) must stay near-zero cost.
- **Global state lives in `_session.py`** — `SESSIONS`, `SERVER_LOOP`, `VIEWER_SOCKETS`, `SHELL_SOCKETS`. Other modules import these by name, never redefine them.
- **Render pipeline is pure functions** — `_render.py` exports stateless functions that take `session` + parameters. No class hierarchy.
- **One HTML file for the entire frontend** — all CSS and JS live in `_viewer.html`. Do not split into separate files.
- **Tests live in `tests/`** — not next to source files. Visual/browser tests use playwright and are marked `@pytest.mark.browser`.
- **All file-format loading goes through `_io.load_data(filepath)`** — no direct format imports in `_server.py` or `_launcher.py`.

## Patterns

**Lazy module import pattern** (used for all heavy dependencies):
```python
# At module level
_nib_mod = None

def _nib():
    global _nib_mod
    if _nib_mod is None:
        import nibabel
        _nib_mod = nibabel
    return _nib_mod

# Usage: _nib().load(filepath)  — never `import nibabel` at top level
```

**Session cache pattern** (LRU via `OrderedDict`):
```python
key = (dim_x, dim_y, key_idx)
if key in session.raw_cache:
    session.raw_cache.move_to_end(key)  # LRU update
    return session.raw_cache[key]
# ... compute result ...
session.raw_cache[key] = result
session._raw_bytes += result.nbytes
while session._raw_bytes > session.RAW_CACHE_BYTES and session.raw_cache:
    _, v = session.raw_cache.popitem(last=False)
    session._raw_bytes -= v.nbytes
```

**Environment detection order** (in `_platform.detect_environment()`):
```
jupyter → vscode → julia → ssh → terminal
```
Always check in this priority order. Never short-circuit.

**Float rendering convention** — slices are always converted to `np.float32` before applying colormap. RGB uint8 arrays skip colormap and go directly to RGBA. Complex arrays are transformed by `apply_complex_mode()` before any colormap step.

## Verify Checklist

Before presenting any code change:
- [ ] New heavy imports are lazy (wrapped in a `_mod = None` / accessor function pattern)
- [ ] Any new `Session` field is initialized in `Session.__init__` and cleared in `reset_caches()` if it's cache-related
- [ ] New file format support goes through `_io.load_data()` and adds the extension to `_SUPPORTED_EXTS`
- [ ] Frontend changes are in `_viewer.html` only — no new JS/CSS files
- [ ] New rendering functions follow the `extract_slice → apply_complex_mode → apply_colormap_rgba` pipeline order
- [ ] Environment detection changes go in `_platform.py`, not inline in `_launcher.py` or `_vscode.py`
- [ ] Cross-mode consistency: any new visual feature verified in all six invocation environments (see `invocation-consistency` skill)
