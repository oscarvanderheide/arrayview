---
name: add-file-format
description: Adding support for a new file format in _io.py — extension registration, lazy loader, and server wiring.
triggers:
  - "file format"
  - "new format"
  - "load_data"
  - "_io.py"
  - "extension"
  - "_SUPPORTED_EXTS"
edges:
  - target: context/architecture.md
    condition: when understanding where _io.py fits in the load path
  - target: context/conventions.md
    condition: for the lazy import pattern and Verify Checklist
  - target: context/render-pipeline.md
    condition: when the new format returns an unusual dtype that may affect the render pipeline
last_updated: 2026-04-29
---

# Add File Format

## Context

All file loading goes through `_io.load_data(filepath)` in `src/arrayview/_io.py`. The function dispatches on file extension. New formats must:
1. Register the extension in `_SUPPORTED_EXTS`
2. Add a lazy import (if the library is not already imported at module level)
3. Add a dispatch branch in `load_data()`
4. Return a numpy array (or array-like with `.shape` and `.dtype`)

The server (`_server.py`) and CLI (`_launcher.py`) never import format libraries directly — all loading goes through `_io.load_data()`.

## Steps

1. **Register the extension** — add to `_SUPPORTED_EXTS` frozenset in `_io.py`:
   ```python
   _SUPPORTED_EXTS = frozenset([
       ...,
       ".myext",
   ])
   ```

2. **Add a lazy import accessor** — follow the `_nib()` pattern if the library is heavy:
   ```python
   _mylib_mod = None

   def _mylib():
       global _mylib_mod
       if _mylib_mod is None:
           import mylib
           _mylib_mod = mylib
       return _mylib_mod
   ```
   If the library is already a standard dependency and fast to import, a top-level import is acceptable only if it costs <10 ms. When in doubt, make it lazy.

3. **Add a dispatch branch in `load_data(filepath)`** — keep the branch order consistent (extensions most likely to be lazy-loaded go last):
   ```python
   elif ext == ".myext":
       arr = _mylib().load(filepath)
       return np.asarray(arr, dtype=arr.dtype)
   ```

4. **Handle dtype edge cases:**
   - Return a numpy array with a standard dtype (`float32`, `float64`, `complex64`, `uint8`, etc.)
   - Complex structured dtypes (like `.mat` files) must be converted to native complex — see `_fix_mat_complex()` for reference.
   - RGB arrays: if the format natively returns H×W×3 uint8, `_setup_rgb()` in `_render.py` will detect the trailing axis automatically. No special handling needed in `_io.py`.

5. **Add to `FULL_LOAD_EXTS`** if the format loads the entire array into RAM (no mmap/lazy access). This enables the RAM guard check in the server.

6. **Update `_peek_file_shape()`** if the format allows fast shape introspection without loading the full array:
   ```python
   if ext == ".myext":
       # fast path — returns list or None
       return list(_mylib().peek_shape(fpath))
   ```

## Gotchas

- **Do not import the format library at the top of `_io.py`** without checking import cost first. Nibabel is the reference: it is lazy because it costs ~200 ms on first import.
- **`.nii.gz` is two extensions** — the dispatch checks the full suffix, not just the last dot. New formats with compound extensions (`.foo.bar`) need special suffix handling in `load_data()`.
- **Zarr is not loaded via `load_data()`** in all paths — zarr arrays can also be passed directly to `view()` as array objects. Ensure zarr-specific logic lives in `_session.py` / `_render.py`, not only in the file-load path.
- **No format-specific code in `_server.py` or `_launcher.py`** — if you find yourself adding a format check in either of those modules, move it to `_io.py`.

## Verify

- [ ] Extension added to `_SUPPORTED_EXTS`
- [ ] New library import is lazy (accessor function pattern) unless it imports in <10 ms
- [ ] `load_data("myfile.myext")` returns a numpy array with correct shape and dtype
- [ ] `view(load_data("myfile.myext"))` opens the viewer without errors
- [ ] If full-load format, added to `FULL_LOAD_EXTS`
- [ ] `uv run pytest` on `tests/test_mode_consistency.py` still passes (no dtype regressions)

## Update Scaffold
- [ ] Update `.mex/ROUTER.md` "Current Project State" if what's working/not built has changed
- [ ] Update any `.mex/context/` files that are now out of date
- [ ] If this is a new task type without a pattern, create one in `.mex/patterns/` and add to `INDEX.md`
