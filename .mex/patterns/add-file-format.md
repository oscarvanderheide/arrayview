---
name: add-file-format
description: Adding support for a new array file format to arrayview. Covers _io.py, _SUPPORTED_EXTS, CLI arg validation, and tests.
triggers:
  - "file format"
  - "new format"
  - "load"
  - "_io.py"
  - "extension"
  - ".ext"
edges:
  - target: context/architecture.md
    condition: for understanding how load_data() fits into the server pipeline
  - target: context/stack.md
    condition: for understanding which loading libraries are already available
  - target: context/conventions.md
    condition: for the lazy import pattern and Verify Checklist
last_updated: 2026-04-13
---

# Add File Format

## Context

All file loading goes through `src/arrayview/_io.py:load_data(filepath)`. The function dispatches on file extension. `_SUPPORTED_EXTS` is a `frozenset` used by the CLI arg validator. `FULL_LOAD_EXTS` tracks formats that materialize fully into RAM (for the heavy-op guard in `_server.py`).

## Steps

1. Add the new extension(s) to `_SUPPORTED_EXTS` in `_io.py`
2. If the format loads the full array into RAM (not mmap/lazy): add to `FULL_LOAD_EXTS`
3. Add a branch to `load_data()` — use the lazy import pattern for any new library:
   ```python
   elif filepath.endswith(".newext"):
       import newlibrary  # inline, not at module top
       return newlibrary.load(filepath)
   ```
4. If the format can return a structured complex dtype (like `.mat`): apply `_fix_mat_complex()`
5. If NIfTI-like (spatial metadata): implement in `load_data_with_meta()` too
6. Update `_peek_file_shape()` if a quick shape-peek is feasible without loading
7. Add the extension to the error message in the `else` branch at the bottom of `load_data()`
8. Add a test in `tests/` — at minimum a unit test that loads a tiny sample file

## Gotchas

- **Lazy imports are mandatory** — do not add `import newlibrary` at the top of `_io.py`. The entire module must import cheaply.
- **`.nii.gz` precedence** — extension matching uses `endswith()`; check `.nii.gz` before `.nii` (already done for nifti, but keep in mind for similar compound extensions).
- **Mmap vs eager** — only formats with seekable backing stores can use mmap. Gzip is not seekable — `.nii.gz` is always materialized. For new compressed formats, materialize too.
- **Multi-array files** — if the format can contain multiple arrays, follow the `.npz`/`.h5`/`.mat` pattern: if `len(keys) == 1` load silently, otherwise raise a `ValueError` telling the user to load manually.
- **Complex arrays from `.mat`** — `scipy.io.loadmat` returns structured dtypes with `real`/`imag` fields; `_fix_mat_complex()` handles this. Apply it to any format with similar behavior.
- **MATLAB v7.3** — these are HDF5 files; `.mat` already falls back to `h5py` via `except NotImplementedError`.

## Verify

- [ ] Extension added to `_SUPPORTED_EXTS` (frozenset in `_io.py`)
- [ ] Full-RAM formats added to `FULL_LOAD_EXTS`
- [ ] New library imported lazily (inside the `elif` branch, not at module top)
- [ ] Error message in the `else` branch updated to include new extension
- [ ] Test file loads cleanly: `uv run pytest tests/ -k "new_format"`
- [ ] `uv run pytest tests/test_view_component_unit.py` still passes

## Debug

If loading hangs: the format is likely not seekable (check if it's compressed). Materialize with `np.asarray(lazy_obj)`.
If wrong dtype comes back: check for structured complex dtype and apply `_fix_mat_complex()`.
If shape is wrong: check dimension ordering — some formats are column-major (MATLAB); may need `.T`.

## Update Scaffold
- [ ] Update `.mex/ROUTER.md` "Current Project State" to list the new format under Working
- [ ] Update `context/stack.md` if a new library was added
- [ ] Add row to `patterns/INDEX.md` if this pattern was newly created
