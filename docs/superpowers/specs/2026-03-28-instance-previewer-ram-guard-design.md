# Instance Previewer Sidebar & RAM Guard

## Overview

Two features for the unified picker (Cmd/Ctrl/Shift+O):

1. **Instance previewer sidebar** — visual preview of all open instances with thumbnails and metadata
2. **RAM guard** — hard block when loading files that would exceed available memory

## Instance Previewer Sidebar

### Layout

The picker modal expands to accommodate a **left sidebar** (200px) alongside the existing file picker (right side). The sidebar lists all open instances; the file picker retains its current behavior (search, file list, multi-select for compare).

When no instances are open (only one session, which is current), the sidebar is hidden and the picker renders at its current width.

### Per-Instance Card

Each card shows:
- **Frozen thumbnail** (48x36px) — snapshot of the instance's current view at the moment the picker opens
- **Name** — session name, ellipsis overflow
- **Shape** — e.g. `(256, 256, 128)`
- **Dtype + current slice** — e.g. `float32 · z=64`
- **"(current)" badge** — yellow text on the active session

Visual treatment:
- Yellow left border (2px) on the current instance
- Hover highlight on non-current instances
- Dark card background matching arrayview's existing dark theme

### Thumbnail Generation

When the picker opens, fetch a snapshot for each session via `GET /frame/{sid}` with small dimensions (e.g. `w=96&h=72`). These are frozen — no live updates while the picker is open.

The `/frame/{sid}` endpoint already exists and returns a JPEG of the current view state. A new query parameter `thumbnail=1` can return a smaller, lower-quality version to reduce latency.

### Interaction

- **Click** an instance card → picker closes immediately, switches to that instance
- **Keyboard**: Arrow keys / j/k navigate the currently focused panel. Tab cycles focus: search field → sidebar → file list. Enter selects the highlighted item in whichever panel has focus. When the picker opens, focus starts on the search field (same as today).
- **Search** filters both sidebar instances and file list simultaneously

### Footer

Bottom of the sidebar shows: `{N} instances · {X} GB used` (sum of estimated memory across all sessions).

### Backend Changes

New endpoint or extension to `GET /sessions`:
```
GET /sessions?thumbnails=1
```
Returns session list with an additional `thumbnail_url` field pointing to `/frame/{sid}?w=96&h=72&q=60`.

Alternatively, the frontend fetches thumbnails in parallel after receiving the session list. This is simpler and avoids changing the `/sessions` response format.

**Recommended approach**: Frontend fetches thumbnails in parallel. The sidebar renders immediately with a placeholder gradient, then swaps in the real thumbnail when the fetch completes. This keeps the picker feeling instant.

### Frontend Changes

- `showUnifiedPicker()` in `_viewer.html`: add sidebar DOM construction
- Widen the picker overlay when instances > 1
- Add CSS for sidebar cards, thumbnail placeholders, hover states
- Add keyboard navigation for sidebar focus

## RAM Guard

### Scope

Only applies to formats that **fully load into RAM**:
- `.pt` / `.pth` (PyTorch — converted to numpy in memory)
- `.tif` / `.tiff` (tifffile loads entire image)
- `.mat` (scipy/h5py loads full array)

Formats that use memory-mapping or lazy access load silently with no guard:
- `.npy` (numpy mmap)
- `.npz` (numpy, small overhead)
- `.zarr` / `.zarr.zip` (chunked lazy access)
- `.nii` / `.nii.gz` (nibabel lazy proxy)
- `.h5` / `.hdf5` (h5py lazy access)

### Estimation

Before loading a full-load format:

1. **Estimate array size**: For `.pt`/`.mat`, use file size as a rough estimate (actual in-memory size may differ due to compression, but file size is a safe lower bound). For `.tif`, use file size (tifffile loads the raw pixel data).
2. **Check available RAM**: `psutil.virtual_memory().available`
3. **Compare**: If estimated size > available RAM → block

### Block Behavior

Hard block with no override. The user sees a dialog:

```
⚠ Not enough memory

model_large.pt requires ~8.2 GB
Available RAM: 3.1 GB

Close other instances or free memory to continue

[OK]
```

The dialog is dismissable (OK or Escape), returning the user to the picker.

### Backend Changes

New endpoint or pre-check in the `/load` handler:

```python
# In POST /load handler, before actually loading:
if _is_full_load_format(filepath):
    estimated = os.path.getsize(filepath)
    available = psutil.virtual_memory().available
    if estimated > available:
        return JSONResponse(
            {"error": "insufficient_memory",
             "estimated_bytes": estimated,
             "available_bytes": available,
             "filename": os.path.basename(filepath)},
            status_code=507  # Insufficient Storage
        )
```

### Frontend Changes

- In the picker's file-open handler: check the `/load` response for `error: "insufficient_memory"`
- Render the block dialog over the picker
- OK / Escape dismisses the dialog, returns to picker

### Environment Variable Override

For advanced users who know what they're doing (e.g. swap-backed systems):

```
ARRAYVIEW_SKIP_RAM_GUARD=1
```

Disables the RAM check entirely. Not exposed in any UI.

## Files to Modify

| File | Changes |
|------|---------|
| `_viewer.html` | Sidebar DOM, CSS, keyboard nav, RAM block dialog |
| `_server.py` | RAM check in `/load`, optional thumbnail params on `/frame` |
| `_session.py` | Helper to estimate session memory usage (for footer total) |
| `_io.py` | Classify formats as full-load vs lazy (already implicit, make explicit) |

## Out of Scope

- Live-updating thumbnails while picker is open
- Memory-mapped format dialogs or notes (silent load)
- RAM guard for lazy formats
- Drag-and-drop onto the sidebar
- Reordering instances in the sidebar
