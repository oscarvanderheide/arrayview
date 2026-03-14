# Design: TODO Batch — Six Feature Items
Date: 2026-03-14

## 1. Multi-view canvas sizing

**Goal:** Each of the three multi-view panes must have the same square bounding box. The rendered image is centered within that box with black padding (letterbox or pillarbox), preserving the slice's native aspect ratio. No stretching.

**Change:** In `mvDrawFrame` (JS), replace the stretched `drawImage(img, 0, 0, T, T)` call with a centered draw that computes offset and scaled dimensions from the image's natural width/height relative to the T×T target box.

**No server-side changes needed** — the server already returns correctly-sized RGBA data.

---

## 2. Overlay: multiple masks

**Goal:** `view(arr, overlay=[mask1, mask2, mask3])` loads each mask as a separate session and composites them in order with auto-assigned colors from a fixed palette (red, green, blue, yellow, cyan, magenta, …). Where masks overlap, the later mask in the list is drawn on top.

**Changes:**
- Python `view()`: accept `overlay` as a single array or list of arrays; load each as its own session; pass list of `overlay_sid` values to the viewer URL
- URL param: `overlay_sid=sid1,sid2,sid3` (comma-separated)
- JS: parse `overlay_sid` into an array; send each sid + assigned color with each WS/HTTP render request
- Server: composite each overlay in a loop, applying the assigned palette color

**Palette:** Fixed list of visually distinct RGBA colors at a reasonable alpha (e.g., 0.5). First overlay = red, second = green, etc.

---

## 3. Overlay: heatmap for continuous masks

**Goal:** When an overlay array has float dtype or more than ~16 unique integer values, render it as a heatmap (desaturated jet colormap) rather than per-label solid colors.

**Change:** Update `_overlay_is_label_map` (in `_render.py`) to return `False` for float arrays and for integer arrays with many unique values. The existing `_build_desaturated_jet` path already handles the heatmap rendering.

**Threshold:** `>16 unique values` distinguishes label maps (organ IDs) from continuous data.

---

## 4. Drag-and-drop array files

**Goal:** User drags a `.npy` or `.mat` file onto an open arrayview viewer window and it opens in a new tab, in all environments (localhost, VS Code, SSH tunnel) because file bytes travel over HTTP — no file path needed.

**Changes:**
- JS in `_viewer.html`: add `dragover` + `drop` event handlers on the main canvas (and compare canvases). On drop, read file bytes via `FileReader`, POST to `/load-upload`.
- New endpoint `POST /load-upload` in `_server.py`: accepts multipart file upload, saves to a `tempfile`, calls existing load logic, returns `{sid, name}`.
- JS: on success, call existing `addTab(sid, name)` to open in shell tab (same behavior as `view()` call).
- Show a brief "Loading <filename>…" toast while upload is in progress.

---

## 5. Colorbar scroll/drag in other modes

**Goal:** Scrolling or dragging on the colorbar adjusts the display range in multi-view and compare modes, not just normal mode.

**Changes:**
- **Multi-view:** Remove the `multiViewActive` guard from the colorbar wheel and drag handlers. The shared colorbar already controls all three panes — no further changes needed.
- **Compare mode:** Each pane has its own colorbar. Track which pane the mouse is currently over (`hoverPane`). Colorbar scroll/drag targets that pane's `manualVmin`/`manualVmax`. Double-click resets that pane's range.

---

## 6. Vector field arrow density (`[` / `]`)

**Goal:** `[` decreases and `]` increases arrow density in vector field mode. Change takes effect immediately on next render.

**Changes:**
- JS: add `vfDensityLevel` state variable (integer, default 0, clamped to −3…+3).
- Keyboard handler: `[` → `vfDensityLevel--`, `]` → `vfDensityLevel++`, then re-render and show toast `"arrow density: " + level`.
- Pass `density_offset=vfDensityLevel` in vectorfield HTTP request.
- Server (`/vectorfield`): `stride = max(1, base_stride >> density_offset)` (positive = denser, negative = sparser) or use `stride * 2^(-density_offset)`.
- Only active when in vector field mode; ignored otherwise.
