# Design: TODO Batch — Six Feature Items
Date: 2026-03-14

## 1. Multi-view canvas sizing

**Goal:** Each of the three multi-view panes must have the same square bounding box. The rendered image is centered within that box with black padding (letterbox or pillarbox), preserving the slice's native aspect ratio. No stretching.

**Current behavior:** `mvScaleAllCanvases` computes `globalScale = T / maxSide` and sets each canvas's CSS size to `lastW * globalScale` × `lastH * globalScale`. Pane containers are T×T. Canvases are CSS-scaled proportionally, but the pane's flex layout may not center them correctly, leaving alignment inconsistencies.

**Change:** In `mvScaleAllCanvases`, after computing the canvas CSS dimensions (`cssW × cssH`), ensure the T×T `.mv-pane` container has a black background and that the canvas/wrap chain is visually centered within it. The actual DOM hierarchy is `.mv-pane` (T×T, flex-centering) > `.mv-inner` > `.mv-canvas-wrap` (sized to `cssW × cssH`) > canvas. If flex centering on `.mv-pane` already centers `.mv-inner` correctly, the fix may be purely a background color on `.mv-pane` (currently transparent, so padding area shows the page background instead of black). Verify during implementation and apply the minimal fix to achieve centered image with uniform black padding.

---

## 2. Overlay: multiple masks

**Goal:** `view(arr, overlay=[mask1, mask2, mask3])` loads each mask as a separate session and composites them in order with auto-assigned colors. Where masks overlap, the later mask in the list is drawn on top.

**Changes:**
- Python `view()`: accept `overlay` as a single array or list of arrays; load each as its own session; pass list of `overlay_sid` values to the viewer URL as `overlay_sid=sid1,sid2,sid3`
- JS: parse `overlay_sid` into an array; for each entry, look up its auto-assigned palette color; send `overlay_sids` (comma-separated) and `overlay_colors` (comma-separated hex, e.g. `ff4444,44ff44,4444ff`) with each WS/HTTP render request
- Server: in the WS handler, parse `overlay_sids` and `overlay_colors`; iterate, calling `_extract_overlay_mask` + `_composite_overlay_mask` for each. Color rules per overlay:
  - **Binary mask** (values 0/1 only): use the palette-assigned color passed from JS
  - **Multi-label integer mask** (values 0…N with N≤16): keep using `LABEL_COLORS` per label value (per-label discrimination matters more than palette assignment)
  - **Continuous/heatmap** (float or >16 unique values): use `_DESATURATED_JET_LUT` as before, palette color ignored
- Add `override_color: tuple[int,int,int] | None` parameter to `_composite_overlay_mask`; only applied when the mask is binary
- **Palette:** Fixed list of visually distinct colors at alpha ~0.5: `[#ff4444, #44cc44, #4488ff, #ffcc00, #ff44ff, #44ffff]`

---

## 3. Overlay: heatmap for continuous masks

**Goal:** When an overlay array has float dtype or more than 16 unique integer values, render it as a heatmap (desaturated jet colormap) rather than per-label solid colors.

**Current behavior:** `_overlay_is_label_map` at `_render.py:448` returns `True` for any integer dtype regardless of the number of unique values.

**Change:** Update `_overlay_is_label_map` to return `False` if the array has float dtype OR if it has integer dtype but more than 16 unique values. Sample from the current slice (not the full N-D array) to keep cost low. The existing `_build_desaturated_jet` / `_DESATURATED_JET_LUT` heatmap path already handles the non-label case correctly.

---

## 4. Drag-and-drop array files

**Goal:** User drags a `.npy` or `.mat` file onto an open arrayview viewer window and it opens in a new tab. Works in all environments because file bytes travel over HTTP — no file path needed.

**Changes:**
- JS in `_viewer.html`: attach `dragover` + `drop` handlers via event delegation on `#viewer-wrap` (parent of all canvases, including dynamically-created compare canvases). On drop, read file bytes via `FileReader.readAsArrayBuffer`, POST to `/load-upload` as `multipart/form-data`.
- New endpoint `POST /load-upload` in `_server.py`: accepts a single file upload, saves to `tempfile.NamedTemporaryFile(suffix=<ext>, delete=False)`, calls the existing load logic, returns `{sid, name}`.
- JS: show toast "Loading <filename>…" during upload; on success call existing `addTab(sid, name)` to open in a shell tab.
- Accept `.npy` and `.mat` only; show toast error for other types.

---

## 5. Colorbar scroll/drag in other modes

**Goal:** Scrolling or dragging on the colorbar adjusts the display range in multi-view and compare modes.

**Multi-view:** Remove `multiViewActive` from the early-return guards on the `slimCbCanvas` wheel handler and the window `mousemove` drag handler (lines ~929 and ~948). The shared `manualVmin`/`manualVmax` already feeds all three panes — no further state changes needed.

**Compare mode:** Requires two steps:
1. **Add per-pane vmin/vmax state:** introduce `cmpManualVmin[]` and `cmpManualVmax[]` arrays indexed by pane (currently all panes share the single `manualVmin`/`manualVmax`). Update each compare render call to use the per-pane value.
2. **Enable colorbar interaction:** remove `compareActive` from the early-return guard; track which compare pane the mouse is currently over (`hoverCmpPane` state updated on `mousemove`); colorbar wheel/drag targets `cmpManualVmin[hoverCmpPane]` / `cmpManualVmax[hoverCmpPane]`. Double-click on a pane's colorbar resets that pane's range.
3. Re-enable `pointer-events` on `.compare-pane-cb` elements (currently set to `none`).

qMRI mode stays disabled for now.

---

## 6. Vector field arrow density (`[` / `]`)

**Goal:** `[` decreases and `]` increases arrow density in vector field mode.

**Key conflict:** `[` and `]` are already bound to overlay alpha (and registration blend in compare mode). The new behavior is added as an additional conditional branch: when `hasVectorfield && !overlay_sid && !registrationMode`, `[`/`]` control density instead.

**Changes:**
- JS: add `vfDensityLevel` state variable (integer, default 0, clamped to −3…+3).
- Keyboard handler: in the `[`/`]` branch, add `else if (hasVectorfield && !overlay_sid && !compareActive)` → adjust `vfDensityLevel`, re-render, show toast `"arrow density: " + level`.
- Pass `density_offset=vfDensityLevel` in the vectorfield HTTP GET request.
- Server (`/vectorfield`): `stride = max(1, round(base_stride * (2 ** -density_offset)))` — positive offset = smaller stride = denser arrows; negative = sparser. (Avoid bit-shift: Python raises `ValueError` for right-shift by negative amounts.)
