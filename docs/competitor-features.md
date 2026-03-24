# Feature Ideas for arrayview

Based on analysis of 3D Slicer, arrShow, and original thinking.
Items already in arrayview are excluded.

---

## From Existing Viewers (not yet in arrayview)

### High Impact

1. **Rock / Flicker in Compare Mode** (from Slicer)
   Auto-toggle between two arrays at 2-4 Hz so differences pop out visually. Timer-based
   opacity swap between compare panes. Trivial to implement — arrayview already has compare
   with overlay blend, wipe, and diff modes. Rock/flicker is the missing classic.

2. **Cursor-Following Wipe** (from Slicer "Layer Reveal Cursor")
   The wipe line follows the mouse cursor instead of being slider-controlled. As you move
   left/right, the split point moves with you. Arrayview has wipe mode (`X` cycle) but it's
   slider-based. Making it cursor-aware is a small JS change.

3. **Auto W/L from ROI** (from Slicer)
   Draw a rectangle, auto-set vmin/vmax from that region's percentiles. Solves "one bright
   pixel ruins the dynamic range." Arrayview has ROI mode (`A`) with stats — just needs to
   feed the stats back into vmin/vmax.

4. **Line Profile Plot** (from arrShow)
   Click two points, show an intensity plot along that line as a small SVG overlay. Ruler
   mode (`u`) already captures two points and computes distance — extend it to plot values.

5. **Display Presets** (from Slicer)
   Named W/L + colormap combos (e.g., "CT Bone", "MRI T1 weighted"). Save/recall via
   localStorage. Especially useful for medical imaging where standard windows are well-known.

6. **Interpolation Toggle** (from Slicer)
   Switch between pixelated (current default) and bilinear smoothing when zoomed in.
   Single CSS property toggle: `image-rendering: pixelated` vs `auto`. Useful for
   continuous data where pixel boundaries are distracting.

### Medium Impact

7. **FOV Auto-Clamping** (from arrShow)
   Prevent panning beyond image bounds. Simple bounds check in `_clampPan()`. Reduces
   confusion when zoomed in.

8. **Zoom-to-ROI** (from arrShow)
   Draw a rectangle, zoom to fit exactly that region. Could reuse existing ROI drawing
   and feed the coordinates into the zoom/pan system.

9. **Checkerboard Compare** (from Slicer)
   Alternating tiles of array A and B in a grid pattern. Reveals local registration errors
   that wipe mode misses. Render both arrays, composite with a tiled clip mask.

---

## Original Ideas — Things No Viewer Has

### Temporal Sparklines
Hover a pixel in a 4D+ array and see a tiny inline chart showing how that voxel's value
changes across the non-displayed dimensions (e.g., time). A mini time-series at your cursor.
Implementation: fetch a 1D vector from the server at the hovered (x,y) across the
slice dimension, render as a small SVG sparkline near the cursor.

### Statistical Projections
Beyond showing a single slice, offer max/min/mean/std projection along any axis.
Toggle between "slice 42" and "max projection along dim 2" with one key. Extremely
useful for finding features in 3D volumes (MIP is standard in radiology, but
mean/std projections are novel for debugging).

### Auto-ROI from Click (Flood Fill)
Click a pixel, auto-grow a connected region of similar intensity (flood-fill with
adjustable tolerance via `[`/`]`). Show stats for the auto-selected region. Solves the
"draw a precise ROI by hand" pain point. Server-side: `scipy.ndimage.label` on a
thresholded neighborhood.

### Shared Cursor / Co-Viewing
Two researchers open the same URL. Both see each other's cursor position in real-time
via WebSocket. One navigates, the other observes. Web-based advantage — impossible in
desktop viewers without screen sharing. Critical for remote radiology review and
mentoring.

### Snapshot Gallery
A persistent side panel collecting screenshots taken during a session (press `s`),
showing thumbnails with slice position metadata. At the end, export as an HTML report
or a grid image. Turns exploration into documentation automatically.

### Annotation Bookmarks
Mark specific slice positions with short notes ("artifact here", "lesion boundary")
that persist across sessions via URL state. Sharable — paste the URL and the recipient
sees all bookmarks. Export as JSON for programmatic use.

### Dimension Reduction Preview
For arrays with 5+ dimensions, show a small tree/diagram visualizing which dimensions
are displayed (x, y), which are sliced (with current index), and which are collapsed.
Makes navigation intuitive instead of "which dim is active again?"

### Conditional Auto-Play
"Show me only slices where the mean intensity exceeds X" — a filter that skips
uninteresting slices during auto-play. Useful for sparse 4D data (e.g., fMRI
activation maps where most time points are baseline).

### Drag-and-Drop Compare
Drag a .npy/.nii file from the OS file manager onto the viewer to instantly add it to
compare mode. The browser drag-and-drop API makes this possible. No CLI round-trip.

### Progressive Blur-Up Loading
For large Zarr arrays over SSH: show a heavily downsampled preview immediately (2x2
thumbnail) while the full-resolution slice streams in. Navigation feels instant even
on slow connections. Server sends a low-res frame first, then upgrades.

### WebGPU Volume Rendering
Use WebGPU compute shaders for real-time MIP / ray casting directly in the browser.
No server round-trip for rotation. This would be a category-defining feature — no
other web-based array viewer does GPU volume rendering.

### Pixel History on Hover
When `--watch` mode is active and the file changes on disk, track per-pixel changes
over reloads. Hover a pixel to see "this value was 0.42, now 0.87 (+107%)". Useful
for iterative reconstruction debugging.
