# Competitor Feature Analysis

Research into two scientific array/image viewers to identify borrowable ideas for arrayview.

---

## 1. 3D Slicer

Desktop application (C++/Python/Qt) for medical image visualization and analysis. Massive ecosystem with extensions, but the core viewer has patterns worth studying.

### Core Viewing Features

- **Three-plane slice views** (Axial/Sagittal/Coronal) with colored bar identifiers (Red, Yellow, Green) -- similar to arrayview's multiview
- **Slice offset slider** with adjustable step size (snaps to voxel spacing by default)
- **Window/level adjustment**: left-click-drag in slice views; Ctrl+drag for auto-optimal W/L in a drawn region
- **Reset W/L**: Ctrl+left-double-click resets to auto range
- **Foreground/background blending**: opacity slider to fade between two overlaid volumes
- **Interpolation toggle**: linear vs. nearest-neighbor (useful for label maps)
- **Threshold transparency**: intensity range below/above threshold becomes transparent in foreground layer

### Colormaps & Lookup Tables

- Two types: discrete (named color lists) and continuous (smooth interpolation between arbitrary control points)
- Categories: Discrete (Grey, Iron, Rainbow, Ocean, fMRI, fMRIPA), FreeSurfer (Heat, BlueRed), PET scales, Cartilage MRI (dGEMRIC), user-generated
- Per-volume colormap assignment (each loaded volume can have its own LUT)
- Predefined display presets that set both W/L and colormap together (e.g., "CT Bone", "CT Soft Tissue")

### Histogram

- Shown in Volumes module: pixel count (y) vs intensity (x), overlaid on the current W/L + threshold mapping
- Serves as visual feedback while adjusting window/level

### Crosshair & Linked Views

- **Shift+mouse move** synchronizes crosshair position across ALL slice views simultaneously
- **View linking**: groups of views share zoom, pan, and scroll position
- **Hot-linked mode**: synchronization happens during drag, not just on release
- Parallel views (same orientation) auto-sync center position

### Compare Volumes Module

- **Crossfade**: smooth blend between foreground/background
- **Rock/Flicker**: animated toggling between two volumes to spot differences
- **Layer Reveal Cursor**: checkerboard pattern of fg/bg follows cursor in real-time
- **Side-by-side layouts**: configurable grid (1-up, 3-over-3, etc.)
- **Common background**: overlay multiple volumes against one reference

### Layout System

- Predefined layouts: Four-Up (3 slices + 3D), Three-over-three, single/dual views
- Layout persists between sessions
- Double-click any view to maximize/restore
- Views are independently configurable but can be linked

### Mouse Interactions (Slice Views)

| Action | Result |
|--------|--------|
| Right-drag up/down | Zoom |
| Ctrl+scroll | Zoom |
| Middle-drag | Pan |
| Shift+left-drag | Pan |
| Left/Right arrows, `b`/`f` | Prev/next slice |
| `r` | Reset zoom & pan |
| `v` | Toggle slice visibility in 3D |
| `g` | Toggle segmentation overlay |
| `t` | Toggle foreground visibility |
| `[`/`]` | Cycle background volume |
| `{`/`}` | Cycle foreground volume |

### Measurements & Annotations

- **Markups module**: point lists, lines, angles, curves, closed curves, planes, ROI boxes
- **Ruler**: distance measurement between two points (Ctrl+m)
- **Segment Statistics**: voxel count, volume (mm3/cm3), min/max/mean/median/stdev of intensity, surface area, shape descriptors (roundness, flatness, elongation), Feret diameter, centroid

### Segmentation Tools

Paint, Draw, Erase, Level Tracing, Threshold, Grow from Seeds, Fill Between Slices, Scissors, Islands, Smoothing, Hollow, Margin, Logical Operators, Mask Volume. Most relevant to arrayview: **Threshold** (interactive intensity-based selection) and **Level Tracing** (auto-outline at cursor intensity).

### Screen Capture & Export

- Single frame, 3D rotation video, slice sweep (animate through slices), slice fade (animate fg/bg blend), sequence playback
- Output: image series, animated GIF (color/grayscale), MP4/AVI
- "Capture all views" mode composites the full layout

### Volume Rendering

- CPU and GPU ray casting
- Transfer function editing (opacity + color vs intensity)
- Presets for anatomy types
- MIP (maximum intensity projection), MinIP
- ROI cropping box
- Adaptive quality (frame-rate responsive)

### Data Probe

- Always-visible panel showing: slice name, RAS coordinates, orientation, voxel IJK, intensity value, segment name at cursor
- Updates in real-time as cursor moves

---

## 2. arrShow (arrayShow)

MATLAB/Java desktop tool (v0.35) specifically designed for multidimensional complex-valued MRI data. Single-developer project by Tilman Sumpf. Much smaller scope than Slicer but highly relevant to arrayview's use case.

### Core Viewing Features

- **N-dimensional array viewer** with value-changer controls per dimension (like sliders/spinners for each axis)
- **True size mode**: pixel-perfect 1:1 rendering
- **Aspect ratio control**: adjustable display aspect ratio
- **Frame playback**: play/pause animation through a dimension at configurable FPS (default 50)
- **Surface plot**: generate 3D surface visualization from 2D slice

### Complex Data Handling (Unique Strength)

- **Complex part chooser**: toggle between Magnitude, Phase, Real, Imaginary, and combined Complex view
- **Phase overlay**: overlays phase as color on magnitude image (even when imaginary part is zero)
- **Separate colormaps** for real data vs phase data
- **Phase circle**: visual phase indicator drawn at cursor position
- **Complex conjugate** and **negation** as one-click operations

### Windowing (Dynamic Range)

- **Center/Width model**: interactive contrast via mouse drag (right-click)
- **Absolute vs Relative windowing**: absolute uses fixed intensity values; relative uses percentage of data range
- **Send windowing to relatives**: broadcast current W/L to all linked viewer instances
- **CLim (color limit)** direct control

### ROI Tools

- **Freehand ROI drawing** with multiple vertices
- **Circular ROI**: auto-convert 2-point selection to circle
- **ROI statistics**: mean and standard deviation within region
- **ROI position copy/paste**: transfer ROI definitions between views
- **ROI persistence**: save/reload ROI definitions to disk

### Statistics & Analysis

- **Image statistics panel**: min, max, L2 norm displayed in control panel
- **Cursor pixel readout**: real-time intensity at cursor position
- **ROI-specific stats**: mean, stdev within selected region
- **impixelregion** (Shift+Z): MATLAB's pixel inspection tool for detailed examination

### Colormaps

- Standard MATLAB colormaps + custom .mat-file colormaps
- Separate phase colormap
- Colormap editor integration (MATLAB's colormapeditor)
- Colorbar toggle (disabled automatically in complex mode)
- Colormap persistence: save current for reuse

### Zoom & Navigation

- **Mouse wheel zoom** with configurable zoom factor (default 1.5x per step)
- **Zoom around centerpoint**: zoom relative to a specific coordinate
- **Zoom copy/paste**: transfer zoom settings between instances
- **FOV auto-clamping**: prevents zooming/panning outside image bounds
- **Dimension permutation & reshaping** from within the viewer

### Data Operations (In-Viewer)

- **FFT/iFFT** with fftshift (Shift+F / Shift+D)
- **90-degree rotation** (forward/backward)
- **Complex conjugate**, **negation**
- **Center-based and zoom-based cropping**
- **Squeeze/Permute**: dimension manipulation without leaving the viewer
- **Destructive selection** (Shift+S): subsetting data permanently
- **Post-processing function hooks**: run custom MATLAB functions on every image update

### Send/Broadcast System ("Relatives")

- **Linked instances**: multiple arrShow windows share settings
- **Selective broadcasting**: send by group or to all relatives
- **Linkable properties**: selection (which slice), windowing, colormap, cursor position, zoom
- **Global object registry** (asObjs): central management of all open viewers

### Export

- **Batch multi-frame export**: image series (PNG/BMP/EPS) or video (AVI)
- **Configurable framerate** for video export
- **Screenshot options**: include/exclude control panel, cursors, ROI overlays
- **Metadata export**: dimensions, statistics, ROI data, version info
- **NaN transparency**: automatic transparent pixels in PNG output

### Markers & Annotations

- **Pixel markers**: place colored markers on specific pixels
- **Phase-aware marker colors**: white for phase data, yellow for real data
- **Image text overlay**: render text directly onto image
- **Title-as-text**: display figure title within the image area

### GUI Layout

- **Control panel** (top): dimension selectors, windowing controls, statistics readout, complex-part chooser
- **Image panel** (center): main display area
- **Bottom panel**: cursor coordinates, status
- **Toolbar**: play/pause, zoom controls, lock, colorbar toggle, phase circle toggle
- **Menu bar**: File, Operations, Tools, Relatives

### Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| Ctrl+E | Export current image |
| Shift+F | FFT all images |
| Shift+D | iFFT all images |
| Ctrl+Shift+F | FFTshift |
| Shift+S | Destructive selection |
| Shift+Z | Pixel inspection (impixelregion) |
| Mouse wheel | Scroll through dimensions |
| +/- | Scroll through dimensions |
| Right-click drag | Adjust windowing |
| Double-click | Reset windowing |

---

## Borrowable Ideas for arrayview

Ranked by impact and feasibility for a web-based viewer.

### High Impact, High Feasibility

1. **Layer Reveal Cursor** (from Slicer Compare Volumes)
   A checkerboard or wipe pattern that follows the cursor, showing array A on one side and array B on the other. Extremely useful for registration comparison. Would be a natural extension of arrayview's existing compare mode. Implementation: render both arrays, use a clip mask or canvas compositing at cursor position.

2. **Crossfade / Rock / Flicker in Compare Mode** (from Slicer)
   Animated toggling or smooth blending between two arrays. Rock/flicker are trivially implementable (timer-based opacity toggle). Crossfade just needs an opacity slider. All three are standard comparison techniques in medical imaging.

3. **Complex Part Chooser** (from arrShow)
   Toggle between Magnitude, Phase, Real, Imaginary views of complex-valued arrays. If arrayview supports complex numpy arrays, this is essential. Could be a keyboard shortcut cycle (like `c` for colormap). Phase overlay on magnitude is a particularly compelling visualization.

4. **Broadcast/Link Settings Across Instances** (from arrShow "Relatives")
   When multiple arrayview sessions are open (e.g., in shell tabs), allow linking zoom, slice position, colormap, and windowing across them. Partial overlap with compare mode, but more general.

5. **Auto Window/Level from ROI** (from Slicer)
   Ctrl+drag a rectangle, auto-compute optimal W/L from that region. Solves the "one bright pixel ruins the dynamic range" problem. Feasible: compute min/max or percentile from selected pixel region.

### High Impact, Medium Feasibility

6. **Display Presets** (from Slicer)
   Named W/L + colormap combinations (e.g., "CT Bone", "MRI T1", "Phase Map"). Users could save and recall presets. Needs a small persistence layer (localStorage or server-side).

7. **Threshold Transparency** (from Slicer)
   Make pixels below/above an intensity threshold transparent, revealing a background layer. Useful for overlaying masks or functional maps on structural images. Would need a dual-layer rendering approach.

8. **Line Profile / Plot Along Dimension** (from arrShow)
   Click a line across the image, see an intensity plot along that line. Common in scientific viewers. Could render as an SVG overlay or a small chart panel.

9. **Per-Pixel Data Probe Panel** (from Slicer)
   Always-visible panel showing coordinates, intensity value, array name, dtype, and current slice position. arrayview already has hover info (`H` key), but a persistent pinned panel is more discoverable and useful during analysis.

10. **In-Viewer FFT Toggle** (from arrShow)
    One-key toggle to view the Fourier transform of the current slice. Very useful for MRI k-space inspection. Server-side computation via numpy, then display the result. Feasible but needs a round-trip.

### Medium Impact, High Feasibility

11. **Slice Sweep Animation** (from Slicer Screen Capture)
    Animate through all slices as a GIF/video. arrayview already has GIF export, but adding a "sweep" mode that automatically cycles through a chosen dimension would be a nice UX improvement.

12. **Zoom-to-ROI** (from arrShow)
    Draw a rectangle, zoom to fit exactly that region. More precise than scroll-zoom. Could use the existing ROI mode infrastructure.

13. **Colorbar with W/L Visualization** (from Slicer)
    Show the current window/level range as highlighted region on the histogram/colorbar. arrayview already has histogram overlay on colorbar; adding W/L indicators (two lines or a shaded band) would give visual feedback during dynamic range adjustment.

14. **Nearest-Neighbor Interpolation Toggle** (from Slicer)
    When zoomed in, toggle between smooth (bilinear) and pixelated (nearest-neighbor) rendering. Useful when you want to see exact pixel boundaries vs smooth display. CSS `image-rendering: pixelated` makes this trivial in a browser.

15. **FOV Auto-Clamping** (from arrShow)
    Prevent panning beyond image bounds. Simple bounds check on the pan offset. Reduces user confusion when zoomed in.

### Lower Priority / Niche

16. **Phase Circle at Cursor** (from arrShow) -- visual phase indicator; niche but elegant for complex data
17. **Surface Plot from 2D Slice** (from arrShow) -- 3D intensity surface; feasible with Three.js but niche
18. **Dimension Permute/Reshape in Viewer** (from arrShow) -- useful for development/debugging
19. **NaN Transparency in Export** (from arrShow) -- nice for compositing exported images
20. **Segment Statistics** (from Slicer) -- min/max/mean/stdev within ROI; arrayview's ROI mode could show this in a tooltip or panel
