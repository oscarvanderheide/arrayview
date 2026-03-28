# Exploded Volume View

## Overview

Press `E` and the current 2D slice view explodes into a 3D perspective fan of slices — like a deck of cards spreading in space. The stack auto-rotates slowly on a turntable. Click any slice to collapse back into normal view at that slice. `[` / `]` to adjust slice density.

## Entry Animation

1. Current slice scales down slightly (~80%) with easing
2. Neighbouring slices appear and fan outward in both directions along a Z-axis in perspective
3. Spring physics easing — slices overshoot slightly, then settle (cubic-bezier approximation)
4. Total animation: ~600-800ms
5. Once settled, slow turntable auto-rotation begins (~15°/sec around vertical axis)

## 3D Layout

- Slices arranged along a depth axis using CSS 3D transforms (`perspective` + `translateZ` + `rotateY`)
- Each slice is a flat plane with the rendered image as background (respects current colormap, dynamic range, etc.)
- Current slice highlighted with yellow border/glow
- Slice index labels on each card (small, non-intrusive)
- Adaptive density: show all slices if volume has <40 along scroll axis, evenly sample ~30 otherwise
- `[` / `]` adjust density with smooth insert/remove animation

## Interaction

- **Auto-rotation:** Slow continuous turntable (~15°/sec). Pauses on hover/interaction, resumes after 2s idle
- **Click a slice:** Spring-collapses back into normal view at that slice index (~500ms reverse animation)
- **Press `E` again:** Collapses back to whichever slice is nearest center
- **Scroll:** Shifts which slice is centered/highlighted, smooth sliding animation
- **`[` / `]`:** Decrease/increase visible slice count (density), animated insertion/removal
- **All other keys disabled** while exploded (except `E`/`Esc` to exit, `s` for screenshot)

## Rendering

- CSS 3D transforms on DOM elements — `transform-style: preserve-3d`, `perspective` on container
- Each slice is a `<div>` with slice image rendered via offscreen canvas → `toDataURL()` → `background-image`
- GPU compositing for smooth 60fps rotation via `requestAnimationFrame`
- Slices pre-rendered on entry from server-provided thumbnails

## Data Flow

1. User presses `E`
2. Frontend determines which slices to show (adaptive sampling based on scroll axis length)
3. Frontend sends `exploded_slices_request` via WebSocket with list of slice indices + target resolution
4. Server renders each slice using existing colormap/range logic, returns as base64 JPEG batch
5. Response: `{slices: [{index: N, image: "data:image/jpeg;base64,..."}, ...]}`
6. Frontend builds DOM elements with CSS 3D transforms, triggers spring entry animation
7. Auto-rotation runs via `requestAnimationFrame`
8. On exit (click slice or `E`/`Esc`), reverse spring animation → remove DOM → restore normal view

## Server-Side

- New WebSocket message type: `exploded_slices_request`
  - Payload: `{indices: [0, 8, 16, ...], width: 256}`
  - Server renders each slice at requested width (smaller than full canvas for performance)
  - Uses existing colormap/dynamic-range pipeline
- Response: `{type: "exploded_slices", slices: [{index: N, image: "data:image/jpeg;base64,..."}, ...]}`

## Visual Polish

- Subtle drop shadows on slices in 3D space
- Background dims slightly (like immersive mode entry) to make the 3D stack pop
- Slice nearest to viewer is slightly larger (scale 1.05)
- Smooth opacity fade on distant slices to prevent visual clutter at high density
- Spring easing on all transitions (entry, exit, density changes, scroll)

## Exit Animation

1. Auto-rotation stops
2. All slices spring-collapse toward the selected/center slice
3. Selected slice scales back up to full size
4. Background un-dims
5. Normal view restored at the selected slice index
6. Total: ~500ms

## Keyboard Summary

| Key | Action |
|-----|--------|
| `E` | Toggle exploded view on/off |
| `Esc` | Exit exploded view |
| `[` / `]` | Decrease / increase slice density |
| Scroll | Shift centered slice |
| Click slice | Exit and jump to that slice |
| `s` | Screenshot |

## Scope Boundaries

- No multi-axis explosion (always fans along current scroll axis)
- No drag-to-rotate (auto-rotation only)
- No slice editing while exploded
- No export of 3D view (screenshot captures as-is)
- No persistence — exploded view is transient
- Only works on 3D+ arrays (silently ignored on 2D)
