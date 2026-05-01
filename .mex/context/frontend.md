---
name: frontend
description: Deeper _viewer.html map for cross-section frontend work. Do not load for small localized follow-up fixes with a known symbol.
triggers:
  - "_viewer.html"
  - "reconciler"
  - "command registry"
  - "GUIDE_TABS"
  - "modeManager"
  - "ColorBar class"
  - "LayoutStrategy"
edges:
  - target: context/architecture.md
    condition: when understanding how the frontend connects to the server
  - target: context/decisions.md
    condition: when understanding why the frontend is a single file or why reconcilers exist
  - target: context/conventions.md
    condition: when writing new frontend code and need section separator conventions
  - target: patterns/frontend-change.md
    condition: when making a concrete change to _viewer.html
last_updated: 2026-05-01
---

# Frontend (_viewer.html)

Load this only when the task crosses sections or when the relevant symbol is not already known.
For a small follow-up tweak, prefer exact code search in `_viewer.html` and skip this file.

`_viewer.html` is a single-file frontend with inline CSS and JS. No build step.

## Load This When

- The change touches reconcilers, mode transitions, `modeManager`, or layout auto-pickers
- A keybind/command change needs the registry + `GUIDE_TABS` rules
- The task spans multiple frontend subsystems and local code reads are no longer enough
- You need the high-level map before editing an unfamiliar area

## Skip This When

- The user is asking for ideas, review, or a tiny follow-up fix in one known function/section
- The target id/function/command is already known and can be reached with one exact `rg`
- The task is local styling or text copy near an already-known DOM node

## Quick Anchors

- Section separators: `/* ── Section Name ── */` in CSS, `// ── Section Name ──` in JS
- Reconcilers: grep `UI Validation and Reconciliation`, `_reconcileUI`, `_reconcileLayout`
- Keybinds/help: grep `commands`, `keybinds`, `GUIDE_TABS`, `dispatchCommand`
- Mode system: grep `Mode Registry`, `modeManager`, `enterMultiView`, `enterCompare`, `enterQmri`
- Layout/scale: grep `scaleCanvas`, `mvScaleAllCanvases`, `compareScaleCanvases`, `qvScaleAllCanvases`
- Colorbars: grep `ColorBar class`, `drawSlimColorbar`, `drawMvColorbar`

## Rules That Matter

- Visibility changes belong in reconcilers, not scattered `style.display` or `classList` toggles
- Keybind changes must update both the command/keybind registry and `GUIDE_TABS`
- Reuse shared layout helpers for viewport/layout decisions; do not add ad hoc per-mode heuristics
- `ColorBar` usage is mixed legacy/new: stay consistent within the section you touch
- Canvas dimensions should be owned by the mode scale/layout functions, not ad hoc writes

## Mode Map

| Area | Main owner |
|------|------------|
| Normal / immersive / compact | `scaleCanvas()` |
| Multiview / ortho | `mvScaleAllCanvases()` |
| Compare / diff / registration | `compareScaleCanvases()` |
| Compare + MV | `compareMvScaleAllCanvases()` |
| qMRI / qMRI mosaic | `qvScaleAllCanvases()` |
| MIP | dedicated WebGL path |

## Important Concepts

- **Reconcilers** — UI visibility/state should converge here after mode changes
- **Command registry** — `commands`, `keybinds`, `dispatchCommand`, `GUIDE_TABS`
- **View component system** — `View`, `Slicer`, `Layer`, `LayoutStrategy`, `modeManager`; still coexists with legacy rendering
- **Dual-write state** — when legacy globals are changed, matching `displayState` fields may need updating too
- **Manual range state** — `manualVmin` / `manualVmax` and per-view locked ranges are real state, not derived UI
- **Tool menu and dynamic island** — central owners for several multi-feature UI surfaces; touch deliberately

## Verification Reminder

If a change touches multiple modes, reconcilers, or layout routing, verify cross-mode behavior instead of trusting a single local interaction.
