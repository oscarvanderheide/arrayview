# ArrayView Tool Menu Redesign — Implementation Plan

## Goal

Redesign the `/` tool menu into a **single-panel, single-active-tool** interface.

The menu has two visible states:

1. **General tool menu** — shows available tools.
2. **Selected tool options** — the same panel, but with contents replaced by options for the selected tool.

Do **not** implement:

- a separate side-by-side options panel
- an automatic active-tool HUD
- multiple active tools
- a search bar, for now
- draggable/minimized panels, for now

---

## Core interaction model

The menu should behave as a small state machine:

```text
Closed
  /   -> ToolMenu

ToolMenu
  /   -> Closed
  Esc -> Closed
  select tool -> ToolOptions(tool)

ToolOptions(tool)
  /   -> Closed
  Esc -> ToolMenu
  back button -> ToolMenu
  close button -> Closed
```

Important distinction:

- `/` means **toggle menu visibility**.
- `Esc` means **go back one level**.
- If already at the top-level tool menu, `Esc` closes the menu.

---

## UI state 1: Closed

No large menu is visible.

The image/data view remains dominant.

Optional: keep the existing small `/` hint button in the bottom-right corner.

```text
[ / ]  [?]
```

Pressing `/` opens the general tool menu.

---

## UI state 2: General tool menu

A single floating panel, matching the current dark HUD-like visual style.

Suggested dimensions:

```text
width: 320–380 px
height: auto
max-height: ~70vh
position: left-side overlay
```

Example layout:

```text
TOOLS                                      ×
────────────────────────────────────────────

AVAILABLE

[icon] Segmentation                     S
       Create and edit segmentations.

[icon] Crop                             C
       Crop or pad the field of view.

[icon] Overlay                          O
       Add overlays and annotations.

[icon] Quantitative MRI mode            Q
       Switch to quantitative workflows.

[icon] Region-of-interest analysis      R
       Measure intensity in regions.

MORE TOOLS                              ˅

────────────────────────────────────────────
/ close · Esc close
```

### General menu design rules

- Show only currently available tools by default.
- Do not show a search bar yet.
- Use right-aligned shortcut keycaps, not underlined shortcut letters.
- Use muted one-line descriptions below each tool label.
- Avoid heavy bordered “islands” around every inactive row.
- Use subtle dividers, spacing, or hover states instead.
- If a tool is currently active/last-selected, highlight it with the app’s yellow accent.

---

## UI state 3: Selected tool options

When a tool is selected, the same panel transforms into that tool’s options.

Example for segmentation:

```text
←  SEGMENTATION                         ×
────────────────────────────────────────────

METHOD

[ Threshold ]  [ Region grow ]  [ Random walker ]  [ Paint ]

PARAMETERS

Threshold
━━━━━━━━━━●━━━━━━              0.499

Smoothing σ
━━━━━━●━━━━━━━━━              1.0

Min. region size px
━━━━●━━━━━━━━━━━              100

Connectivity
[ 4 ] [ 8 ] [ 26 ]

Fill holes                         on
Preview overlay                    on

Overlay color                      [■]
Opacity
━━━━━━●━━━━━━━━━              0.45

ACTIONS

[ Preview ] [ Reset ] [ Clear ] [ Save… ]

────────────────────────────────────────────
Esc tools · / close
```

### Header behavior

Use clear, distinct meanings:

```text
←  SEGMENTATION                         ×
```

- `←` or `Esc` returns to the general tool menu.
- `×` or `/` closes the panel entirely.

Do not show a separate floating active-tool HUD.

---

## Keyboard behavior

### Global handling

Only handle `/` and `Esc` when appropriate. Avoid stealing input from text fields later.

```ts
function onKeyDown(event: KeyboardEvent) {
  if (event.key === "/" && !isTypingInInput(event.target)) {
    event.preventDefault()
    toggleToolPanelVisibility()
    return
  }

  if (event.key === "Escape") {
    event.preventDefault()
    goBackOrCloseToolPanel()
    return
  }
}
```

### `/` behavior

```ts
function toggleToolPanelVisibility() {
  if (toolPanelState.kind === "closed") {
    openToolMenu()
  } else {
    closeToolPanel()
  }
}
```

### `Esc` behavior

```ts
function goBackOrCloseToolPanel() {
  switch (toolPanelState.kind) {
    case "closed":
      return

    case "tool-menu":
      closeToolPanel()
      return

    case "tool-options":
      openToolMenu()
      return
  }
}
```

### Tool shortcuts

When the general tool menu is open:

```text
S -> Segmentation
C -> Crop
O -> Overlay
Q -> Quantitative MRI mode
R -> ROI analysis
```

When a tool options panel is open, shortcuts can be tool-specific:

```text
Segmentation:
T -> Threshold
G -> Region grow
W -> Random walker
P -> Paint
```

Tool-specific shortcuts should only be active while that tool’s options panel is open.

---

## Suggested state model

Use a discriminated union or enum-like state.

```ts
type ToolPanelState =
  | { kind: "closed" }
  | { kind: "tool-menu" }
  | { kind: "tool-options"; toolId: ToolId }

type ToolId =
  | "segmentation"
  | "crop"
  | "overlay"
  | "quantitative_mri"
  | "roi"
```

Keep the active tool separate from menu visibility:

```ts
type ActiveToolState = {
  toolId: ToolId | null
  options: Record<ToolId, unknown>
}
```

Reason: closing the menu should not necessarily deactivate the tool. It should only hide the panel.

Recommended behavior:

- Selecting `Segmentation` makes segmentation the active tool.
- Closing the panel with `/` leaves segmentation active.
- Pressing `/` from closed should open the general tool menu, not jump directly back to previous tool options.
- The active tool can be highlighted in the general menu.

---

## Tool availability

By default, only show available tools.

Each tool should expose availability:

```ts
type ToolAvailability =
  | { available: true }
  | { available: false; reason: string }

type ToolDefinition = {
  id: ToolId
  label: string
  description: string
  shortcut: string
  icon: IconName
  availability: ToolAvailability
}
```

### General menu behavior

- Render available tools in the main list.
- Render unavailable tools only under `More tools`.
- Unavailable tools cannot be activated.

Collapsed:

```text
MORE TOOLS                              ˅
```

Expanded:

```text
MORE TOOLS                              ˄

[lock] Quantitative MRI mode            Q
       Requires quantitative data.

[lock] Vector field                     V
       No vector field found in this dataset.

[lock] Compare center                   M
       Requires at least two loaded arrays.
```

Unavailable tool rows should be:

- muted
- non-clickable
- marked with lock/info icon
- optionally tooltip-enabled
- hidden by default

---

## Visual style

Keep the current visual identity:

- dark translucent panel
- subtle blur/glass effect if already used
- yellow accent for selected/active state
- monospace or current app font
- compact scientific/HUD feel
- image remains visually dominant

Avoid:

- search bar for now
- underlined shortcut letters
- large bordered islands around inactive tools
- showing unavailable tools in the main list
- separate tool options panel
- multi-tool HUD stack

---

## Suggested component structure

```text
ToolPanel
  ToolMenu
    ToolRow
    MoreToolsSection
  ToolOptionsPanel
    ToolOptionsHeader
    SegmentationOptions
    CropOptions
    OverlayOptions
    RoiOptions
    QuantitativeMriOptions
```

Example top-level API:

```tsx
<ToolPanel
  state={toolPanelState}
  tools={tools}
  activeToolId={activeToolId}
  onOpenToolMenu={openToolMenu}
  onClose={closeToolPanel}
  onBack={openToolMenu}
  onSelectTool={selectTool}
/>
```

Tool selection logic:

```ts
function selectTool(toolId: ToolId) {
  if (!isToolAvailable(toolId)) return

  setActiveToolId(toolId)
  setToolPanelState({ kind: "tool-options", toolId })
}
```

---

## Tool-specific options

### Segmentation

Start with:

```text
METHOD
Threshold / Region grow / Random walker / Paint

PARAMETERS
Threshold
Smoothing σ
Min. region size
Connectivity: 4 / 8 / 26
Fill holes
Preview overlay
Overlay opacity

ACTIONS
Preview / Reset / Clear / Save…
```

Behavior:

- Changing method changes interaction behavior in the slice pane.
- Threshold uses threshold-driven segmentation.
- Region grow / random walker use seed-based interaction.
- Paint uses brush interaction.
- Preview overlay toggles live visualization.
- Save writes segmentation to disk.

### Crop

Example:

```text
MODE
Rectangle / Square / Circle

OPTIONS
Keep aspect ratio
Pad with: edge value / zero / NaN / custom
Margins: left / right / top / bottom

ACTIONS
Apply / Reset
```

### ROI analysis

Example:

```text
SHAPE
Circle / Rectangle / Freehand / Flood-fill

MEASUREMENT
Mean / Median / Std / Min / Max

ACTIONS
Add ROI / Delete ROI / Export
```

Use **Flood-fill** in the UI unless there is a specific reason to call it “landfill”.

---

## Slice pane integration

When `activeToolId` changes, the slice pane should switch interaction mode.

```ts
switch (activeToolId) {
  case "segmentation":
    slicePaneInteraction = "segmentation"
    break

  case "crop":
    slicePaneInteraction = "crop"
    break

  case "roi":
    slicePaneInteraction = "roi"
    break

  default:
    slicePaneInteraction = "view"
}
```

Closing the panel does **not** necessarily reset the interaction mode.

The selected tool remains active until another tool is selected or the user explicitly disables it.

Add an explicit `Deactivate tool` / `Exit tool` action later only if needed.

---

## Acceptance criteria

The implementation is done when:

1. Pressing `/` from image view opens the general tool menu.
2. Pressing `/` while any menu/panel is open closes it.
3. Pressing `Esc` from the general tool menu closes it.
4. Pressing `Esc` from selected tool options returns to the general tool menu.
5. Selecting a tool replaces the tool list with that tool’s options in the same panel.
6. No second side-by-side options panel appears.
7. No active tool HUD appears.
8. Only available tools are shown by default.
9. Unavailable tools appear only inside expanded `More tools`, with reasons.
10. Shortcut keycaps appear on the right side of tool rows.
11. Inactive tools do not have heavy bordered islands.
12. Active/selected tool is visibly highlighted when returning to the tool list.
13. The image remains visually dominant and the panel acts as an overlay.

---

## Minimal implementation order

1. Implement `ToolPanelState`.
2. Implement `/` and `Esc` behavior.
3. Build the general `ToolMenu`.
4. Build `ToolOptionsPanel` with only Segmentation.
5. Add Crop.
6. Add availability filtering and `More tools`.
7. Polish styling.
8. Add remaining tools.

Do not implement search, multi-tool activation, draggable panels, or minimized HUDs yet.
