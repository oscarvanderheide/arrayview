# Prompt for Agent: Fix ArrayView Tool Menu

You need to fix the tool menu layout and interaction polish. Do not spend time improving or replacing the icons yet. Keep the current icons as placeholders; they will be replaced later. However, every row, including unavailable rows, must still reserve and render the icon column so alignment stays consistent.

## Current problems to fix

The current menu looks visually unbalanced:

- Tool rows are not aligned in a clean grid.
- Icons, titles, descriptions, and shortcut keycaps do not line up consistently.
- Descriptions are too long and get awkwardly truncated.
- Unavailable tools have no icons, breaking the row rhythm.
- The shortcut keycaps feel detached from the rows.
- Inactive rows feel like floating text instead of selectable rows.
- The panel is too visually loose; spacing and hierarchy need tightening.

## Desired behavior

The `/` menu is a single stateful panel.

State machine:

```text
Closed
  /   -> ToolMenu

ToolMenu
  /   -> Closed
  Esc -> Closed
  click tool / tool shortcut -> ToolOptions(tool)

ToolOptions(tool)
  /   -> Closed
  Esc -> ToolMenu
  back button -> ToolMenu
  close button -> Closed
```

Rules:

- `/` always toggles visibility.
- `Esc` goes back one level.
- If already at the general tool menu, `Esc` closes the menu.
- Only one tool can be active at a time.
- Selecting a tool replaces the general tool list with that tool’s options in the same panel.
- Do not create a separate side-by-side options panel.
- Do not create a minimized active-tool HUD.
- Do not implement search.

## Panel layout

Use a floating dark translucent panel on the left, keeping the current visual identity:

- dark background
- subtle border
- yellow accent
- monospace/current font
- scientific/HUD-like style

Suggested panel CSS:

```css
.tool-panel {
  width: min(560px, calc(100vw - 32px));
  max-height: min(760px, calc(100vh - 48px));
  overflow: hidden;
  border-radius: 18px;
  border: 1px solid rgba(255, 255, 255, 0.1);
  background: rgba(18, 18, 18, 0.94);
  box-shadow: 0 20px 60px rgba(0, 0, 0, 0.45);
}
```

Inner padding:

```css
.tool-panel-inner {
  padding: 24px 26px 20px;
}
```

Header:

```css
.tool-panel-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  height: 36px;
  margin-bottom: 20px;
}

.tool-panel-title {
  color: var(--accent);
  font-size: 18px;
  font-weight: 700;
  letter-spacing: 0.22em;
  text-transform: uppercase;
}
```

Header buttons:

- General menu: title `TOOLS`, close button on right.
- Tool options: back button on left or near title, title as tool name, close button on right.

Use clear semantics:

- Back button / `Esc` = return to tool menu.
- Close button / `/` = close panel.

## Tool row layout

Every row must use the same 3-column grid:

```css
.tool-row {
  display: grid;
  grid-template-columns: 36px minmax(0, 1fr) 44px;
  column-gap: 12px;
  align-items: center;
  min-height: 68px;
  padding: 10px 12px;
  border-radius: 10px;
}
```

Columns:

1. Icon column: fixed 36px
2. Text column: flexible
3. Shortcut keycap column: fixed 44px

The icon must be vertically centered relative to the whole row.

The title and description are grouped in the middle column:

```css
.tool-title {
  font-size: 16px;
  font-weight: 650;
  line-height: 1.2;
  color: rgba(255, 255, 255, 0.86);
}

.tool-description {
  margin-top: 5px;
  font-size: 12px;
  line-height: 1.25;
  color: rgba(255, 255, 255, 0.38);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
```

Important:

- Do not let descriptions enter the shortcut column.
- Do not underline shortcut letters in labels.
- Keep all shortcuts as keycaps on the right.

Shortcut keycap:

```css
.shortcut-key {
  justify-self: end;
  width: 32px;
  height: 32px;
  border-radius: 8px;
  border: 1px solid rgba(255, 255, 255, 0.18);
  display: grid;
  place-items: center;
  color: rgba(255, 255, 255, 0.48);
  background: rgba(255, 255, 255, 0.035);
  font-size: 13px;
}
```

Icons:

```css
.tool-icon {
  width: 22px;
  height: 22px;
  color: rgba(255, 255, 255, 0.52);
  display: grid;
  place-items: center;
}

.tool-icon svg {
  width: 22px;
  height: 22px;
  stroke-width: 1.8;
}
```

Keep the current icon components for now. They are placeholders. The important part is that the icon slot is always present.

## Row states

Inactive available rows should be simple, not boxed heavily.

```css
.tool-row.available {
  cursor: pointer;
}

.tool-row.available:hover {
  background: rgba(255, 255, 255, 0.045);
}
```

Active/selected row:

```css
.tool-row.active {
  background: rgba(245, 190, 40, 0.08);
  outline: 1px solid rgba(245, 190, 40, 0.55);
}

.tool-row.active .tool-title,
.tool-row.active .tool-icon {
  color: var(--accent);
}

.tool-row.active .shortcut-key {
  border-color: rgba(245, 190, 40, 0.45);
  color: var(--accent);
}
```

Unavailable rows:

```css
.tool-row.unavailable {
  opacity: 0.42;
  cursor: default;
}

.tool-row.unavailable:hover {
  background: transparent;
}
```

Unavailable rows must still render:

- icon column
- title
- reason as description
- shortcut keycap

Do not omit icons for unavailable tools.

## Section layout

Use sections:

```text
AVAILABLE

[tool rows...]

CURRENTLY UNAVAILABLE  ˄/˅

[unavailable rows if expanded...]
```

Section label style:

```css
.section-label {
  margin: 18px 0 8px;
  font-size: 11px;
  letter-spacing: 0.22em;
  text-transform: uppercase;
  color: rgba(255, 255, 255, 0.36);
}
```

Add subtle separators only between groups, not after every row unless it helps readability.

Example:

```css
.tool-section {
  padding-top: 8px;
  border-top: 1px solid rgba(255, 255, 255, 0.07);
}
```

## Text content

Use shorter descriptions to avoid ugly truncation:

- Crop: `Crop the array and save the crop`
- Overlay: `Overlay segmentation masks or heatmaps`
- Quantitative MRI mode: `Use appropriate colormaps for T_1, T_2, etc`
- Region-of-interest analysis: `Draw regions and measure statistics.`
- Segmentation: `Create or edit masks interactively.`
- Difference: `Compare two loaded arrays.`
- Vector field: `Inspect displacement or flow fields.`

Unavailable reasons:

- Difference: `Requires two loaded arrays.`
- Vector field: `No vector field found in this dataset.`
- Quantitative MRI mode: `Requires quantitative metadata.`

## Ordering

Use this order:

```text
AVAILABLE
Segmentation
Crop
Region-of-interest analysis
Overlay
Quantitative MRI mode

CURRENTLY UNAVAILABLE
Difference
Vector field
```

If a normally unavailable tool becomes available, move it into the `AVAILABLE` section using the same row layout.

## Tool options panel

When selecting a tool, the same panel changes content.

Example header:

```text
←  SEGMENTATION                                      ×
```

Footer hint:

```text
Esc tools · / close
```

Do not show the tool list and tool options side by side.

## Acceptance criteria

- The menu uses one clean 3-column grid for every tool row.
- Available and unavailable rows have the same structure.
- Unavailable tools show icons.
- Shortcut keycaps line up vertically.
- Descriptions do not overlap or run under shortcuts.
- Current icons remain but are treated as placeholders.
- `/` toggles the menu closed/open.
- `Esc` backs out of tool options to the general tool menu.
- `Esc` closes the general tool menu.
- Selecting a tool replaces the tool menu with that tool’s options in the same panel.
- No search bar.
- No active tool HUD.
- No side-by-side options panel.
