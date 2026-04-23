---
name: frontend-designer
description: Use when making any styling or layout change to _viewer.html. Ensures new UI is visually consistent with the established design language — dark theme, monospace typography, yellow accents, and minimal chrome.
---

# ArrayView Frontend Design Skill

## Design Philosophy

Minimal chrome. Let arrays fill the screen. UI elements are dim until needed; the array is always the primary focus.

- Controls fade in on hover or keypress, not permanently visible
- Text is small and monospaced
- No decorative elements — every pixel either shows data or provides affordance
- All four themes must look correct; never hardcode colors

---

## Design Tokens (CSS Custom Properties)

All colors come from CSS variables defined on `:root`. Never use raw hex values in new code.

| Variable | Dark default | Purpose |
|----------|-------------|---------|
| `--bg` | `#0c0c0c` | Page/canvas background |
| `--surface` | `#141414` | Panel/overlay backgrounds |
| `--surface-2` | `#1c1c1c` | Input fields, nested surfaces |
| `--border` | `#2c2c2c` | Borders, dividers |
| `--text` | `#d8d8d8` | Primary text |
| `--muted` | `#5a5a5a` | Secondary/dim text, inactive labels |
| `--highlight` | `#fff` | Maximum contrast text |
| `--active-dim` | `#f5c842` | **Primary accent** — active state, keyboard keys, clim handles |
| `--spatial-dim` | `#b48ead` | Spatial (x/y) dimension labels |
| `--help-key` | `#f5c842` | Keyboard shortcut key labels in help overlay |
| `--overlay-bg` | `rgba(0,0,0,0.72)` | Modal backdrop |
| `--blur` | `blur(14px)` | Backdrop blur for overlays |
| `--radius` | `10px` | Standard border radius |
| `--radius-lg` | `14px` | Large panel border radius |

Four themes exist: `dark` (default), `.light`, `.solarized`, `.nord`. Each redefines all tokens. Test with `T` key.

---

## Typography

Single font stack throughout:
```css
font-family: 'SF Mono', ui-monospace, 'Cascadia Code', 'JetBrains Mono', monospace;
```

Sizes used in practice:
- `10px` — tooltips, micro labels
- `11px` — colorbar value labels, secondary info
- `12px` — dim labels, position info
- `13px` — array name, status bar, picker items
- `15px` — dim scrubber labels
- `16px` — mode/view headers

Never use a sans-serif font. Never use `font-weight: bold` except via the `.highlight` or `.active-dim` classes.

---

## Layout Principles

- `#wrapper`: flexbox column, centers content vertically and horizontally
- Canvas fills available space; UI chrome is positioned absolutely around it
- Bottom bar (`#info`): fixed height, monospace, dim text
- Overlays (`#help-overlay`, `#uni-picker`, `#inline-prompt`): centered modal with `--overlay-bg` backdrop
- Colorbar (`#slim-cb-wrap`): thin strip below canvas, expands only in Lebesgue mode

**Canvas sizing:** canvas width/height are set by JavaScript (`scaleCanvas` and friends), not CSS. Do not set canvas dimensions in CSS.

---

## Component Patterns

### Status/info text
```css
color: var(--muted); font-size: 12px;  /* passive info */
color: var(--text);  font-size: 12px;  /* active info */
color: var(--active-dim); font-weight: bold;  /* highlighted state */
```

### Panel/overlay
```css
background: var(--surface);
border: 1px solid var(--border);
border-radius: var(--radius-lg);
padding: 16px 20px;
```

### Modal backdrop
```css
background: var(--overlay-bg);
backdrop-filter: var(--blur);
-webkit-backdrop-filter: var(--blur);
```

### Keyboard key labels (in help overlay)
```html
<span class="highlight">X</span>
```
CSS: `color: var(--help-key); font-weight: bold;`

---

## Do / Don't

| Do | Don't |
|----|-------|
| Use `var(--active-dim)` for active/selected state | Hardcode `#f5c842` |
| Keep new controls hidden by default, shown on interaction | Add persistent toolbar buttons |
| Use `opacity` or `color: var(--muted)` for inactive state | Use `visibility: hidden` (breaks layout) |
| Use `transition: opacity 0.15s` for hover reveals | Animate position/size (janky on canvas) |
| Test all four themes with `T` key | Assume dark theme only |
| Match existing font sizes | Introduce new size values |

---

## Checklist Before Shipping a UI Change

- [ ] Tested in all four themes (dark, light, solarized, nord) with `T` key
- [ ] No hardcoded color values — all via `var(--...)`
- [ ] Font is monospace, size matches existing scale
- [ ] New panel/overlay uses `--surface` + `--border` + `--radius-lg`
- [ ] `viewer-ui-checklist` skill followed (smoke test updated)
- [ ] `modes-consistency` skill followed if canvas/colorbar is involved
