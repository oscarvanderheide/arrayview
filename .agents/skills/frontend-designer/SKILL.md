---
name: frontend-designer
description: Use when making styling or layout changes to `_viewer.html`. Keeps UI work aligned with arrayview's established visual language without dragging in release-only or cross-mode audit steps by default.
---

# ArrayView Frontend Design Skill

## Fast Path

Use this for real styling/layout work.
For behavior-only edits, code cleanup, or tiny copy changes with no design decision, do not expand beyond this file unless blocked.

## Design Direction

Minimal chrome. The array is the product.

- Controls stay dim or hidden until interaction
- Text is small and monospaced
- No decorative chrome
- All four themes must work

## Core Rules

- Use existing CSS vars only. Especially: `--surface`, `--border`, `--text`, `--muted`, `--active-dim`, `--overlay-bg`, `--radius`, `--radius-lg`.
- Never hardcode colors.
- Typography is always monospace:
  `'SF Mono', ui-monospace, 'Cascadia Code', 'JetBrains Mono', monospace`
- Match the existing size scale. Most UI text is `10px` to `13px`; headers are `15px` to `16px`.
- Keep inactive UI dimmed with color/opacity, not permanent visibility.
- Prefer opacity/color transitions over motion that changes layout.
- Canvas dimensions are owned by JS layout functions, not CSS.
- Test themes with `T`.

## Defaults

- Passive info: `var(--muted)`
- Active text: `var(--text)`
- Selected/highlighted state: `var(--active-dim)`
- Panel/overlay: `var(--surface)` + `1px solid var(--border)` + `var(--radius-lg)`
- Modal backdrop: `var(--overlay-bg)` with the existing blur treatment

## Avoid

- Persistent toolbar-style controls
- New font families
- New one-off sizes or accent colors
- Layout animations that fight canvas sizing

## Before Shipping

- [ ] All four themes still look correct
- [ ] No hardcoded color values
- [ ] Font and sizing match existing UI
- [ ] New panels/overlays use existing surface/border/radius tokens
- [ ] `viewer-ui-checklist` is used only if the user asked for docs/help/test sync or this is release prep
- [ ] `modes-consistency` is used if canvas/colorbar/layout behavior crosses modes
