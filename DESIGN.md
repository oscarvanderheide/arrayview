# ArrayView Design

> This is what looking at arrays should feel like.

---

## Principles

**Array fills the screen.**
The data is the product. Chrome is either invisible, dimmed, or hidden until the user needs it. When in doubt, remove UI.

**Reveal, don't accumulate.**
New features should reuse existing surfaces rather than add new ones. The colorbar flips to reveal save options. The dimbar edge flashes to show playback speed. The same key that enters a mode exits it. One surface, multiple jobs.

**Joy through restraint.**
One typeface (monospace). One accent color (yellow) for active state. Four themes, same structure. The consistency itself is the delight — the user never has to re-learn the visual language when switching modes.

**Modes are views, not apps.**
Normal, multiview, compare, diff, registration, qMRI, and MIP are different *lenses* on the same data. The UI adapts but never reinvents. A reconciler ensures what is true in one mode does not break another.

---

## Visual Language

| Token | Usage |
|-------|-------|
| `--bg` | Page background |
| `--surface` / `--surface-2` | Panels, overlays, islands |
| `--text` | Primary text |
| `--muted` | Passive info, dimmed chrome |
| `--active-dim` (yellow) | Active element, selection, focus |
| `--border` / `--border-subtle` | Panel edges, separators |
| `--radius-lg` (14px) | All panels, islands, popovers |

**Rules:**
- Monospace only. No exceptions.
- Never hardcode hex values — always use CSS custom properties.
- Backdrop blur on floating panels (`backdrop-filter: blur(16px)`).
- No decorative shadows, gradients, or icons without function.

See `.agents/skills/frontend-designer/SKILL.md` for the full theme system and CSS constraints.

---

## Interaction Model

**Progressive disclosure.**
Controls start hidden or dimmed. Hover or keypress brings them forward. They fade back out when idle. The array never competes with its own UI.

**Keyboard-first.**
Every feature has a shortcut. The help overlay (`?`) renders directly from the command registry — no undocumented keys. Adding a feature means adding a keybind and a help row.

**Immediate feedback.**
Parameter changes show a transient visual cue instead of opening a dedicated widget. Playback speed flashes the dimbar edge. A mode switch updates the canvas within one frame.

---

## Smart Reuse Patterns

These are intentional, not incidental. New features should extend this vocabulary rather than invent new surfaces.

### 1. The Flip
A 3-D CSS rotateX animation reveals a secondary control plane on the back face of an existing UI element.

- **Oblique mode:** In ortho multiview, the colorbar flips 180° on `Alt` (hold) to expose 90° Lock, Save planes, and Load planes.
- **Save menu:** In single view, the colorbar flips on `s` to expose Screenshot, Animated GIF, Save full array, and Copy shareable URL.

Same timing curve (`400ms cubic-bezier`), same structure, different content.

### 2. The Dimbar Fill
The `#info` pill (dimbar) uses a clipped border overlay to transiently indicate a parameter value.

- **Playback speed:** A left-to-right orange fill proportional to FPS.
- **Vector density:** A blue fill for field density.
- **Vector length:** A green fill for arrow length.

The fill lingers for 1s, fades over 1.4s, then resets silently. No progress bar widget needed.

### 3. The Egg
A composable pill badge below the canvas indicating an active transform (FFT, LOG, MAGNITUDE, PHASE, RGB, ALPHA, PROJECTION). Eggs stack horizontally and auto-hide when they would overlap the histogram. They are read-only status, not controls.

### 4. The Dynamic Island
A floating panel that appears contextually and disappears when no longer relevant. Used for ROI stats, segmentation controls, tool menus, and dimension sliders. Same surface tokens as all other panels (`--surface`, `--border`, `--radius-lg`).

---

## Modes

ArrayView has six core modes. Every visual feature must work across all of them.

| Mode | Entry | What it does |
|------|-------|--------------|
| Normal | — | Single canvas, full array |
| Multiview | `v` | Ortho slices + oblique |
| Compare | `B` / `P` | Side-by-side arrays |
| Diff | `X` (in compare) | Difference overlay |
| Registration | `R` (in compare) | Alignment overlay |
| qMRI | `q` | Quantitative maps |
| MIP | `p` (in multiview) | WebGL maximum intensity projection |

The reconciler architecture (`_reconcileUI`, `_reconcileLayout`, `_reconcileCompareState`, `_reconcileCbVisibility`) ensures UI visibility converges through a single path. No ad-hoc `style.display` toggles.

See `.agents/skills/modes-consistency/SKILL.md` and `.agents/skills/ui-consistency-audit/SKILL.md` for the 35-rule cross-mode compliance catalog.

---

## When to Break the Rules

Break a rule only when the alternative is significantly more confusing. Document the exception here.

*(None yet.)*

---

## Related Documents

- `CONTRIBUTING.md` — Code conventions, shortcut rules, popup styling standards
- `.agents/skills/frontend-designer/SKILL.md` — Theme system, CSS constraints
- `.agents/skills/modes-consistency/SKILL.md` — Cross-mode visual rules
- `.agents/skills/ui-consistency-audit/SKILL.md` — 35-rule compliance catalog
- `.mex/context/decisions.md` — Architectural decisions (single-file frontend, reconcilers, etc.)
