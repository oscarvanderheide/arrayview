# Loading Screen with Fade-In Transition

## Problem
When opening pywebview native windows, the dimbar appears before the image pane and starts in the wrong position, jumping when the pane loads. This creates an unpolished first impression.

## Solution
Hide all content initially, show a centered pulsing logo with the array name during loading, then fade everything in once the first WebSocket frame arrives.

## Loading State (before first frame)
- Body gets class `av-loading`
- `#loading-overlay` displays a centered 64×64 copy of the logo SVG with the existing `av-logo-pulse` counter-clockwise animation, plus the array name below it
- Wrapper contents below `#array-name` (`#info`, `#canvas-wrap`, `#slim-cb-wrap`) have `opacity: 0` via the body class
- `#array-name` in the header stays hidden too (it would be redundant with the loading logo showing the name)

## Ready Transition (first frame arrives)
- `#loading-overlay` fades out (opacity 1→0, ~300ms ease)
- Body class `av-loading` removed, wrapper contents fade in (opacity 0→1, ~300ms ease)
- After fade completes, loading overlay gets `display: none`

## What doesn't change
- Initial pywebview spinner (covers server startup before HTML loads)
- Welcome screen (no sid) — skips loading screen, goes straight to plasma demo
- Error display in loading overlay — still works within the overlay

## Scope
- **CSS only in `_viewer.html`:**
  - `body.av-loading #info, body.av-loading #canvas-wrap, body.av-loading #slim-cb-wrap, body.av-loading #array-name` → `opacity: 0`
  - Transitions on those elements for the fade-in
  - `#loading-overlay` content styling (centered logo, array name text)
- **JS only in `_viewer.html`:**
  - On page load: add `av-loading` class to body, populate `#loading-overlay` with logo SVG clone + array name
  - On first frame: fade out overlay, remove `av-loading` class, after transition set overlay `display: none`
- **No Python changes**
