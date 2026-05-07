# Unified Colormap Picker

## Goal

Replace the split colormap UX with one floating picker island:

- `c` opens the picker in browse mode
- repeated `c` cycles the browse selection
- `Enter` commits
- `Esc` restores the previous colormap and closes
- clicking or focusing the search field fades the browse swatches and enables fuzzy search across all Matplotlib colormaps
- `Shift+C` remains as a temporary compatibility alias that opens the same picker with search focused

## Design

- The picker is a separate floating island, not an extension of the colorbar host.
- The active colorbar host is still used as an anchor for placement.
- Browse mode shows the active configured colormap pool as tall vertical swatches.
- Search mode keeps the same island and reveals a fuzzy-search result list below the swatches.
- The old fullscreen `Shift+C` picker is removed as a separate surface.

## State Model

- `closed`
- `browse`
- `search-focus`
- `search-results`

The picker captures prior state on open, live-previews during browse or search navigation, commits on `Enter` or click, and restores on `Esc`.

## Implementation Steps

1. Reuse the existing mode-aware host resolution from the integrated picker, but only for anchor placement and mode-specific apply/restore behavior.
2. Convert the old `#cmap-picker` fullscreen modal into a compact floating island.
3. Add a swatch strip region for the active browse pool.
4. Merge the existing `Shift+C` fuzzy-search logic into the same picker controller.
5. Add browse/search state classes so the swatches dim when search becomes active.
6. Route both `c` and `Shift+C` into the same controller.
7. Preserve diff-center and custom-colormap preview behavior by extending the existing apply/restore helpers to support arbitrary colormap names.
8. Update help text and command descriptions to describe the unified picker.

## Verification

- Normal single-view anchor and commit/cancel
- Immersive anchor and overlap behavior
- Multiview shared-colorbar anchor
- Compare center / diff-center anchor and diff colormap pool
- Search focus, fuzzy results, keyboard navigation, and `Esc` restore
