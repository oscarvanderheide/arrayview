# VS Code Window Routing Plan

## Problem

ArrayView can open the viewer tab in the wrong VS Code window when multiple
local windows and multiple remote/tunnel windows are active. The unsafe cases
are the ones where Python cannot prove the originating window and falls back to
writing multiple signal files or a shared signal file.

## Contract

- An exact `ARRAYVIEW_WINDOW_ID` wins when its registration exists.
- A missing or stale `ARRAYVIEW_WINDOW_ID` with multiple active registrations
  fails closed with a diagnostic.
- PID ancestry and IPC hook matching are allowed only when they identify a
  single window.
- tmux client fallback is allowed only when all attached clients for the current
  tmux session report the same `ARRAYVIEW_WINDOW_ID`.
- File-explorer opens launched by the extension keep using the extension's own
  `ARRAYVIEW_WINDOW_ID`.

## Implementation Slice

- Tighten `src/arrayview/_vscode_signal.py` window selection.
- Add focused lifecycle tests for local ambiguity, stale exact IDs, exact-ID
  ownership, and tmux multi-client ambiguity.
- Avoid changing `_app.py` or the viewer frontend.

## Manual Verification Targets

- Local VS Code terminal, one window: opens in that window.
- Local VS Code terminal, several windows: stale/missing exact ID prints a
  diagnostic instead of opening elsewhere.
- Remote/tunnel terminal, several windows: exact ID opens the matching window;
  stale/missing ID fails closed.
- File explorer custom editor: spawned Python receives the extension window ID
  and opens back into the same window.
