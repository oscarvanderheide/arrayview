# TODO

# done (2026-03-31, batch 2)
- [x] FFT loading animation (spinning 2x2 blocks, 150ms delay)
- [x] Remove compare mode strip on X key (title already shows mode)
- [x] Configurable nnInteractive server (env var + config.toml)
- [x] nnInteractive docs in README
- [x] Keybind remap: R=ROI, A=alpha, w=RGB; M freed
- [x] 'o' key resets dragged UI in immersive mode
- [x] Island drag position reset on immersive exit
- [x] Oblique slice performance (draft quality during drag)
- [x] Vector fields in oblique slices (projected onto plane)

# done (2026-03-31)
- [x] Add a "SUM" mode to keybind p which just sums.
- [x] Minimap northeast fallback in immersive view
- [x] 3-view crosshair fixed width and less transparency
- [x] Draggable dimbar and colorbar in immersive view (snap-back on exit)
- [x] Pane centering in native window mode
- [x] ROI island stays visible after freehand draw
- [x] ROI CSV export fix (session.data not session.array)
- [x] ROI island initial position near top-left of canvas

When working through the TODO list, always:
- Make a plan first and write it down.
- Make each finished item a separate commit.
- No need for separate branches unless you work on things in parallel. In that case, do the merge (no
merge commits pls, just rebase) afterwards.
- Spawn subagents where appropriate.
- clear/compact context
- Remember to use the skill that makes sure auto-open of simplebrowser tab keeps working and
  that things keep working over the vscode remote tunnel. It happened often already that you
  broke these things and i had to spend a lot of time to fix it.
- Make sure to update tests with new functionality
- Update README too
- Use and update skills where appropriate, especially the ui-consistentency-audit
- Remember I like a minimal approach: the app should be feature rich but not cluttered.
  Users should feel encouraged to explore all the cool things this app can do. They should
  think to themselves: "It would be great if it could do .... wait ... WHAT ... WHAT THE FUCK ITS
  ALREADY THERE HOLY SHIT THIS IS AWESOME AND WOW ALSO THIS OTHER THING I NEVER REALIZED I WANTED"
- use ui-consistency-audit skill to check behaviour across modes
- update dev/lessons_learned.md with things that may be important in future sessions
- when i say im going to sleep, dont ask me for confirmation, just make your own decisions
  regarding todo items. when i wake up i expect to be impressed.


<!-- 
- make the number of colormaps shown in the previewer (c) depend on how many fit within the
  dynamic island. right now its fixed so when the island is wide it doesnt use all the real
  esttae and when its too small it falls outside of the island -->