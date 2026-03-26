# TODO

- ~~in multi-view, only show the red/green/blue lines when hovering over the panes~~ ✅ DONE

- when pressing p, make the behaviour the same as when pressing m for now: just switch up
  the eggs only, no previewer. the eggs are now all purple, in the preview the options all
  have different colors. the p in the dimbar also uses those colors. give the p the same
  color as used for the egg so that its fixed for this mode.

- when pressing p, and then h or l to change active scroll dim, it changes the p mode to
  that new active scroll dim which is then no longer scrollable. fix this bug.

- ~~when pressing d, histogram auto-disappears when hovering~~ ✅ DONE — histogram stays open while mouse is over it

- the indicators in the histogram for vmin and vmax suck. use upward pointing triangles
  beneath the bins, they should be draggable. show the value at the arrows (vmin vmax) above
  the histogram. maybe like rotated 45 degrees? im sure you can think of some nice, minimal,
  non-intrusive yet informative way. the histograms are in a rectangle whose
  background color should be that of the dynamic island its in. right now thats not the case. 

- ~~get rid of w keymap, remap i→I, H→i~~ ✅ DONE

- ~~axes indicator stays in bottom-left during zoom~~ ✅ DONE

- ~~remove gray fading status text~~ ✅ DONE — showStatus() is now a no-op

- bring back compact mode. the default zoom for all modes i think is already to have the
  panes at maximum size before UI elements start overlapping. in compact mode - wait what a
  stupid name, i guess theres a better name - the canvas for single array normal mode should
  grow until its almost at a viewport boundary. for single array, the colorbar and dimbar
  island go within the canvas at the bottom and top respectively. the eggs go above the
  colorbar. the array name goes below the dimbar. for multiple array situations i dont know
  yet. for single-array but stuff like qmri mode, do the same behaviour as for normal mode.

- after pressing = another time when in compact mode, enter zoom mode where the miniviewer appears.

- for some edge cases like really thin arrays where the dynamic island does not fit within
  the array or would overlap with the miniviewer, maybe just not go into compact mode but
  zoom mode directly.similarlty for wide arrays where there is not enough vertical space to
  fit all the ui elements.

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
