# TODO

- cycle X when having two arrays loaded. make screenshots. you will see many issues with
  colorbars and positioning of panes. fix them, then take screenshots to see if the issues
  are resolved. also eggs are in wrong positions. 

- ~~in X flicker mode, use [ and ] to change flickering rate~~ ✅ DONE

- pinch to zoom and ctrl/cmd scroll to zoom are both way too sensitive to the point where
  they are unusable. ~~ ✅ DONE
  
- the dim bar island and colorbar island should have the same height. make the colorbar
  island a bit thicker to match the dim bar height. font for the vmin and vmax can be
  slightly larger ~~ ✅ DONE

- ~~in jupyter inline mode, hide logo+array name~~ ✅ DONE — body.inline-embed #array-name hidden

- ~~in multi-view, only show the red/green/blue lines when hovering over the panes~~ ✅ DONE

- ~~P key matches M key egg-only behavior, fixed purple color~~ ✅ DONE
- ~~P mode independent of h/l scroll dim change~~ ✅ DONE — projectionDim locked on activation

- ~~when pressing d, histogram auto-disappears when hovering~~ ✅ DONE — histogram stays open while mouse is over it

- the indicators in the histogram for vmin and vmax suck. use upward pointing triangles
  beneath the bins, they should be draggable. show the value at the arrows (vmin vmax) above
  the histogram. maybe like rotated 45 degrees? im sure you can think of some nice, minimal,
  non-intrusive yet informative way. the histograms are in a rectangle whose
  background color should be that of the dynamic island its in. right now thats not the case. 

- ~~get rid of w keymap, remap i→I, H→i~~ ✅ DONE

- ~~axes indicator stays in bottom-left during zoom~~ ✅ DONE

- ~~remove gray fading status text~~ ✅ DONE — showStatus() is now a no-op

- ~~bring back compact/immersive mode~~ ✅ DONE — auto-enters on first zoom-in (=), exits on zoom-out to fit (0 or -)
- ~~zoom mode with miniviewer on further zoom~~ ✅ DONE — minimap appears when canvas exceeds viewport
- ~~skip immersive for thin/wide arrays~~ ✅ DONE — _canFitImmersive() checks min 200×150px

- ~~single array + multiview: two colorbars visible~~ ✅ DONE — shared colorbar hidden in multiview
- ~~multiple arrays, normal view: colorbar too wide~~ ✅ DONE — capped at 500px
- ~~multiple arrays, multiview: crosshair lines don't fade~~ ✅ DONE — fixed sid tracking in fade animation
- ~~multiple arrays, multiview: colorbar overlaps with panes~~ ✅ DONE — shared colorbar hidden in compare+multiview
- multiple arrays, multiview: very slow to load
- ~~multiple arrays, loading screen: weird egg-like UI element~~ ✅ DONE — eggs hidden until data loads

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
