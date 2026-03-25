# TODO

(empty — all items completed 2026-03-25)

<!-- DONE 2026-03-25:
- [x] colorbar dynamic island misalignment with borders+zoom — fixed centering with translateX(-50%)
      and hidden vmin/vmax spans in compact overlay mode
- [x] dynamic island colorbar in all modes — applied glassmorphism to multi-view, compare, diff,
      qMRI colorbars via shared .cb-island class with theme variants
- [x] screenshot (s) broken — fixed download trigger by appending link to DOM before .click()
      (also fixed gallery export and GIF save)
- [x] CSV export in ROI mode broken — same DOM-append fix, changed MIME to application/octet-stream
- [x] ROI trash icons — added × delete button per ROI row, wired delegated click handler
- [x] ROI right-click blue screen — skip scrub/pan drag in ROI mode, stopImmediatePropagation,
      user-select:none on canvas-viewport
- [x] ROI mode (A) misalignment — verified clean layout via ui-consistency-audit
- [x] projection mode dim bar shows "p" — like qMRI shows "q", colored per projection type
- [x] removed "off" from projection cycle — same pattern as ROI (A): last press deactivates
-->


<!-- DONE 2026-03-24:
- [x] mode switch crossfade — pressing v/q/K/F fades out, reconfigures, fades back in (~300ms)
  instead of jarring element jumps
-->

<!-- DONE 2026-03-24:
- [x] opening arrayview for the first time takes several seconds — fixed import regression,
  `import arrayview` now ~138ms (was ~343ms). Remaining latency is browser/webview cold start.
- [x] research 3D Slicer and arrShow — see docs/competitor-features.md
- [x] show histogram when pressing d in multi-view — drawMvColorbar now supports histogram
- [x] colormap previewer gone in multiview — fixed anchor to mv-cb-wrap
- [x] zoom broken / miniviewer not appearing — removed aggressive zoom cap on +/= key
- [x] make r rotate x and y in multiview mode — now rotates the active pane 90° CW
- [x] vector-field mode slow with many vectors — client-side cache, server arrow cap at 4096
- [x] colorbar preview redesign — fade out colorbar, larger thumbs, fade back in after 1.5s
-->

<!-- - think as a modern UI designer and how the colorbar and dim bar can be positioned nicely to
  make more space for the array. compact mode needs a rethink — maybe overlay colorbar on
  canvas and dim bar too, with semi-transparent background.
- review borrowable features from docs/competitor-features.md and pick the best ones to
  implement (layer reveal cursor, rock/flicker compare, complex part chooser, etc.) -->

<!-- - current value in hover mode flickers a lot
- the dim bar that was centered above the canvas is aligned all the way to the left wtf did
  you do
- get rid of the white indicator in the colorbar / histogram that shows value of currently
  selected voxel.
- b to toggle boxes around canvas works in multiview but not normal mode wtf
- by default, window opens with maximum canvas size that does not push other elements of
  screen. when pressing =, it should go to compact mode and be at maximum allows size before
  canvas would get outside of window. then it should zoom in while keeping canvas same size
  and showing only part of the array with the miniviewer. right now 
- im not sure yet about compact mode (K). Maybe just show the colorbar on top of the canvas
  and the dim bar too. make sure they have like small blackground thats align with the theme.
- i want to be able to click-drag in that miniviewer. right now its only clicking thats
  working.
- when pressing c, the colorbar gets dimmed and a preview for colormaps appears. dont dim
  the colorbar. instead, change colorbar with new colormap and show previews of up to two
  items on both sides of the colorbar. Same when histogram is active btw -->

<!-- 
- When i press d to change vmin and vmax, the app regularly crashes. Also when I try to
  manually change vmin or vmax by dragging the line in the histogram that appears. 
- When scrolling from slice 99 to 100, there's a subtle but annoying displacement of UI
  components that needs to be fixed.
- The mini-viewer that appears when zooming in beyond max canvas size should dissapear when
  going to multi-view mode. Now it stays there.
- The font sizes for vmin vmax, the (x,y) labels near the arrows in the bottom left corner
  of a canvas that appear when changing dims with h/l or left/right, and the numbers of the
  ROIs (keybind A for ROI mode) are too small. The positioning of the ROI numbers is also
  weird and random. Sometimes inside, sometimes outside the ROI. Just use the center to
  place the number.
- When hovering over a ROI, just show mean +- std (and use unicode to get + on top of -).
  When hover info is on (H), make sure it doesn't clash, think like a true modern UI
  designer here.
- Hover info mode (H) should only show value at the cursor: the x,y values should be shown
  at the arrows in bottom left corner that normally appear when changing dims. In hover mode
  I want those arrows shown permanently instead of fading away like normal. Make them
  persist across mode changes (I think hover should work in all modes). -->


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









