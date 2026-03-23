# TODO

- Single array, normale mode:
  - The default way to open the viewer is in the maximum canvas size just before compact mode would hit when pressing +. This should be for all modes by the way.
  - In compact mode, the eggs are sort of on top of the canvas. Stack the eggs vertically to the left of the canvas, aligned to the top. 
  - When zooming in s.t. the miniviewer appears, the miniviewer should appear in the top right insead of bottom right. It should be in the same location as the miniviewer with the 3 planes in multi-view mode. 
  - When the colormap previewer is open, h and l and left and right should switch colormap. When it dissapeared again, h and l and left and right should do what they normally do of course.
  - The font size for the x and y labels of the arrows indicatin x and y directions need to be larger.
  - border mode (b) is no longer working. message appears but no border is visible.
  - when showing histogram instead of colorbar (w), the viewer still regularly crashes without showing error messages or whatsoever. i need to close the viewer and open a new one to continue. it happens when i press d sometimes, or slide one of the vertical lines to chage vmin or vmax
  - when i use vector-field mode, for exampe uv run arrayview tests/data/phantom.npy --vectorfield tests/data/vfield.npy, then whe i scroll through one of thespatial dims, there is annoyig flickering of the vector fields. maybe it has to do with a commit from one or two days ago where it was tried to have a 80 or 9 ms fade between slices? i dunno 

- Multi-array:
  - when i open multi-view mode, it does not auto-stretch to square like in single array mode
  - in multi-view mode, when i press c to cange colorbar it shows the previewer but colormap is not changig
  - i also want the crosshair lines in the diff canvas in X mode

 
make a plan first and write it down. make each finished item a separate commit. no need for separate branches unless you work on things in parallel. in that case, do the merge (no merge commits pls, just rebase) afterwards. remember to use the skill that makes sure auto-open of simplebrowser tab keeps working and that things keep working over the vscode remote tunnel. it happened often already that you broke these things and i had to spend a lot of time to fix it. also make sure to update tests with new functionality







