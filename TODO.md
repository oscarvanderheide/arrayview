# TODO


- compare mode still broken. when i open it i get a third large colorbar in the ui. sometimes i get a wrong grid (see wrong_grid.png). I think i also want to "merge" the registration overlay (R) and swipe mode (W) into a unified thing with diff mode (X): They all do the same thing, i.e. giving a third canvas in the middle with some sort of way to compare the two arrays. how about something similar to the preview thingy for colormaps (c) but then for ways that the two arrays can be compared (A-B, |A-B|, |A-B|/|A|, overlay, wipe). make sure to remove non-used keybinds also from readme, help, etc.
- I typically use default black theme. In diff mode with A - B, it uses RdBl_r which has
  white in the middle. I'd rather have black in the middle. it should be possible, see
  https://stackoverflow.com/questions/65523844/colormap-diverging-from-black-instead-of-white.
  For |A-B|, and |A-B|/|A| I think i want colormap afmhot because it has black as 0. i guess
  i also want to be able to change colormaps for the center canvas when i hover over it and
  press c. otherwise c changes colormaps of the two compared arrays.
- in compare mode it shows the name of the first arrow above the dim bar right next to the logo. that doesnt make sense. show the array name above each canvas. at the top next to the logo just say comparing {2,3,4} arrays
- the "log" in the histogram is great but it should be written vertically because now it extends beyond the bar it applies to.
- when i switch colormap with c i get the nice preview window so no need for the fading text
- click-dragging results in a square ROI. dont do that. i want to make a separate roi mode (dont know a good keybind) where i can draw multiple rois and export their statistics (mean std min max num_voxels) to csv
- right now i get the minimap when zooming to 105%. i like it but its not useful at the moment. the reason is that when i zoom, the canvas increases in size and when its about to push other ui elements out of the window, further zooming is no longer allowed. what i want is to then continue to zoom but without increasing the canvas, resulting in a small array patch being visible. then in the minimap i should see with the yellow square which part of the image is shown in (zoomed) in the canvas.
- the Alt+hover thing is nice but feels kinda useless

- i have some ui elements (dim bar, logo, name) eabove the canvas in normal mode and some ui elements below (colorbar, eggs). when the screen is little and the window too, this kinda limits zoom too much (i.e. increasing the canvas size). how about being able to move those elements elsewhre. colorbar could be vertical on the side. the other elements i dont have a clear idea. i also dont know if the user experience would be best if this is something the user can toggle, or if it happens automatically when the canvas is restricted from growing further. 
- when scrolling through some dimension of say size 256, the width of the total dimba changes when i go from 9 to 10 and from 99 to 100. there should be padding s.t. everything stays in place when scrolling. 

make a plan first and write it down. make each finished item a separate commit. no need for separate branches unless you work on things in parallel. in that case, do the merge (no merge commits pls, just rebase) afterwards. remember to use the skill that makes sure auto-open of simplebrowser tab keeps working and that things keep working over the vscode remote tunnel. it happened often already that you broke these things and i had to spend a lot of time to fix it. also make sure to update tests with new functionality




