# TODO

- ~~dragging into simplebrowser from vscode file explorer doesnt work~~ (technical limitation)
- ~~when i run uv run arrayview in terminal (with or without passing array), it still opens in native window not vscode simplebrowser~~ (done)
- ~~i want to go back to situation where killing the last open window (be it native, vscode browser, whatever) kills the server. right now stuff keeps lingering for too long and it leads to confusing situations and difficulties with debugging~~ (done)
- ~~instead of shft + o, can i use cmd + o on mac and ctrl + o on linux/windows for open?~~ (done)
- ~~when i use the file picker and switch to compare or overlay mode, it doesnt work, i can select an array but i dont get them side-by-side.~~ (done)
- ~~allow me to specify a slice index number with number keys and then enter to go to that slice for the current active slice dim.~~ (done)
- ~~in qMRI mode, there can be 3 to 6 parameter maps. when i press q the first time, show them all. when i press q another time, only show T1, T2 and abs(PD) as three canvasses horizontally aligned.~~ (done)
- ~~when in qmri mode, q goes to "compact mode" with T1 and T2 and PD only as required in one of the above todo items. but then when i press q again it should quit qmri mode.~~ (done)
- ~~it still doesnt auto-open in vscode simplebrowser when i run from vscode terminal. please use your skills and this issue has been coming up several times and took many efforts to fix again. when you have the fix, write it the fuck down somewhere.~~ (done - see VSCODE_DETECTION.md)
- see overlay_misalignment.png, the colorbar below the overlay canvas is misaligned. there's also text there that should not be there at all.
- the arrows indicating (x,y), (y,z) and (x,z) directions should always be visible in multi-view mode
- look at welcome. its shit. the canvas needs to be like 50% height, not square, and the text to open with cmd+o or ctrl+o or drop should be below it in yellow. i'd like some lofi pixel art animation of some cozy forest or city as an array on the welcome screen but forget about it if thats too complicated.
- on the tunnel, when i do uvx --from git+https://github.com/oscarvanderheide/arrayview arrayview senserefscan_csm.npy, it says: Updated https://github.com/oscarvanderheide/arrayview (f9ddddb26fe6178aaf82a7e2aef2f7f51411992a) Built arrayview @ git+https://github.com/oscarvanderheide/arrayview@f9ddddb26fe6178aaf82a7e2aef
Installed 49 packages in 54ms
  VS Code Ports tab: right-click port 8000 → Port Visibility → Public
  Press Enter once done (or the viewer retries automatically)... %                                   

ForOscarTest/HV_3Dand2D/results [ master]
❯ 
no port opens and i get err_connection_refused and i did not press enter myself. Again, im quite pissed this is broken agian because it took a long time to get things running over the tunnel


each item a separate commit. no need for separate branches unless you work on things in parallel. in that case, do the merge (no merge commits pls, just rebase) afterwards. 