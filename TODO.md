# TODO

- ~~dragging into simplebrowser from vscode file explorer doesnt work~~ (technical limitation)
- ~~when i run uv run arrayview in terminal (with or without passing array), it still opens in native window not vscode simplebrowser~~ (done)
- ~~i want to go back to situation where killing the last open window (be it native, vscode browser, whatever) kills the server. right now stuff keeps lingering for too long and it leads to confusing situations and difficulties with debugging~~ (done)
- ~~instead of shft + o, can i use cmd + o on mac and ctrl + o on linux/windows for open?~~ (done)
- ~~when i use the file picker and switch to compare or overlay mode, it doesnt work, i can select an array but i dont get them side-by-side.~~ (done)
- ~~allow me to specify a slice index number with number keys and then enter to go to that slice for the current active slice dim.~~ (done)
- ~~in qMRI mode, there can be 3 to 6 parameter maps. when i press q the first time, show them all. when i press q another time, only show T1, T2 and abs(PD) as three canvasses horizontally aligned.~~ (done)

each item a separate commit. no need for separate branches unless you work on things in parallel. in that case, do the merge (no merge commits pls, just rebase) afterwards. 