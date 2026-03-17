# TODO


- no need for automatic colormap selection as introduced in commit 952fc1a1fe596ab6b2007c2796b38df49ae8fc2f, get rid of it
- the linked crosshair from a778ab75ef3bd4ae2cf33b33bbfa6bac17f5a696 is broken. get rid of it, dont need it somehow changes what i see in the viewer (like small version of the array in top left corner of the window with the info overlayed). fix this. 
- where do screenshots and slices (npy) get saved? 
- rect ROI is ok, i want to focus on a special roi mode later on
- the histogram with W is nice. right now i see the yellow vertical liens for current clim. would be nice to be able to drag them to chang clims. and see the value of current bin when hovering over the histogram
- automatically opening in vscode browser tab is still not working locally: arrayview main
❯ uv run arrayview --diagnose   
{
  "env": {
    "TERM_PROGRAM": "vscode",
    "VSCODE_IPC_HOOK_CLI": null,
    "SSH_CONNECTION": null,
    "SSH_CLIENT": null,
    "VSCODE_INJECTION": "1",
    "VSCODE_AGENT_FOLDER": null,
    "DISPLAY": "/private/tmp/com.apple.launchd.MzjoGL4Vi1/org.xquartz:0",
    "WAYLAND_DISPLAY": null
  },
  "detection": {
    "in_vscode_terminal": true,
    "is_vscode_remote": false,
    "in_vscode_tunnel": true,
    "can_native_window": false,
    "in_jupyter": false,
    "vscode_ipc_hook_recovered": null
  },
  "pid": 29667,
  "ppid": 29666,
  "platform": "darwin",
  "python": "/Users/oscar/Projects/packages/python/arrayview/.venv/bin/python3"
}

- make shift + o open picker as well since cmd or ctrl + o does not work well in vscode. welcome should say something like: pick array with {cmd,ctrl,shift} + o or drop array in window. keep it minimal. no cmd emoji thingy
- when i enable hover info, i need to move mouse for it to appear. tahts annoying it should appear immediatly.
- picker needs a rewrite. it should allows me to directly open another array with enter like it does now. no switching between modes with tab. i want to be able to select arrays with <space> and then it should switch to compare mode and also change the UI color like it does now. i want to be able to select up to four arrays for compare. they should then appear in 2x2 grid. for 2, it should be 1x2, for three 1x3. for more arrays, users just needs to cat them themselves. adjust the compare mode s.t. 4 is maximum. the diff and registration modes should only work (like now) when two arrays are selected.


each item a separate commit. no need for separate branches unless you work on things in parallel. in that case, do the merge (no merge commits pls, just rebase) afterwards.  