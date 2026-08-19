---
name: handoff-vscode-jupyter-inline
description: Resolved 2026-08-19 — inline array display inside a VS Code notebook cell over a tunnel. The blocker was never a webview sandbox; a private devtunnel forward answers the cell with a GitHub sign-in redirect.
last_updated: 2026-08-19
---

# Resolved: inline `view()` output in a VS Code notebook cell, over a tunnel

`av.view(arr)` in a `.ipynb` opened natively in VS Code, over a tunnel,
renders the array in the cell. Verified on the real host, 2026-08-19.

## What it actually was

A cell's output renders in a webview on the VS Code **client**. Over a tunnel
that client has nothing listening on the remote host's `localhost`, and it
carries no devtunnel cookie, so the forwarded address answers it with a
sign-in redirect. Measured, not inferred — `curl` of the forwarded address
from the remote host:

```
HTTP/2 302
location: https://global.rel.tunnels.api.visualstudio.com/auth/github?...
```

The viewer **tab** never hit this: VS Code fetches that page itself,
remote-side, over the private route. That asymmetry is why the tab kept
working and why three earlier sessions (`cda022b`/`f0a8634` 2026-03-31,
`4ecd23c` 2026-04-16, and the first half of this one) all concluded "the
notebook webview cannot reach localhost, it's a sandbox". Nothing is
sandboxed. The address simply does not resolve to the backend, and the only
address that does is a **public** forward.

The user's memory was correct on both counts: it did work before, and the
change that broke it was theirs. `5542682` (2026-07-29) removed the
`remote.tunnel.privacypublic` promotion; `4d33a61` (2026-08-03), the
classic-Jupyter proxy-route change, then broke it a second, separate way by
adding `jupyter_server_proxy` as a dependency and removing the fallback that
had been covering for it.

## What was changed

- **`extension.js`** — `resolvePublicTunnelBase()` restores the pre-`5542682`
  promotion **verbatim** (`ensurePortPublic`, `_tunnelItem`,
  `_publicBaseFromTunnelResult`, `_cachedTunnelBases`, the
  `EXTERNAL_URI_ATTEMPTS` ladder, the 1.5 s wait before promotion). Do not
  rewrite these; they were arrived at by measurement and a from-scratch
  version was tried here first and was worse.
- Driven by a dedicated `~/.arrayview/public-port-request-*.json` signal with
  its own ack, claimed by atomic rename. Deliberately **separate from the
  open-preview pipeline** so tab delivery cannot regress.
- **`_launcher.py`** — `_inline_url_for_vscode_tunnel()` rewrites the inline
  URL onto the published route, memoised per (port, backend) for the process.
  Only `_in_vscode_tunnel()` reaches it.
- **The proxy-vs-direct choice is made by the page, not by Python.**
  `_build_jupyter_inline_html` emits both addresses and picks with
  `/^https?:$/.test(window.location.protocol)`: a classic Notebook/Lab page has
  an http origin for the relative `/proxy/<port>/` path to resolve against, a
  VS Code cell's webview does not. `_should_use_jupyter_proxy_inline()` now
  only reports whether `jupyter_server_proxy` is importable.

  A first attempt gated this in Python on `_in_vscode_terminal()`, which
  **regressed a real user**: a `jupyter lab` started from a VS Code terminal on
  a remote and opened in an ordinary browser is indistinguishable from a VS Code
  notebook kernel by process ancestry, and lost its proxy route. Do not
  reintroduce an environment sniff here — the page is the only thing that
  knows. Escape hatch if it is ever needed: `ARRAYVIEW_JUPYTER_PROXY=1|0`.
- **`_vscode_signal.py`** — `_public_tunnel_base()` returns `(base, reason)`.
- Failure now renders VS Code's own explanation in the cell instead of black.

## The standing hazard — read this before debugging a black cell again

Publishing needs a **free forwarded-port slot on the tunnel**, and VS Code
auto-forwards every listening port it notices. On this host that was MATLAB's
service host, the notebook kernel's zmq ports, and VS Code's own server —
eight slots, none of them ArrayView's, and the tunnel hit its
`PortsPerTunnel` limit. Symptom: ~26 s of nothing, then a black cell.
The real message was in `~/.arrayview/extension.log` all along:

```
429 Too Many Requests ... "Resource limit exceeded" ... 'PortsPerTunnel'
```

`arrayview --kill` does **not** help: those forwards belong to other programs
and live on Microsoft's side.

The cap is **10 ports per tunnel**, and the Ports view is not the whole truth —
6 of the 10 here were orphans no window listed. Use Microsoft's CLI to see and
clear them (installed at `~/bin/devtunnel`, no sudo, not on `PATH`):

```bash
devtunnel list                                  # tunnel id + port count
devtunnel port list <tunnel-id>                 # every registration
ss -ltnp                                        # which of them are actually live
devtunnel port delete <tunnel-id> -p <port>     # only the dead ones
```

Prevention: `remote.autoForwardPortsSource` is set to `output` (in
`~/.vscode-server/data/Machine/settings.json` — that is the data root a tunnel
server actually reads, *not* `~/.vscode/data`, which is where a first attempt
wrongly wrote it and had no effect). The default `process` source forwards
every listening port it scans, which is what filled this tunnel.

ArrayView also takes a fresh port per launch, which on a tunnel feeds the same
wall. The inline path now reuses one published route per session; the tab
path's cold-start port is untouched and still mints one per launch.

## Settled, do not reopen

Making the port public is now **deliberate and inline-only**, agreed with the
user on 2026-08-19 after being shown that it means anyone with the link can
see that array while the cell is open. This does not reopen the general
"stop trying to make ports public" decision in `CLAUDE.md`: the viewer tab
still uses the private route and must keep doing so.
