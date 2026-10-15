# Configuration

## Preferences menu

Open the gear menu in the lower-right corner. Viewer changes apply immediately
and are saved on the machine running ArrayView.

Window defaults apply to the next launch. Remote SSH, VS Code Tunnel, and remote
Jupyter sessions save to the remote backend user's `~/.arrayview/config.toml`.

## Per-environment defaults

Set a window backend for each detected environment:

```bash
arrayview config set window.terminal browser
arrayview config set window.vscode vscode
arrayview config set window.jupyter inline
arrayview config set window.ssh browser
arrayview config set window.default browser
```

```bash
arrayview config list     # show current config
arrayview config reset    # delete config file
```

Stored in `~/.arrayview/config.toml`.

## Viewer defaults

```bash
arrayview config set viewer.rounded_panes false
arrayview config set viewer.ortho_layout big-left
arrayview config set viewer.dimbar_mode extended
```

Ortho layouts: `horizontal`, `big-left`.

Dimbar modes: `compact`, `extended`.

## Environment variable

```bash
ARRAYVIEW_WINDOW=browser uvx arrayview img.npy
```

## Priority

Explicit `--window` flag > `ARRAYVIEW_WINDOW` env var > config file > built-in default.

## Detected environments

| Environment | Detected when |
|-------------|---------------|
| `terminal` | Plain terminal (no VS Code, SSH, or Jupyter) |
| `vscode` | VS Code integrated terminal |
| `jupyter` | Jupyter / IPython notebook kernel |
| `ssh` | SSH session without VS Code |
| `julia` | Julia via PythonCall / PyCall |

MATLAB uses the detected terminal, VS Code, SSH, or remote route. Local MATLAB
prefers a native window when available. See [MATLAB and Julia](foreign-hosts.md).

## Reopen where you left off

Turn on **Reopen where you left off** in the Preferences menu (or
`arrayview config set viewer.remember_view true`) and opening a file again
brings back its slice, view (normal, 3-plane, qMRI), colormap and rotation.
The display range comes back too, unless the file has changed since. Options
given at launch, such as `--index` or `--cmap`, take precedence. Views are
kept in `~/.arrayview/view_memory.json`; a comparison of several arrays is not
remembered.
