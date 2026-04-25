# Configuration

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

Rounded panes are on by default. Disable them with:

```bash
arrayview config set viewer.rounded_panes false
```

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
