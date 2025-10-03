# arrayview

**arrayview** is a fast, hotkey-driven viewer for 2D slices of N-dimensional NumPy arrays.
The base functionality is copied from [sigpy's
ImagePlot](https://github.com/mikgroup/sigpy). It is slightly adapted and extended as a
standalone package for broader use.

---

## Features

- 🖱️ Fully keyboard-driven (press `h` for hotkey help)
- Switch viewing dimensions, scroll dimension, transpose and reverse dimensions
- Supports arbitrary-dimensional arrays
- 🧑‍💻 Usable in scripts, interactive sessions, or as a CLI
- ⚡ Uses matplotlib (PyQt5 backend)
- 🖥️ Works over SSH with X forwarding

---

## Quick Start

### Installation

This package is *not* published to PyPi. To add it as a dependency to your Python project, use the GitHub URL directly: 

```uv add "arrayview @ git+https://github.com/oscarvanderheide/arrayview"```

### Interactive Python Example

```python
import numpy as np
from arrayview import ArrayView

ArrayView(np.random.rand(2, 4, 128, 128))
```

### Command-Line Usage

Instead of adding the package as a dependency to a Python project, it can be used as a command-line tool as well:

```sh
uvx --from https://github.com/oscarvanderheide/arrayview.git arrayview example_array.nii.gz
```

This will automatically use the latest commit of the `main` branch on GitHub. It might be useful to make an alias, e.g.

```sh
alias av='uvx --from https://github.com/oscarvanderheide/arrayview.git arrayview'
```

---

## Hotkeys

Press `h` to see hotkeys

---

## Notes
- Supports `.nii.gz`, `.nii`, `.npy` and `.mat` files from the command line.
- Supports `torch` tensors can be passed to `ArrayView` directly in interactive/scripting mode.
- All dependencies (including PyQt5) are installed automatically when using `uv add "arrayview @ git+https://github.com/oscarvanderheide/arrayview"`.
- Works over ssh (using the `-X` or `-Y` option)
