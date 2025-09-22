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

```uv add "arrayview @ git+https://github.com/oscarvanderheide/arrayview"```

### Command-Line Usage

```sh
uvx --from https://github.com/oscarvanderheide/arrayview.git arrayview example_array.nii.gz
```

### Interactive Python Example

```python
import numpy as np
from arrayview import ArrayView

ArrayView(np.random.rand(2, 4, 128, 128))
```

### Scripting: Multiple Plots

```python
import numpy as np
from arrayview import ArrayView
import matplotlib.pyplot as plt

ArrayView(np.random.rand(2, 4, 100, 100))
ArrayView(np.random.rand(100, 100))
# ... more plots ...

plt.show()  # Keeps all windows open until you close them
```

---

## Hotkeys

- `h` — Show/hide hotkey menu
- `x/y/z` — Set current axis as x/y/z
- `t` — Swap x and y axes
- `c` — Cycle colormaps
- Arrow keys — Change axis/slice
- `a` — Toggle axes/labels
- `m/p/r/i/l` — Magnitude/phase/real/imag/log mode
- `s` — Save as PNG
- `g/v` — Save as GIF/video
- ...and more! (see in-app help)

---

## Notes
- Supports `.nii.gz`, `.nii`, and `.npy` files from the command line.
- All dependencies (including PyQt5) are installed automatically when using `uv add arrayview` or `uv pip install -e .`.

---

Enjoy fast, keyboard-driven array exploration!
