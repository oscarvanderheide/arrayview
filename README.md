# arrayview

A fast, minimal viewer for multi-dimensional arrays, inspired by the great
[arrShow](https://github.com/tsumpf/arrShow).

- CLI and Python useage
- Inline in Jupyter / vscode interactive window
- Runs locally, over SSH, and through VS Code tunnels

## CLI
`uvx arrayview your_array.npy`

## Python
```python
from arrayview import view
import numpy as np

np.random.rand(256,256,32,2)
view(x)
```


