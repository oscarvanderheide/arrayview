# arrayview

A fast, minimal viewer for multi-dimensional arrays, inspired by the great
[arrShow](https://github.com/tsumpf/arrShow) and [sigpy's ImagePlot](https://github.com/mikgroup/sigpy/blob/5da0e8605f166be41e520ef0ef913482487611d8/sigpy/plot.py#L48).

- CLI and Python useage
- Inline in Jupyter / vscode interactive window
- Runs locally, over SSH, and through VS Code tunnels

## CLI

`uvx arrayview your_array.npy`

## Python

`uv add arrayview`

```python
from arrayview import view
import numpy as np

np.random.rand(256,256,32,2)
view(x)
```
