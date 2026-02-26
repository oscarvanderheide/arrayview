# arrayview

A fast, minimal viewer for multi-dimensional arrays, inspired by the great
[arrShow](https://github.com/tsumpf/arrShow) and [sigpy's ImagePlot](https://github.com/mikgroup/sigpy/blob/5da0e8605f166be41e520ef0ef913482487611d8/sigpy/plot.py#L48).

- CLI and Python useage
- Inline in Jupyter / vscode interactive window
- Runs locally, over SSH, and through VS Code tunnels

## CLI

`uvx arrayview your_array.npy`

Opens in a native window. To open in browser, pass `--browser` flag.

## Python

`uv add arrayview`

```python
from arrayview import view
import numpy as np

np.random.rand(256,256,32,2)
view(x)
```

## Development

```bash
git clone https://github.com/oscarvanderheide/arrayview
cd arrayview
uv sync --group test
uv run playwright install chromium
```

Run the tests:

```bash
uv run pytest tests/              # all (API + browser, ~100s)
uv run pytest tests/test_api.py   # HTTP layer only (~40s)
uv run pytest tests/test_browser.py  # Playwright/Chromium only (~60s)
```

Visual regression baselines live in `tests/snapshots/`. Delete a file there to reset its baseline.
