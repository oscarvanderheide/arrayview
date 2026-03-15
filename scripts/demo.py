"""
ArrayView demo script — exercises all major viewing modes.

Run from the project root:
    uv run python scripts/demo.py

Opens a native window with tabs for several synthetic arrays.
"""

import numpy as np
import arrayview as av

rng = np.random.default_rng(42)

# ── 2D grayscale ───────────────────────────────────────────────────────────
arr_2d = np.outer(np.sin(np.linspace(0, 4 * np.pi, 200)),
                  np.cos(np.linspace(0, 4 * np.pi, 200)))

# ── 3D volume: scroll slices, v → 3-plane, V → custom dims ────────────────
x, y, z = np.mgrid[-1:1:64j, -1:1:64j, -1:1:64j]
arr_3d = np.exp(-(x**2 + y**2 + z**2) * 6) + 0.15 * rng.standard_normal((64, 64, 64))

# ── 4D time series: Space → auto-play ─────────────────────────────────────
t = np.linspace(0, 2 * np.pi, 20)
arr_4d = np.stack([np.sin(tt + x[:, :, 0]) for tt in t], axis=-1)  # 64×64×20

# ── RGB image: c/d keys no-op, colorbar hidden ────────────────────────────
yy, xx = np.mgrid[0:256, 0:256]
arr_rgb = np.stack([
    np.uint8(127 + 127 * np.sin(xx / 20.0)),
    np.uint8(127 + 127 * np.cos(yy / 20.0)),
    np.uint8(127 + 127 * np.sin((xx + yy) / 28.0)),
], axis=-1)

# ── Overlay: base + two binary masks ([ / ] → alpha) ─────────────────────
mask_a = np.zeros((200, 200), dtype=np.uint8)
mask_a[40:90, 40:90] = 1
mask_b = np.zeros((200, 200), dtype=np.uint8)
mask_b[110:160, 110:160] = 1

# ── Complex: m → cycle mag / phase / real / imag ──────────────────────────
arr_complex = (arr_2d + 1j * rng.standard_normal(arr_2d.shape)).astype(np.complex64)

print("Opening ArrayView demo — all modes in separate tabs.")
print()
print("  [2d]      — 2D grayscale                 c: colormap  d: range")
print("  [3d]      — 3D volume                    v: 3-plane   scroll: slices")
print("  [4d]      — 4D time series               Space: auto-play")
print("  [rgb]     — RGB image                    (colorbar hidden)")
print("  [overlay] — 2D + two masks               [/]: alpha")
print("  [complex] — complex array                m: mag/phase/re/im")
print()
print("Universal: O: open/compare  ?: help  T: theme  L: log  Cmd+Shift+[/]: tabs")

# First call opens the native window; subsequent calls inject tabs.
av.view(arr_2d, name="2d")
av.view(arr_3d, name="3d")
av.view(arr_4d, name="4d")
av.view(arr_rgb, name="rgb", rgb=True)
av.view(arr_2d, name="overlay", overlay=[mask_a, mask_b])
av.view(arr_complex, name="complex")
