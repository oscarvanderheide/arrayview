"""Generate test arrays for vector field overlay testing.

Run: uv run python tests/make_vectorfield_test_arrays.py
Produces: tests/data/phantom.npy and tests/data/vfield.npy
"""

import os
import numpy as np

out_dir = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(out_dir, exist_ok=True)

# ── Image: 3D sphere phantom (shape 40×80×80) ──────────────────────────
D, H, W = 192, 256, 256
z = np.linspace(-1, 1, D)[:, None, None]
y = np.linspace(-1, 1, H)[None, :, None]
x = np.linspace(-1, 1, W)[None, None, :]
r = np.sqrt(x**2 + y**2 + z**2)
image = np.where(r < 0.7, 200 - 100 * r / 0.7, 20).astype(np.uint8)

# ── Vector field: smooth rotation + radial push (shape 40×80×80×3) ─────
# Components: (dz, dy, dx) — each voxel gets a 3-vector
shape = (D, H, W)
vx = np.broadcast_to(-y * 0.4, shape).copy()  # rotate in x-y plane
vy = np.broadcast_to(x * 0.4, shape).copy()
vz = np.broadcast_to(z * 0.2, shape).copy()  # slight axial expansion
vfield = np.stack([vz, vy, vx], axis=-1).astype(np.float32)

# add a time dimension to the vector fields where the vectors rotate over time
T = 25  # number of time steps
vfield_time = np.zeros((T, D, H, W, 3), dtype=np.float32)
for t in range(T):
    angle = t / T * 2 * np.pi  # full rotation over T steps
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    # Rotate the x-y components of the vector field
    vfield_time[t, ..., 0] = vz  # z component stays the same
    vfield_time[t, ..., 1] = cos_a * vy + sin_a * vx  # rotate y component
    vfield_time[t, ..., 2] = cos_a * vx - sin_a * vy  # rotate x component

np.save(os.path.join(out_dir, "phantom.npy"), image)
np.save(os.path.join(out_dir, "vfield.npy"), vfield)
np.save(os.path.join(out_dir, "vfield_time.npy"), vfield_time)
print(f"phantom:     {image.shape} {image.dtype}")
print(f"vfield:      {vfield.shape} {vfield.dtype}")
print(f"vfield_time: {vfield_time.shape} {vfield_time.dtype}")
print(f"saved to {out_dir}/")
