# Large Arrays in arrayview

arrayview works well with arrays from a few hundred MB to tens of GB.
This guide covers when to keep your data as `.npy` and when to convert to `.zarr`,
plus a ready-to-run conversion recipe.

---

## Format guide

### Keep `.npy` when

- Your array fits comfortably in RAM (or close to it).
- You scroll mostly sequentially along one axis on a fast local disk.
- You don't need random axis-swaps on very large volumes.

`.npy` is loaded with `mmap_mode="r"` so it's already lazy —
but the first frame is slower (OS page-faults the file in) and RSS
grows proportional to the data touched.

### Use `.zarr` when

- Your array is large (hundreds of MB to tens of GB).
- You frequently swap axes or jump around non-sequentially.
- Your storage is remote or high-latency (NFS, S3, SFTP).
- You want predictable, low first-frame latency regardless of file size.

`.zarr` with correct chunk shapes loads only the 2D slice you're viewing.
RAM delta stays near zero; first-frame latency is consistent.

---

## Recommended chunk shapes

arrayview exports `zarr_chunk_preset(shape)` that returns the recommended
chunk tuple for your array:

```python
from arrayview import zarr_chunk_preset

shape = my_array.shape          # e.g. (512, 512, 300)
chunks = zarr_chunk_preset(shape)   # → (512, 512, 1)
```

The presets by archetype:

| Shape | Preset | Notes |
|-------|--------|-------|
| `(Y, X, Z)` | `(Y, X, 1)` | One Z-slice per chunk |
| `(Y, X, Z, T)` | `(Y, X, 1, 2)` | One Z-slice, 2 T-frames |
| `(Y, X, Z, T, C)` | `(Y, X, 1, 1, C)` | Full channel axis in one chunk |
| `(Y, X, Z)` with Y or X > 1024 | `(1024, 1024, 1)` | XY tiling for very large planes |

For T-dominant playback (you scroll T more than Z), you may want a deeper
T chunk and shallower Z, e.g. `(Y, X, 1, 10)`.  Benchmark both.

---

## Conversion recipe

### From `.npy`

```python
import numpy as np
import zarr
from arrayview import zarr_chunk_preset

src = "mydata.npy"
dst = "mydata.zarr"

arr = np.load(src, mmap_mode="r")       # lazy load
chunks = zarr_chunk_preset(arr.shape)

z = zarr.open(dst, mode="w",
              shape=arr.shape,
              dtype=arr.dtype,
              chunks=chunks)
z[:] = arr                              # streams chunk-by-chunk

print(f"Wrote {dst}  chunks={chunks}")
```

### From NIfTI (`.nii` / `.nii.gz`)

```python
import nibabel as nib
import numpy as np
import zarr
from arrayview import zarr_chunk_preset

img = nib.load("scan.nii.gz")
arr = np.asarray(img.dataobj)           # materialise once (NIfTI is small)
chunks = zarr_chunk_preset(arr.shape)

zarr.save_array("scan.zarr", arr, chunks=chunks)
```

### With compression (remote / high-latency storage)

Add a lightweight compressor to save bandwidth without hurting decode speed:

```python
from numcodecs import Blosc

compressor = Blosc(cname="lz4", clevel=3, shuffle=Blosc.BITSHUFFLE)

z = zarr.open(dst, mode="w",
              shape=arr.shape,
              dtype=arr.dtype,
              chunks=chunks,
              compressor=compressor)
z[:] = arr
```

For maximum compression ratio at the cost of slightly more decode time, use
`cname="zstd"` with `clevel=3`.

---

## Verification checklist

After conversion, confirm parity:

```python
import numpy as np
import zarr

npy = np.load("mydata.npy", mmap_mode="r")
z   = zarr.open("mydata.zarr", mode="r")

assert npy.shape == z.shape,  f"shape mismatch: {npy.shape} vs {z.shape}"
assert npy.dtype == z.dtype,  f"dtype mismatch: {npy.dtype} vs {z.dtype}"

# Spot-check first and middle slice (axis 2)
mid = npy.shape[2] // 2
np.testing.assert_allclose(npy[:, :, 0],   z[:, :, 0],   rtol=1e-5)
np.testing.assert_allclose(npy[:, :, mid], z[:, :, mid], rtol=1e-5)

print("Parity OK")
print(f"Storage: {npy.nbytes/1e6:.1f} MB  →  zarr on disk (with compression)")
```

---

## Known tradeoffs

| | `.npy` memmap | `.zarr` |
|--|--|--|
| First-frame latency | Slow for large files (page-fault) | Fast (one chunk) |
| Warm scroll latency | Very fast (OS cache warm) | Slightly slower (decompress per chunk) |
| RAM usage | Grows with pages touched | Stays near zero |
| Axis-swap cost | Re-faults new pages | Same chunk overhead |
| Storage overhead | None | ~Same without compression; smaller with |
| Tooling | Universal | Needs zarr package |

---

## Tuning for advanced users

- **Larger T-depth**: increase `ZARR_T_DEPTH` in `_app.py` (default 2) if T-playback stutters.
- **XY tiling threshold**: adjust `ZARR_LARGE_XY_TILE` (default 1024 px) if your XY planes are between 1024–2048 and you see slow first frames.
- **No compressor** (default) is correct for local NVMe.  Add Blosc lz4 for network storage.

---

## Environment variable reference

| Variable | Default | Description |
|----------|---------|-------------|
| `ARRAYVIEW_RAW_CACHE_MB` | 5% of RAM | Raw float32 slice cache per session |
| `ARRAYVIEW_RGBA_CACHE_MB` | 10% of RAM | RGBA rendered slice cache per session |
| `ARRAYVIEW_MOSAIC_CACHE_MB` | 2.5% of RAM | Mosaic (z-grid) rendered cache per session |
| `ARRAYVIEW_HEAVY_OP_LIMIT_MB` | 500 | Max array size for FFT / GIF / grid operations |

Example — tighten memory on a shared machine:
```bash
export ARRAYVIEW_RAW_CACHE_MB=256
export ARRAYVIEW_RGBA_CACHE_MB=512
export ARRAYVIEW_HEAVY_OP_LIMIT_MB=200
arrayview large_scan.zarr
```

## Debug endpoint

```
GET /cache_info/{sid}
```

Returns current cache usage and budgets:
```json
{
  "raw_cache":    {"entries": 12, "used_mb": 12.3, "budget_mb": 859.0},
  "rgba_cache":   {"entries":  8, "used_mb": 32.1, "budget_mb": 1718.0},
  "mosaic_cache": {"entries":  0, "used_mb":  0.0, "budget_mb":  429.0},
  "heavy_op_limit_mb": 500.0
}
```
