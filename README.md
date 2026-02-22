# ndviewer

Lightning-fast browser-based viewer for multi-dimensional scientific arrays (MRI, fMRI, etc.).
Works locally and over VS Code tunnels / SSH port-forwarding.

```bash
uvx ndviewer data.npy
uvx ndviewer scan.nii.gz
uvx ndviewer volume.zarr
```

---

## Why it feels instant

### The problem with naive viewers

When you press `j` to scroll to the next slice, a viewer has to:

1. **Read** raw numbers from disk
2. **Normalize** them (find min/max, scale to 0–255)
3. **Apply a colormap** (map each value to an RGBA colour)
4. **Send the pixels** to your screen

Steps 2–4 are fast — a few milliseconds on any modern CPU.
Step 1 — **disk I/O** — is the killer. An SSD can do ~500 MB/s, but a single 256×256 float32
slice is only ~256 KB. The latency to *start* a read (seek time + OS overhead) is easily
10–30 ms. That's enough to make scrolling feel laggy.

### Fix 1 — preload everything into RAM on startup

When the viewer starts, a **background thread** immediately reads and renders every slice
in the current scroll dimension into an in-memory cache.

```
you open the file
  → background thread starts rendering all 200 slices
  → you see slice 100 instantly (it was fetched first)
  → ~2 seconds later: all 200 slices are in RAM
  → scrolling from that point = pure memory lookup, zero disk
```

For a typical fMRI volume (256×256×200, ~52 MB of RGBA) this completes in 1–3 seconds.
After that, no matter how fast you scroll, the viewer never touches disk again — every
keypress is answered from RAM in under a millisecond.

A status line shows progress: `Preloading dim 2: 87/200 (43%)`.

If the array is very large (>500 MB RGBA for the active dimension), the full preload is
skipped and the viewer falls back to prefetching 4 slices ahead of your current position.

### Fix 2 — WebSockets instead of HTTP requests

Matplotlib and most simple viewers use the pattern: keypress → HTTP request → wait for
response → draw. Each HTTP request has overhead (TCP handshake, headers, etc.) even on
localhost.

This viewer keeps a **persistent WebSocket connection** open. A keypress sends a tiny JSON
message (< 100 bytes) over the already-open socket. The server replies with raw RGBA bytes.
Round-trip latency is ~1 ms on localhost vs ~10–30 ms for HTTP.

More importantly: **stale frames are dropped**. If you hold down `k` and 10 keypress
messages pile up while the server is rendering slice 50, the server skips straight to
slice 60. You never wait for intermediate frames you've already scrolled past.

### Fix 3 — LUT rendering instead of Pillow/Matplotlib

Colormaps are pre-computed as **lookup tables** (256 × 4 RGBA arrays). Applying a
colormap to a slice is then a single numpy index operation:

```python
rgba = LUT[(normalized_slice * 255).astype(uint8)]  # one array index, very fast
```

No Pillow colormap pipeline, no Matplotlib figure overhead — just a numpy fancy-index.

### Fix 4 — raw RGBA over the wire

The server sends **raw RGBA bytes** (not JPEG or PNG). The browser receives them and
calls `ctx.putImageData()` directly — no decode step. For a 256×256 image that's 256 KB,
which transfers in < 1 ms on localhost.

---

## Keybindings

| Key | Action |
|-----|--------|
| `j` / `↓` | previous slice |
| `k` / `↑` | next slice |
| `h` / `←` | previous slice dimension |
| `l` / `→` | next slice dimension |
| `x` | swap horizontal dim with slice dim |
| `y` | swap vertical dim with slice dim |
| `Space` | toggle auto-play |
| `z` | toggle grid (mosaic of all slices) |
| `c` | cycle colormap (gray / viridis / plasma / RdBu_r) |
| `d` | cycle dynamic range (0–100% / 1–99% / 5–95% / 10–90%) |
| `s` | save screenshot (PNG) |
| `g` | save GIF of current slice dimension |
| scroll | scroll through slices |
| `?` | toggle help overlay |

---

## Supported formats

| Format | Notes |
|--------|-------|
| `.npy` | NumPy array (memory-mapped, no full load on startup) |
| `.nii`, `.nii.gz` | NIfTI (nibabel, memory-mapped) |
| `.zarr`, `.zarr.zip` | Zarr array (chunked, good for arrays that don't fit in RAM) |

---

## Installation

```bash
# Run directly without installing (requires uv)
uvx ndviewer myfile.npy

# Or install into a virtualenv
pip install ndviewer
ndviewer myfile.npy --port 8001
```

---

## Architecture

```
CLI (argparse)
  └─ loads file (mmap / nibabel / zarr)
  └─ samples data for global contrast stats
  └─ starts FastAPI + uvicorn

Browser ──── WebSocket /ws ────► server
             keypress (JSON)       └─ render_rgba() [thread pool]
          ◄── raw RGBA bytes ──────   └─ extract_slice() → LUT → cache

Background thread (on startup / dim change):
  → renders all N slices into _rgba_cache
  → JS polls /preload_status every 500 ms → shows progress bar
```
