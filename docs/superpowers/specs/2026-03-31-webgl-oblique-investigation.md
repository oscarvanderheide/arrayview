# WebGL Oblique Slice Acceleration — Investigation Notes

## Current architecture
- Python backend computes oblique slices via `scipy.ndimage.map_coordinates`
- Result is JPEG-encoded and sent to browser over HTTP
- Draft mode (half-res, nearest-neighbor) during drag, full quality on release
- ~60ms throttle between renders

## WebGL approach
Upload volume as `TEXTURE_3D` (R16F half-float), render oblique plane in a fragment shader.
Hardware trilinear interpolation. Colormap as 256x1 RGBA8 texture, vmin/vmax as uniforms.

### Pros
- Eliminates server round-trip during oblique drag (pure client-side, 60fps)
- Colormap and contrast adjustments become instant (no fetch)
- Works for all users with a browser (95%+ WebGL2 support)
- Draft mode no longer needed

### Cons / concerns
- **4D data**: each 4th-dimension index change requires re-uploading the 3D subvolume
  - 512^3 R16F = 256MB raw, ~80MB gzipped
  - Localhost: ~0.5-1s, LAN: ~6-8s, remote tunnel: 20-80s
  - Dealbreaker for fast 4D scrubbing over remote connections
- **GPU memory**: volume lives in both server RAM and browser GPU memory
  - 512^3 R16F = 256MB GPU. Tight on integrated/mobile GPUs
- **R16F precision**: ~3.3 decimal digits, max 65504. Fine for most medical imaging
- **Context loss**: GPU can kill WebGL context anytime, must handle and fall back
- **MAX_3D_TEXTURE_SIZE**: spec minimum is 256, desktop GPUs do 2048+, some mobile only 256
- **OES_texture_float_linear**: needed for hardware filtering on float textures, not guaranteed on mobile

### Rejected: CuPy
Near drop-in `cupyx.scipy.ndimage.map_coordinates`, 10-50x speedup on CUDA.
Rejected because `pip install cupy-cuda12x` requires users to know their CUDA version.

### Integration strategy (if implemented)
- Render to hidden WebGL canvas, draw onto 2D canvas via `ctx2d.drawImage(webglCanvas)`
- All existing overlay drawing (crosshairs, labels) stays on 2D canvas
- Fallback to CPU path if WebGL2 unavailable or texture upload fails
- Hybrid: only use WebGL during oblique drag, keep server path for normal slicing

### Decision
Parked for now. The 4D re-upload cost makes this unsuitable as a universal replacement.
Worth revisiting if oblique drag performance becomes a priority or if a volume caching
strategy can amortize the upload cost.
