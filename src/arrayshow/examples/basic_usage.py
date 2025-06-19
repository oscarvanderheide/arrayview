import numpy as np

from arrayshow import arrayshow

print("Generating 5D sample data...")
shape = (128, 128, 30, 20, 10)
grid = np.ogrid[tuple(slice(-1, 1, 1j * s) for s in shape)]
hypersphere = sum(g**2 for g in grid) < 0.9**2
numpy_data = hypersphere * 200 + np.random.rand(*shape) * 50
arrayshow(numpy_data)
