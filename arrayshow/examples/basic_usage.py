import numpy as np
import matplotlib.pyplot as plt
from arrayshow import arrayshow

# Create or load your multi-dimensional array
array = np.random.rand(100, 100, 30, 5)

# Initialize the viewer
viewer = arrayshow(array)

