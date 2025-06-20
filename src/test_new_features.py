"""
Test crosshair cursor and screenshot export functionality
"""

import numpy as np
from arrayshow import arrayshow

# Create test data with interesting patterns
data = np.zeros((100, 100, 5))

for i in range(5):
    x, y = np.ogrid[0:100, 0:100]
    # Create different patterns for each slice
    if i == 0:
        # Checkerboard pattern
        data[:, :, i] = ((x//10) + (y//10)) % 2 * 100
    elif i == 1:
        # Circular gradient
        center_x, center_y = 50, 50
        data[:, :, i] = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    elif i == 2:
        # Linear gradient
        data[:, :, i] = x + y
    elif i == 3:
        # Sine wave pattern
        data[:, :, i] = 50 * (1 + np.sin(x/10) * np.cos(y/10))
    else:
        # Random noise
        data[:, :, i] = np.random.rand(100, 100) * 100

# add a 4th dimension that scales the data
data = data[:, :, :, np.newaxis] * np.linspace(1, 2, 75)[np.newaxis, np.newaxis, np.newaxis, :]

print("Testing new features:")
print("🎯 CROSSHAIR CURSOR:")
print("• Press 'x' to toggle crosshair cursor on/off")
print("• Move mouse over the image to see pixel coordinates and values")
print("• Values appear in small yellow text at top-left corner")
print("• Format: (x, y): value")
print("")
print("📸 SCREENSHOT EXPORT:")
print("• Click the 'Screenshot' button to save current view")
print("• Will include image, colorbar, and all UI elements")
print("• File dialog will open to choose save location")
print("")
print("🎨 COLORBAR:")
print("• Press 'b' to toggle colorbar visibility")
print("• Min/max values shown at bottom/top of colorbar")
print("• Try changing colormaps and color limits")
print("")
print(f"Data shape: {data.shape}")
print(f"Data range: [{data.min():.1f}, {data.max():.1f}]")

arrayshow(data)