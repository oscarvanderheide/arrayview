import numpy as np

from arrayshow import arrayshow

print("Generating complex array sample data...")

# Create a 3D complex array with interesting structure
shape = (64, 64, 20)
x, y, z = np.ogrid[0:shape[0], 0:shape[1], 0:shape[2]]

# Create a complex wave pattern
real_part = np.sin(2 * np.pi * x / 16) * np.cos(2 * np.pi * y / 16)
imag_part = np.cos(2 * np.pi * x / 12) * np.sin(2 * np.pi * y / 12)

# Add some z-dependence
real_part = real_part * np.exp(-z / 10)
imag_part = imag_part * np.exp(-z / 15)

# Create complex array
complex_data = real_part + 1j * imag_part

# Add some noise
complex_data += 0.1 * (np.random.randn(*shape) + 1j * np.random.randn(*shape))

print(f"Complex array shape: {complex_data.shape}")
print(f"Array dtype: {complex_data.dtype}")
print(f"Real part range: [{np.real(complex_data).min():.3f}, {np.real(complex_data).max():.3f}]")
print(f"Imaginary part range: [{np.imag(complex_data).min():.3f}, {np.imag(complex_data).max():.3f}]")
print(f"Magnitude range: [{np.abs(complex_data).min():.3f}, {np.abs(complex_data).max():.3f}]")
print(f"Phase range: [{np.angle(complex_data).min():.3f}, {np.angle(complex_data).max():.3f}]")

print("\nUse the dropdown menu or press 'c' to cycle between:")
print("- Magnitude")
print("- Phase") 
print("- Real")
print("- Imaginary")

arrayshow(complex_data)