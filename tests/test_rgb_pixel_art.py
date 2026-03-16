import numpy as np
from PIL import Image
import arrayview as av


def generate_animated_landscape_array(
    image_path, num_dn_steps=20, num_wind_steps=20, target_size=None
):
    # 1. Load the image
    img = Image.open(image_path).convert("RGB")

    # 2. Downsample the image if a target size is provided
    if target_size is not None:
        # Use NEAREST to preserve the sharp, blocky edges of pixel art!
        img = img.resize(target_size, resample=Image.Resampling.NEAREST)

    base_img = np.array(img)
    H, W, C = base_img.shape

    # Initialize the 5D numpy array: (DayNight_Steps, Wind_Steps, Height, Width, Channels)
    animated_array = np.zeros((num_dn_steps, num_wind_steps, H, W, C), dtype=np.uint8)

    # Pre-compute coordinate grids for the wind distortion
    x_coords, y_coords = np.meshgrid(np.arange(W), np.arange(H))

    # Create a mask so the wind only affects the lower part of the image (fields/flowers)
    wind_mask = np.clip((y_coords - H * 0.45) / (H * 0.1), 0, 1)

    for dn in range(num_dn_steps):
        # --- DAY/NIGHT CYCLE LOGIC ---
        t = (dn / num_dn_steps) * 2 * np.pi
        cycle = np.cos(t)
        night_weight = (1 - cycle) / 2

        base_mult = np.array([1.0, 1.0, 1.0])
        night_mult = np.array([0.25, 0.35, 0.65])

        current_color_mult = base_mult * (1 - night_weight) + night_mult * night_weight

        for w in range(num_wind_steps):
            # --- WIND ANIMATION LOGIC ---
            phase = (w / num_wind_steps) * 2 * np.pi

            shift = 2.0 * np.sin(phase + y_coords * 0.05 + x_coords * 0.02)
            shift_int = np.round(shift * wind_mask).astype(int)
            new_x = np.clip(x_coords + shift_int, 0, W - 1)

            wind_frame = base_img[y_coords, new_x]

            # --- APPLY AND STORE ---
            final_frame = np.clip(wind_frame * current_color_mult, 0, 255).astype(
                np.uint8
            )
            animated_array[dn, w] = final_frame

    return animated_array


# Example usage:
animated_grid = generate_animated_landscape_array(
    "vangogh.jpg",
    target_size=(768, 512),
)
print(animated_grid.shape)  # Output will be: (20, 20, Height, Width, 3)

av.view(animated_grid, rgb=True)
