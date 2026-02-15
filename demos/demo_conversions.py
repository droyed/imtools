import os
import cv2
import numpy as np
from imtools import imwrite
from demo_config import get_output_dir

# Create output directory
output_dir = get_output_dir('formats')
os.makedirs(output_dir, exist_ok=True)


# =============================================================================
# Data Type Tests
# =============================================================================

# 1. Boolean Mask (Black/White) -> PNG
bool_arr = np.random.choice([True, False], (100, 100))
imwrite(bool_arr, os.path.join(output_dir, "test_bool.png"))

# 2. Float (0.0 - 1.0) -> JPG
float_arr = np.random.rand(100, 100).astype(np.float32)
imwrite(float_arr, os.path.join(output_dir, "test_float.jpg"))

# 3. Float Out of Bounds (Should Crash)
try:
    bad_float = float_arr.copy()
    bad_float[0, 0] = 1.5  # Invalid
    imwrite(bad_float, os.path.join(output_dir, "crash.jpg"))
except ValueError as e:
    print(f"Caught expected error: {e}")

# 4. Uint16 -> PNG (Preserves 16-bit, check file properties)
uint16_arr = np.random.randint(0, 65535, (100, 100), dtype=np.uint16)
imwrite(uint16_arr, os.path.join(output_dir, "test_16bit.png"))

# 5. Uint16 -> JPEG (Auto-downscales to 8-bit)
imwrite(uint16_arr, os.path.join(output_dir, "test_16bit_downscaled.jpg"))


# =============================================================================
# Alpha Channel and BGR Handling Tests
# =============================================================================

# Create a sample BGR image (300x300, solid blue color)
img_bgr = np.full((300, 300, 3), (255, 0, 0), dtype=np.uint8)

height, width = img_bgr.shape[:2]
print(f"Original Shape: {img_bgr.shape}")  # (300, 300, 3)

# Create an Alpha Channel: Opaque (255) at top, Transparent (0) at bottom
alpha_channel = np.linspace(255, 0, height, dtype=np.uint8)
alpha_channel = np.tile(alpha_channel[:, np.newaxis], (1, width))

# Stack them together - 3-channel BGR + 1-channel Alpha = 4-channel BGRA
img_bgra = np.dstack((img_bgr, alpha_channel))

print(f"New Shape: {img_bgra.shape}")  # (300, 300, 4)

# Save with transparency using BGR input order
# The function handles swapping Blue and Red while keeping Alpha in place
imwrite(img_bgra, os.path.join(output_dir, "output_with_transparency.png"), input_channel_order="BGR")

# Save to JPEG - function auto-composites RGBA on white background
imwrite(img_bgra, os.path.join(output_dir, "output_flattened.jpg"), input_channel_order="BGR")
