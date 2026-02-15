"""
imtools - Image processing and visualization toolkit.

Core utilities for image conversion, mask operations, converters,
color generation, and visualization.
"""

__version__ = "0.2.0"

"""imtools - Image processing and visualization toolkit."""

# Image I/O
from .formats import pil_to_opencv, opencv_to_pil, to_pil_image, to_numpy_image, imwrite

# Mask operations
from .mask_utils import get_biggest_blob, bbox_from_mask, fill_holes_mask

# Converters
from .converters import binary_mask_to_label_image

# Benchmarks
from . import benchmarks

# Visualization
from .viz_pipeline import create_label_overlay_from_labelimg, overlay_visualize
from .viz import overlay_viewer
from .display_utils import show_cv2, show_mpl

# Colors
from .color_gen import generate_colors

__all__ = [
    "pil_to_opencv", "opencv_to_pil", " to_pil_image", "to_numpy_image", "imwrite",
    "get_biggest_blob", "bbox_from_mask", "fill_holes_mask",
    "binary_mask_to_label_image",
    "create_label_overlay_from_labelimg", "overlay_visualize",
    "show_cv2", "show_mpl",
    "generate_colors",
    "benchmarks",
]