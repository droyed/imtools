"""
imtools - Image processing and visualization toolkit.

Core utilities for image conversion, mask operations, converters,
color generation, and visualization.
"""

__version__ = "0.2.0"

# --- Core types & format converters ---
from imtools.core.types import Annotation, BlendConfig, TitleConfig, LabelStyle
from imtools.core.formats import (
    pil_to_opencv,
    opencv_to_pil,
    to_pil_image,
    to_numpy_image,
    imwrite,
)

# --- Mask operations & converters ---
from imtools.masks.ops import get_biggest_blob, bbox_from_mask, fill_holes_mask
from imtools.masks.converters import masks_to_label_image, binary_mask_to_label_image

# --- Annotation parsing & label formatting ---
from imtools.annotations.parsers import yolo_to_annotations, label_image_to_annotations
from imtools.annotations.labels import LabelFormat, LabelContext, resolve_label
from imtools.annotations.yolo import yolo_to_label_image

# --- Visualization ---
from imtools.viz.pipeline import overlay_visualize, draw_labels, create_label_overlay_from_labelimg
from imtools.viz.colors import generate_colors, create_color_palette_image
from imtools.viz.compose import text_on_canvas, stack_images, add_title
from imtools.viz.display import show_cv2, show_mpl, show_cv2_fullscreen
from imtools.viz.overlay_viewer import OverlayViewer, run_overlay_viewer
