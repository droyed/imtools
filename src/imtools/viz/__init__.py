"""Visualization subpackage for imtools.

Re-exports color generation, the overlay pipeline, image composition
utilities, display helpers, and the interactive overlay viewer.

Public API:
    - :func:`~imtools.viz.colors.generate_colors`
    - :func:`~imtools.viz.colors.create_color_palette_image`
    - :func:`~imtools.viz.pipeline.overlay_visualize`
    - :func:`~imtools.viz.pipeline.draw_labels`
    - :func:`~imtools.viz.pipeline.create_label_overlay_from_labelimg`
    - :func:`~imtools.viz.compose.text_on_canvas`
    - :func:`~imtools.viz.compose.stack_images`
    - :func:`~imtools.viz.compose.add_title`
    - :func:`~imtools.viz.display.show_cv2`
    - :func:`~imtools.viz.display.show_mpl`
    - :func:`~imtools.viz.display.show_cv2_fullscreen`
    - :class:`~imtools.viz.overlay_viewer.OverlayViewer`
    - :func:`~imtools.viz.overlay_viewer.run_overlay_viewer`
"""

from imtools.viz.colors import generate_colors, create_color_palette_image
from imtools.viz.pipeline import overlay_visualize, draw_labels, create_label_overlay_from_labelimg
from imtools.viz.compose import text_on_canvas, stack_images, add_title
from imtools.viz.display import show_cv2, show_mpl, show_cv2_fullscreen
from imtools.viz.overlay_viewer import OverlayViewer, run_overlay_viewer
