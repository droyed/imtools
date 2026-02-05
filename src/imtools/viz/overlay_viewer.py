"""
Dear PyGui Mask Overlay Viewer

A minimal GUI application to visualize a mask overlay on an image
with real-time alpha blending control via a slider and dynamic
method/parameter dropdowns.

Usage:
    from overlay_viewer import run_overlay_viewer
    blended_image, settings = run_overlay_viewer(image, label_image)

Returns:
    tuple: (PIL.Image.Image, dict) - The blended image and settings dictionary
           containing 'alpha', 'method', and method-specific parameters.
"""

import os
import platform
from typing import Any, Callable

import numpy as np
import cv2
import dearpygui.dearpygui as dpg
from PIL import Image
from imtools.color_gen import generate_colors

# Constants
MIN_WINDOW_WIDTH = 600
MIN_WINDOW_HEIGHT = 400
PADDING = 20
CONTROLS_WIDTH = 280

# Visual styling constants
SHADOW_OFFSET = 4
BACKGROUND_COLOR = (18, 18, 18)
CONTROL_BG_COLOR = (28, 28, 28)
CONTROL_BG_HOVER = (38, 38, 38)
CONTROL_BG_ACTIVE = (45, 45, 45)
PANEL_BG_COLOR = (24, 24, 24)
TEXT_COLOR = (220, 220, 220)
TEXT_DIM_COLOR = (140, 140, 140)
ACCENT_COLOR = (70, 70, 70)
FONT_SIZE = 15

# DearPyGui tag constants
PRIMARY_WINDOW_TAG = "primary_window"
WINDOW_HANDLER_TAG = "window_handler"


def get_system_font_path():
    """Try to find a suitable system font path."""
    system = platform.system()
    
    font_candidates = []
    
    if system == "Linux":
        font_candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/TTF/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
            "/usr/share/fonts/noto/NotoSans-Regular.ttf",
        ]
    elif system == "Darwin":  # macOS
        font_candidates = [
            "/System/Library/Fonts/SFNSText.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "/Library/Fonts/Arial.ttf",
        ]
    elif system == "Windows":
        font_candidates = [
            "C:/Windows/Fonts/segoeui.ttf",
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/calibri.ttf",
        ]
    
    for font_path in font_candidates:
        if os.path.exists(font_path):
            return font_path
    
    return None


def create_dark_theme():
    """Create a minimal global dark theme (zero padding for main window)."""
    with dpg.theme() as theme:
        with dpg.theme_component(dpg.mvAll):
            # Background colors
            dpg.add_theme_color(dpg.mvThemeCol_WindowBg, BACKGROUND_COLOR)
            dpg.add_theme_color(dpg.mvThemeCol_ChildBg, PANEL_BG_COLOR)
            dpg.add_theme_color(dpg.mvThemeCol_PopupBg, (25, 25, 25))
            
            # Text colors
            dpg.add_theme_color(dpg.mvThemeCol_Text, TEXT_COLOR)
            dpg.add_theme_color(dpg.mvThemeCol_TextDisabled, TEXT_DIM_COLOR)
            
            # Control colors
            dpg.add_theme_color(dpg.mvThemeCol_FrameBg, CONTROL_BG_COLOR)
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, CONTROL_BG_HOVER)
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive, CONTROL_BG_ACTIVE)
            
            # Slider
            dpg.add_theme_color(dpg.mvThemeCol_SliderGrab, ACCENT_COLOR)
            dpg.add_theme_color(dpg.mvThemeCol_SliderGrabActive, (90, 90, 90))
            
            # Button
            dpg.add_theme_color(dpg.mvThemeCol_Button, CONTROL_BG_COLOR)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, CONTROL_BG_HOVER)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, CONTROL_BG_ACTIVE)
            
            # Header
            dpg.add_theme_color(dpg.mvThemeCol_Header, CONTROL_BG_COLOR)
            dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered, CONTROL_BG_HOVER)
            dpg.add_theme_color(dpg.mvThemeCol_HeaderActive, CONTROL_BG_ACTIVE)
            
            # Borders
            dpg.add_theme_style(dpg.mvStyleVar_FrameBorderSize, 0)
            dpg.add_theme_style(dpg.mvStyleVar_WindowBorderSize, 0)
            dpg.add_theme_style(dpg.mvStyleVar_PopupBorderSize, 0)
            
            # Rounding
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 4)
            dpg.add_theme_style(dpg.mvStyleVar_PopupRounding, 4)
            
            # Spacing
            dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 8, 8)
            dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 10, 6)
            
            # Global Window Padding (Zero for flush image)
            dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 0, 0)
            
    return theme


def create_panel_theme():
    """Create a specific theme for the sidebar panel (restores padding)."""
    with dpg.theme() as theme:
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 16, 16)
    return theme


def calculate_fit_dimensions(image_w, image_h, container_w, container_h, padding=PADDING):
    """Calculate dimensions to fit image in container while preserving aspect ratio."""
    available_w = container_w - 2 * padding
    available_h = container_h - 2 * padding

    if available_w <= 0 or available_h <= 0 or image_h == 0:
        return 0, 0, 0, 0

    image_aspect = image_w / image_h
    container_aspect = available_w / available_h

    if image_aspect > container_aspect:
        display_w = available_w
        display_h = available_w / image_aspect
    else:
        display_h = available_h
        display_w = available_h * image_aspect

    x_offset = (container_w - display_w) / 2
    y_offset = (container_h - display_h) / 2

    return display_w, display_h, x_offset, y_offset


# Matplotlib Colormaps
MATPLOTLIB_COLORMAPS = {
    'Perceptually Uniform Sequential': [
        'viridis', 'plasma', 'inferno', 'magma', 'cividis'
    ],
    'Sequential': [
        'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
        'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
        'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn'
    ],
    'Sequential (2)': [
        'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
        'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
        'hot', 'afmhot', 'gist_heat', 'copper'
    ],
    'Diverging': [
        'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
        'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic',
        'berlin', 'managua', 'vanimo'
    ],
    'Cyclic': [
        'twilight', 'twilight_shifted', 'hsv'
    ],
    'Qualitative': [
        'Pastel1', 'Pastel2', 'Paired', 'Accent', 'Dark2',
        'Set1', 'Set2', 'Set3', 'tab10', 'tab20', 'tab20b', 'tab20c'
    ],
    'Miscellaneous': [
        'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
        'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg',
        'gist_rainbow', 'rainbow', 'jet', 'turbo', 'nipy_spectral', 'gist_ncar'
    ]
}

COLORMAP_CATEGORIES = list(MATPLOTLIB_COLORMAPS.keys())

# Method Parameters
METHOD_PARAMS = {
    'preset': None,
    'hsv': {
        'label': 'saturation',
        'options': ['Pastel (0.3)', 'Muted (0.5)', 'Medium (0.7)', 'Vibrant (0.9)', 'Full (1.0)'],
        'values': [0.3, 0.5, 0.7, 0.9, 1.0],
        'default_index': 3,
        'has_third_dropdown': False,
    },
    'golden_ratio': {
        'label': 'saturation',
        'options': ['Pastel (0.3)', 'Muted (0.5)', 'Medium (0.7)', 'Vibrant (0.9)', 'Full (1.0)'],
        'values': [0.3, 0.5, 0.7, 0.9, 1.0],
        'default_index': 3,
        'has_third_dropdown': False,
    },
    'kmeans': {
        'label': 'saturation',
        'options': ['Pastel (0.3)', 'Muted (0.5)', 'Medium (0.7)', 'Vibrant (0.9)', 'Full (1.0)'],
        'values': [0.3, 0.5, 0.7, 0.9, 1.0],
        'default_index': 3,
        'has_third_dropdown': False,
    },
    'colormap': {
        'label': 'category',
        'options': COLORMAP_CATEGORIES,
        'values': COLORMAP_CATEGORIES,
        'default_index': 5,
        'has_third_dropdown': True,
        'third_label': 'colormap',
        'third_default_index': 9,
    },
    'colormap_extended': {
        'label': 'category',
        'options': COLORMAP_CATEGORIES,
        'values': COLORMAP_CATEGORIES,
        'default_index': 5,
        'has_third_dropdown': True,
        'third_label': 'colormap',
        'third_default_index': 9,
    },
}

METHOD_LIST = list(METHOD_PARAMS.keys())


class OverlayViewer:
    """Handles the overlay visualization and Dear PyGui interaction."""

    @staticmethod
    def _validate_inputs(img: np.ndarray, label_image: np.ndarray, initial_method: str) -> None:
        """Validate input arguments and raise ValueError with descriptive messages."""
        # Validate img
        if not isinstance(img, np.ndarray):
            raise ValueError(f"img must be a numpy array, got {type(img).__name__}")
        if img.size == 0:
            raise ValueError("img cannot be empty")
        if img.ndim != 3:
            raise ValueError(f"img must be 3-dimensional (H, W, C), got {img.ndim} dimensions")
        if img.shape[2] != 3:
            raise ValueError(f"img must have 3 channels (RGB), got {img.shape[2]} channels")
        
        # Validate label_image
        if not isinstance(label_image, np.ndarray):
            raise ValueError(f"label_image must be a numpy array, got {type(label_image).__name__}")
        if label_image.size == 0:
            raise ValueError("label_image cannot be empty")
        if label_image.ndim != 2:
            raise ValueError(f"label_image must be 2-dimensional (H, W), got {label_image.ndim} dimensions")
        
        # Validate matching dimensions
        if img.shape[:2] != label_image.shape:
            raise ValueError(
                f"img and label_image must have matching spatial dimensions: "
                f"img is {img.shape[:2]}, label_image is {label_image.shape}"
            )
        
        # Validate initial_method
        if initial_method not in METHOD_PARAMS:
            raise ValueError(
                f"initial_method must be one of {list(METHOD_PARAMS.keys())}, got '{initial_method}'"
            )

    @staticmethod
    def _remap_sparse_labels(label_image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Remap sparse label values to a contiguous range [0, N).
        
        This prevents memory issues when label IDs are sparse (e.g., COCO dataset
        where an object might have ID 20000 but only 3 unique labels exist).
        
        Args:
            label_image: 2D array of integer labels (may be sparse).
            
        Returns:
            tuple: (unique_labels, contiguous_labels)
                - unique_labels: Sorted array of unique label values from input
                - contiguous_labels: Label image remapped to range [0, len(unique_labels))
        """
        unique_labels = np.unique(label_image)
        num_unique = len(unique_labels)
        max_label = unique_labels[-1]  # np.unique returns sorted array
        
        # Determine if remapping is needed (sparse if max_label significantly exceeds count)
        is_sparse = (max_label + 1) > (2 * num_unique)
        
        if not is_sparse:
            # Labels are already dense enough; use as-is
            # But still need contiguous mapping for LUT indexing
            if unique_labels[0] == 0 and np.array_equal(unique_labels, np.arange(num_unique)):
                # Already contiguous starting from 0
                return unique_labels, label_image
        
        # Remap to contiguous range using searchsorted (works for any sparsity level)
        # searchsorted finds insertion indices, which equals the contiguous index
        # since unique_labels is sorted
        contiguous_labels = np.searchsorted(unique_labels, label_image).astype(np.int32)
        
        return unique_labels, contiguous_labels

    def __init__(self, img: np.ndarray, label_image: np.ndarray,
                 initial_method: str = 'colormap', initial_saturation: float = None,
                 initial_category: str = None, initial_colormap: str = None):
        
        # Input validation
        self._validate_inputs(img, label_image, initial_method)
        
        self.img = img
        self.h, self.w = img.shape[:2]
        
        # Store original label image
        self.label_image = label_image.astype(np.int32)
        
        # Handle sparse labels by remapping to contiguous range [0, N)
        # This prevents memory issues when max_label >> num_unique_labels
        self._unique_labels, self._contiguous_labels = self._remap_sparse_labels(self.label_image)
        self.num_labels = len(self._unique_labels)
        self.has_labels = self.num_labels > 1

        # Pre-allocate buffers for performance (avoid repeated allocation in blend loop)
        # Pre-normalize image to float32 (0.0-1.0) once
        self._img_float = self.img.astype(np.float32) / 255.0
        # Pre-allocate RGB buffer for cv2.addWeighted (must be contiguous)
        self._rgb_buffer = np.zeros((self.h, self.w, 3), dtype=np.float32)
        # Pre-allocate flat display buffer for DPG texture (RGBA float32)
        # Initialize alpha channel to 1.0 (fully opaque)
        self._display_buffer = np.ones((self.h * self.w * 4,), dtype=np.float32)
        # Create a view for convenient (H, W, 4) access without copying
        self._buffer_view = self._display_buffer.reshape((self.h, self.w, 4))

        # Current state
        self.current_method = initial_method
        self.current_param_label = None
        self.current_param_value = None
        self.current_param_index = 0
        self.current_third_label = None
        self.current_third_value = None
        self.current_third_index = 0
        self.current_alpha = 0.3
        self._lut_float = None

        self._init_param_from_method(initial_method, initial_saturation, initial_category, initial_colormap)
        self._compute_lut()

        # Display settings
        self.max_display_size = 1000
        self.initial_display_w, self.initial_display_h = self._calculate_initial_display_size()

        # Dear PyGui tags
        self.texture_tag = "overlay_texture"
        self.drawlist_tag = "image_drawlist"
        self.sidebar_tag = "controls_sidebar"
        
        self.method_combo_tag = "method_combo"
        self.param_combo_tag = "param_combo"
        self.param_text_tag = "param_text"
        self.param_group_tag = "param_group"
        self.third_combo_tag = "third_combo"
        self.third_text_tag = "third_text"
        self.third_group_tag = "third_group"
        self.alpha_slider_tag = "alpha_slider"

    def _init_param_from_method(self, method: str, initial_saturation=None,
                                 initial_category=None, initial_colormap=None):
        config = METHOD_PARAMS.get(method)
        if config is None:
            self.current_param_label = None
            self.current_param_value = None
            self.current_param_index = 0
            self.current_third_label = None
            self.current_third_value = None
            self.current_third_index = 0
        elif config.get('has_third_dropdown', False):
            self.current_param_label = config['label']
            if initial_category is not None and initial_category in config['values']:
                self.current_param_index = config['values'].index(initial_category)
            else:
                self.current_param_index = config['default_index']
            self.current_param_value = config['values'][self.current_param_index]
            
            self.current_third_label = config['third_label']
            colormaps_in_category = MATPLOTLIB_COLORMAPS[self.current_param_value]
            if initial_colormap is not None and initial_colormap in colormaps_in_category:
                self.current_third_index = colormaps_in_category.index(initial_colormap)
            else:
                self.current_third_index = min(config['third_default_index'], len(colormaps_in_category) - 1)
            self.current_third_value = colormaps_in_category[self.current_third_index]
        else:
            self.current_param_label = config['label']
            if initial_saturation is not None and initial_saturation in config['values']:
                self.current_param_index = config['values'].index(initial_saturation)
            else:
                self.current_param_index = config['default_index']
            self.current_param_value = config['values'][self.current_param_index]
            self.current_third_label = None
            self.current_third_value = None
            self.current_third_index = 0

    def _calculate_initial_display_size(self) -> tuple[int, int]:
        scale = min(self.max_display_size / self.w, self.max_display_size / self.h, 1.0)
        return int(self.w * scale), int(self.h * scale)

    def _redraw_image(self, window_w: int, window_h: int) -> None:
        """Clear drawlist and redraw image at current size with shadow."""
        image_area_w = window_w - CONTROLS_WIDTH
        image_area_h = window_h
        
        if image_area_w <= 0: image_area_w = 10
        if image_area_h <= 0: image_area_h = 10

        display_w, display_h, x_off, y_off = calculate_fit_dimensions(
            self.w, self.h, image_area_w, image_area_h
        )

        dpg.delete_item(self.drawlist_tag, children_only=True)

        if display_w > 0 and display_h > 0:
            shadow_layers = [
                (SHADOW_OFFSET, (0, 0, 0, 25)),
                (SHADOW_OFFSET - 1, (0, 0, 0, 20)),
                (SHADOW_OFFSET - 2, (0, 0, 0, 15)),
            ]
            
            for offset, color in shadow_layers:
                dpg.draw_rectangle(
                    pmin=(x_off + offset, y_off + offset),
                    pmax=(x_off + display_w + offset, y_off + display_h + offset),
                    color=color,
                    fill=color,
                    parent=self.drawlist_tag
                )
            
            dpg.draw_image(
                self.texture_tag,
                pmin=(x_off, y_off),
                pmax=(x_off + display_w, y_off + display_h),
                parent=self.drawlist_tag
            )

    def on_window_resize(self, sender: int, app_data: Any) -> None:
        """Handle window resize - recalculate and redraw."""
        window_width = dpg.get_item_width(PRIMARY_WINDOW_TAG)
        window_height = dpg.get_item_height(PRIMARY_WINDOW_TAG)

        window_width = max(window_width, MIN_WINDOW_WIDTH)
        window_height = max(window_height, MIN_WINDOW_HEIGHT)
        
        image_area_w = max(1, window_width - CONTROLS_WIDTH)
        image_area_h = max(1, window_height)

        dpg.configure_item(self.drawlist_tag, width=image_area_w, height=image_area_h)
        dpg.configure_item(self.sidebar_tag, height=image_area_h)
        self._redraw_image(window_width, window_height)

    def _compute_lut(self) -> None:
        if not self.has_labels:
            self._lut_float = None
            return

        config = METHOD_PARAMS.get(self.current_method)
        kwargs = {}
        if config is not None:
            if config.get('has_third_dropdown', False):
                kwargs['colormap'] = self.current_third_value
            elif self.current_param_label is not None and self.current_param_value is not None:
                kwargs[self.current_param_label] = self.current_param_value

        # Use global generate_colors function
        colors = generate_colors(self.num_labels - 1, method=self.current_method, **kwargs)
        # Store as float32 (0.0-1.0) for efficient blending without repeated conversion
        self._lut_float = np.vstack([[0, 0, 0], colors]).astype(np.float32) / 255.0

    def blend(self, alpha: float) -> np.ndarray:
        """
        Blend the image with the label overlay using pre-allocated buffers.
        
        Returns the internal display buffer directly (no copy) for efficiency.
        """
        if self.has_labels and self._lut_float is not None:
            overlay = self._lut_float[self._contiguous_labels]
            # Blend into contiguous RGB buffer (cv2.addWeighted requires contiguous array)
            cv2.addWeighted(
                self._img_float, 1 - alpha,
                overlay, alpha,
                0.0,
                dst=self._rgb_buffer
            )
        else:
            # No labels - copy the pre-normalized image to RGB buffer
            np.copyto(self._rgb_buffer, self._img_float)

        # Copy RGB buffer to RGBA display buffer view
        self._buffer_view[:, :, :3] = self._rgb_buffer
        # Alpha channel already set to 1.0 at init; no action needed
        
        # Return the flat buffer directly (no copy)
        return self._display_buffer

    def _cycle_combo(self, combo_tag: str, items: list[str], values: list[Any], direction: int, callback: Callable) -> None:
        current_display = dpg.get_value(combo_tag)
        try:
            current_index = items.index(current_display)
        except ValueError:
            current_index = 0
        
        new_index = (current_index + direction) % len(items)
        new_display = items[new_index]
        dpg.set_value(combo_tag, new_display)
        callback(combo_tag, new_display)

    def on_key_press(self, sender: int, app_data: int) -> None:
        key = app_data
        
        if dpg.is_item_hovered(self.alpha_slider_tag):
            step = 0.05
            if key == dpg.mvKey_Right:
                new_value = min(1.0, self.current_alpha + step)
                dpg.set_value(self.alpha_slider_tag, new_value)
                self.on_alpha_change(self.alpha_slider_tag, new_value)
            elif key == dpg.mvKey_Left:
                new_value = max(0.0, self.current_alpha - step)
                dpg.set_value(self.alpha_slider_tag, new_value)
                self.on_alpha_change(self.alpha_slider_tag, new_value)
            return
        
        if dpg.is_item_hovered(self.method_combo_tag):
            direction = 1 if key == dpg.mvKey_Down else (-1 if key == dpg.mvKey_Up else 0)
            if direction:
                self._cycle_combo(self.method_combo_tag, METHOD_LIST, METHOD_LIST, direction, self.on_method_change)
            return

        config = METHOD_PARAMS.get(self.current_method)
        if config is not None and dpg.is_item_hovered(self.param_combo_tag):
            direction = 1 if key == dpg.mvKey_Down else (-1 if key == dpg.mvKey_Up else 0)
            if direction:
                self._cycle_combo(self.param_combo_tag, config['options'], config['values'], direction, self.on_param_change)
            return
        
        if config is not None and config.get('has_third_dropdown', False) and dpg.is_item_hovered(self.third_combo_tag):
            colormaps_in_category = MATPLOTLIB_COLORMAPS[self.current_param_value]
            direction = 1 if key == dpg.mvKey_Down else (-1 if key == dpg.mvKey_Up else 0)
            if direction:
                self._cycle_combo(self.third_combo_tag, colormaps_in_category, colormaps_in_category, direction, self.on_third_change)

    def _refresh_display(self) -> None:
        texture_data = self.blend(self.current_alpha)
        dpg.set_value(self.texture_tag, texture_data)
        try:
            window_width = dpg.get_item_width(PRIMARY_WINDOW_TAG)
            window_height = dpg.get_item_height(PRIMARY_WINDOW_TAG)
            if window_width > 0 and window_height > 0:
                self._redraw_image(window_width, window_height)
        except (SystemError, RuntimeError):
            # Window may not be fully initialized during early callbacks;
            # silently skip redraw until window is ready
            pass

    def _get_blended_pil(self) -> Image.Image:
        """
        Get the blended image as a PIL Image for export.
        
        Called once at close time, so conversion overhead is acceptable.
        """
        if self.has_labels and self._lut_float is not None:
            overlay = self._lut_float[self._contiguous_labels]
            blended_float = cv2.addWeighted(
                self._img_float, 1 - self.current_alpha,
                overlay, self.current_alpha,
                0.0
            )
            # Convert back to uint8 for PIL
            blended = np.clip(blended_float * 255.0, 0, 255).astype(np.uint8)
        else:
            blended = self.img
        return Image.fromarray(blended)

    def get_result(self) -> dict:
        settings = {'alpha': self.current_alpha, 'method': self.current_method}
        config = METHOD_PARAMS.get(self.current_method)
        if config is not None:
            if config.get('has_third_dropdown', False):
                settings['category'] = self.current_param_value
                settings['colormap'] = self.current_third_value
            else:
                settings['saturation'] = self.current_param_value
        return self._get_blended_pil(), settings

    def on_alpha_change(self, sender: int, app_data: float) -> None:
        self.current_alpha = app_data
        self._refresh_display()

    def on_method_change(self, sender: int, app_data: str) -> None:
        self.current_method = app_data
        config = METHOD_PARAMS.get(self.current_method)

        if config is None:
            self.current_param_label = None
            self.current_param_value = None
            dpg.configure_item(self.param_group_tag, show=False)
            dpg.configure_item(self.third_group_tag, show=False)
        elif config.get('has_third_dropdown', False):
            self.current_param_label = config['label']
            default_index = config['default_index']
            self.current_param_value = config['values'][default_index]

            dpg.set_value(self.param_text_tag, config['label'])
            dpg.configure_item(self.param_combo_tag, items=config['options'], default_value=config['options'][default_index])
            dpg.configure_item(self.param_group_tag, show=True)
            
            self.current_third_label = config['third_label']
            colormaps_in_category = MATPLOTLIB_COLORMAPS[self.current_param_value]
            third_idx = min(config['third_default_index'], len(colormaps_in_category) - 1)
            self.current_third_value = colormaps_in_category[third_idx]
            self.current_third_index = third_idx
            
            dpg.set_value(self.third_text_tag, config['third_label'])
            dpg.configure_item(self.third_combo_tag, items=colormaps_in_category, default_value=colormaps_in_category[third_idx])
            dpg.configure_item(self.third_group_tag, show=True)
        else:
            self.current_param_label = config['label']
            default_index = config['default_index']
            self.current_param_value = config['values'][default_index]
            
            dpg.set_value(self.param_text_tag, config['label'])
            dpg.configure_item(self.param_combo_tag, items=config['options'], default_value=config['options'][default_index])
            dpg.configure_item(self.param_group_tag, show=True)
            dpg.configure_item(self.third_group_tag, show=False)

        self._compute_lut()
        self._refresh_display()

    def on_param_change(self, sender: int, app_data: str) -> None:
        config = METHOD_PARAMS.get(self.current_method)
        if config is None: return

        try:
            index = config['options'].index(app_data)
            self.current_param_value = config['values'][index]
            self.current_param_index = index
        except ValueError: return

        if config.get('has_third_dropdown', False):
            colormaps_in_category = MATPLOTLIB_COLORMAPS[self.current_param_value]
            self.current_third_index = 0
            self.current_third_value = colormaps_in_category[0]
            dpg.configure_item(self.third_combo_tag, items=colormaps_in_category, default_value=colormaps_in_category[0])

        self._compute_lut()
        self._refresh_display()

    def on_third_change(self, sender: int, app_data: str) -> None:
        config = METHOD_PARAMS.get(self.current_method)
        if config is None or not config.get('has_third_dropdown', False): return
        
        try:
            colormaps_in_category = MATPLOTLIB_COLORMAPS[self.current_param_value]
            index = colormaps_in_category.index(app_data)
            self.current_third_value = app_data
            self.current_third_index = index
        except ValueError: return

        self._compute_lut()
        self._refresh_display()

    # =========================================================================
    # UI Building Helper Methods
    # =========================================================================

    def _apply_theme(self) -> None:
        """Create and apply the global dark theme."""
        theme = create_dark_theme()
        dpg.bind_theme(theme)

    def _setup_fonts(self) -> None:
        """Load system font if available."""
        font_path = get_system_font_path()
        if font_path:
            with dpg.font_registry():
                default_font = dpg.add_font(font_path, FONT_SIZE)
            dpg.bind_font(default_font)

    def _create_texture(self, initial_alpha: float) -> None:
        """Create the dynamic texture for the overlay display."""
        initial_texture = self.blend(initial_alpha)
        with dpg.texture_registry():
            dpg.add_dynamic_texture(
                width=self.w,
                height=self.h,
                default_value=initial_texture,
                tag=self.texture_tag
            )

    def _build_alpha_slider(self, initial_alpha: float, width: int) -> None:
        """Build the alpha control slider."""
        dpg.add_text("alpha", color=TEXT_DIM_COLOR)
        dpg.add_slider_float(
            label="",
            default_value=initial_alpha,
            min_value=0.0, max_value=1.0,
            callback=self.on_alpha_change,
            width=width,
            tag=self.alpha_slider_tag
        )

    def _build_method_combo(self, width: int) -> None:
        """Build the method selection combo box."""
        dpg.add_text("method", color=TEXT_DIM_COLOR)
        dpg.add_combo(
            items=METHOD_LIST,
            default_value=self.current_method,
            label="",
            callback=self.on_method_change,
            tag=self.method_combo_tag,
            width=width
        )
        dpg.add_spacer(height=5)

    def _build_param_combo(self, initial_config: dict | None, show_param: bool,
                           width: int) -> None:
        """Build the parameter combo box (saturation/category)."""
        if show_param:
            param_label = initial_config['label']
            param_options = initial_config['options']
            param_default = param_options[self.current_param_index]
        else:
            param_label = ""
            param_options = []
            param_default = ""

        with dpg.group(horizontal=False, tag=self.param_group_tag, show=show_param):
            dpg.add_text(param_label, tag=self.param_text_tag, color=TEXT_DIM_COLOR)
            dpg.add_combo(
                items=param_options,
                default_value=param_default,
                callback=self.on_param_change,
                tag=self.param_combo_tag,
                width=width,
            )
        dpg.add_spacer(height=5)

    def _build_third_combo(self, initial_config: dict | None, width: int) -> None:
        """Build the third combo box (colormap selection)."""
        show_third = (initial_config is not None and 
                      initial_config.get('has_third_dropdown', False))
        
        if show_third:
            third_label = initial_config['third_label']
            colormaps_in_category = MATPLOTLIB_COLORMAPS[self.current_param_value]
            third_default = colormaps_in_category[self.current_third_index]
        else:
            third_label = ""
            colormaps_in_category = []
            third_default = ""

        with dpg.group(horizontal=False, tag=self.third_group_tag, show=show_third):
            dpg.add_text(third_label, tag=self.third_text_tag, color=TEXT_DIM_COLOR)
            dpg.add_combo(
                items=colormaps_in_category,
                default_value=third_default,
                callback=self.on_third_change,
                tag=self.third_combo_tag,
                width=width,
            )

    def _build_control_panel(self, initial_alpha: float, initial_config: dict | None,
                             show_param: bool, content_width: int) -> None:
        """Build the control panel with all widgets."""
        with dpg.child_window(tag=self.sidebar_tag, width=CONTROLS_WIDTH, border=False):
            dpg.add_text("CONTROLS", color=TEXT_DIM_COLOR)
            dpg.add_spacer(height=10)

            self._build_alpha_slider(initial_alpha, content_width)

            dpg.add_separator()
            dpg.add_spacer(height=10)

            self._build_method_combo(content_width)
            self._build_param_combo(initial_config, show_param, content_width)
            self._build_third_combo(initial_config, content_width)

            dpg.add_spacer(height=20)
            dpg.add_separator()
            dpg.add_spacer(height=10)

    def _build_ui(self, initial_alpha: float) -> tuple[int, int]:
        """
        Build the main window and all UI components.
        
        Returns:
            tuple: (window_width, window_height) for viewport configuration.
        """
        initial_config = METHOD_PARAMS.get(self.current_method)
        show_param = initial_config is not None

        # Sizing calculations
        image_area_w = self.initial_display_w + 40
        image_area_h = self.initial_display_h + 40
        window_w = image_area_w + CONTROLS_WIDTH
        window_h = image_area_h
        control_content_width = CONTROLS_WIDTH - 40

        with dpg.window(label="Overlay Viewer", tag=PRIMARY_WINDOW_TAG):
            with dpg.group(horizontal=True):
                # LEFT: Image Area (Drawlist)
                dpg.add_drawlist(
                    tag=self.drawlist_tag,
                    width=image_area_w,
                    height=image_area_h
                )
                # RIGHT: Control Panel
                self._build_control_panel(initial_alpha, initial_config,
                                          show_param, control_content_width)

        # Apply panel theme to sidebar
        panel_theme = create_panel_theme()
        dpg.bind_item_theme(self.sidebar_tag, panel_theme)

        return window_w, window_h

    def _register_handlers(self) -> None:
        """Register keyboard and window resize handlers."""
        # Keyboard handlers
        with dpg.handler_registry():
            dpg.add_key_press_handler(dpg.mvKey_Up, callback=self.on_key_press)
            dpg.add_key_press_handler(dpg.mvKey_Down, callback=self.on_key_press)
            dpg.add_key_press_handler(dpg.mvKey_Left, callback=self.on_key_press)
            dpg.add_key_press_handler(dpg.mvKey_Right, callback=self.on_key_press)

        # Window resize handler
        with dpg.item_handler_registry(tag=WINDOW_HANDLER_TAG):
            dpg.add_item_resize_handler(callback=self.on_window_resize)
        dpg.bind_item_handler_registry(PRIMARY_WINDOW_TAG, WINDOW_HANDLER_TAG)

    def _configure_viewport(self, window_w: int, window_h: int) -> None:
        """Configure and display the viewport."""
        dpg.create_viewport(
            title="Mask Overlay Viewer",
            width=window_w,
            height=window_h,
            min_width=MIN_WINDOW_WIDTH,
            min_height=MIN_WINDOW_HEIGHT
        )
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window(PRIMARY_WINDOW_TAG, True)

        # Initial draw
        self._redraw_image(window_w, window_h)

    # =========================================================================
    # Main Entry Point
    # =========================================================================

    def run(self, initial_alpha: float = 0.3) -> tuple[Image.Image, dict]:
        """
        Launch the overlay viewer GUI.
        
        Args:
            initial_alpha: Starting alpha blend value (0.0-1.0).
            
        Returns:
            tuple: (blended_PIL_image, settings_dict) with final user selections.
        """
        self.current_alpha = initial_alpha
        dpg.create_context()

        try:
            self._apply_theme()
            self._setup_fonts()
            self._create_texture(initial_alpha)
            window_w, window_h = self._build_ui(initial_alpha)
            self._register_handlers()
            self._configure_viewport(window_w, window_h)

            dpg.start_dearpygui()
            return self.get_result()
        finally:
            dpg.destroy_context()


def run_overlay_viewer(img: np.ndarray, label_image: np.ndarray,
                       initial_method: str = 'colormap',
                       initial_saturation: float = None,
                       initial_category: str = None,
                       initial_colormap: str = None,
                       initial_alpha: float = 0.3) -> tuple[Image.Image, dict]:
    # CHANGED: Passed label_image directly to constructor
    viewer = OverlayViewer(img, label_image, initial_method,
                           initial_saturation, initial_category, initial_colormap)
    return viewer.run(initial_alpha)
