import cv2
import logging
from typing import Dict, List, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .color_gen import generate_colors
from .common import BlendConfig, LabelStyle, TitleConfig
from .compose import add_title


logger = logging.getLogger(__name__)

# ── Private Helpers ─────────────────────────────────────────────────────────────

def _ensure_rgba(img: Image.Image) -> Image.Image:
    """Convert PIL image to RGBA mode if needed."""
    return img.convert('RGBA') if img.mode != 'RGBA' else img

def _load_font(name: Optional[str], size: int) -> ImageFont.FreeTypeFont:
    """
    Robust font loader.
    1. Tries the requested font (e.g., 'times.ttf').
    2. Tries system fallbacks (Linux/Windows standards).
    3. Falls back to PIL default bitmap (last resort).
    """
    # 1. Try requested
    if name:
        try:
            return ImageFont.truetype(name, size)
        except (OSError, IOError):
            pass  # Fail silently and try fallbacks

    # 2. Try common cross-platform standards
    fallbacks = ["arial.ttf", "DejaVuSans.ttf", "FreeSans.ttf", "LiberationSans-Regular.ttf"]
    for font in fallbacks:
        try:
            return ImageFont.truetype(font, size)
        except (OSError, IOError):
            continue

    # 3. Last resort: Bitmap default
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        # Older Pillow versions don't support size in load_default
        return ImageFont.load_default()

# ── Annotate tools ─────────────────────────────────────────────────────────────

def draw_labels(
    img: Image.Image,
    labels: List[Dict],
    style: Optional[LabelStyle] = None
) -> Image.Image:
    """Add text annotations to an image."""
    if not isinstance(img, Image.Image):
        raise TypeError(f'img must be a PIL Image, got {type(img)}')
    
    style = style or LabelStyle()
    img = _ensure_rgba(img)
    font = _load_font(style.font_name, style.font_size)
    
    if style.show_boxes:
        overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        measure_draw = ImageDraw.Draw(img)
        
        for label in labels:
            bbox = measure_draw.textbbox(
                (label['x'], label['y']),
                label['text'],
                font=font,
                anchor='mm'
            )
            # Add padding
            box_coords = [
                bbox[0] - style.padding,
                bbox[1] - style.padding,
                bbox[2] + style.padding,
                bbox[3] + style.padding,
            ]
            overlay_draw.rectangle(
                box_coords,
                fill=(*style.box_fill, style.alpha),
                outline=style.box_outline,
                width=style.box_outline_width
            )
        
        img = Image.alpha_composite(img, overlay)
    
    draw = ImageDraw.Draw(img)
    for label in labels:
        draw.text(
            (label['x'], label['y']),
            label['text'],
            fill=style.text_color,
            font=font,
            anchor='mm'
        )
    
    return img


def create_label_overlay_from_labelimg(img, label_image, config: BlendConfig = None, **kwargs):
    """
    Generates a color overlay from a binary mask and blends it with an image.
    
    Args:
        img (np.ndarray): The base image (H, W, 3).
        label_image (np.ndarray): Label image with integer labels for each component.
        config (BlendConfig): Configuration object (alternative to individual params).
        **kwargs: alpha, method, and additional arguments passed to generate_colors.
    """
    # Use BlendConfig as single source of defaults
    if config is None:
        config = BlendConfig.from_params(**kwargs)
    
    num_labels = label_image.max() + 1
    colors = generate_colors(num_labels - 1, method=config.method, **config.params)
    
    lut = np.vstack([[0, 0, 0], colors]).astype(np.uint8)
    overlay = lut[label_image]
    blended = cv2.addWeighted(img, 1 - config.alpha, overlay, config.alpha, 0)
    
    return Image.fromarray(blended)


def overlay_visualize(img, 
                          label_image, 
                          annotations=None, 
                          blend_config=None, 
                          label_style=None, 
                          title_config=None, 
                          title=None, 
                          savepath=None,
                          show=False):
    """    
    Args:
        img: Source image (numpy array or PIL Image).
        label_image: Label mask (numpy array or PIL Image).
        annotations: List of annotations (optional).
        blend_config, label_style, title_config: Config objects (optional).
        title: Title text string (optional).
        savepath: Path to save the final image (optional).
        show: If True, opens the image in the default system viewer (optional).
        
    Returns:
        The resulting PIL Image object.
    """

    # --- 1. Setup Defaults ---
    if blend_config is None:
        blend_config = BlendConfig.from_params(
            alpha=0.6, method='colormap', colormap='viridis'
        )

    if label_style is None:
        label_style = LabelStyle(
            font_size=14, padding=4, alpha=160, show_boxes=True, 
            text_color='black', box_fill=(255, 255, 255), 
            box_outline='black', box_outline_width=2, font_name='arial.ttf'
        )

    if title_config is None:
        title_config = TitleConfig(
            font_path="DejaVuSans-Bold.ttf", font_size=12, line_spacing=2,
            text_color=(0, 0, 0), bg_color=(90, 90, 90), 
            padding=15, align='left'
        )

    # --- 2. Input Normalization (PIL -> Numpy BGR) ---
    # Convert PIL inputs to Numpy (BGR) for OpenCV processing
    if isinstance(img, Image.Image):
        img = np.array(img)
        # PIL is RGB, OpenCV expects BGR
        if img.ndim == 3 and img.shape[2] == 3:
             img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if isinstance(label_image, Image.Image):
        label_image = np.array(label_image)

    # --- 3. Shape Validation ---
    h_label, w_label = label_image.shape[:2]
    h_img, w_img = img.shape[:2]

    if (h_img != h_label) or (w_img != w_label):
        print(f"Warning: Resizing image from {(w_img, h_img)} to {(w_label, h_label)}.")
        img = cv2.resize(img, (w_label, h_label))

    # --- 4. Generate Overlay (OpenCV pipeline) ---
    current_img = create_label_overlay_from_labelimg(
        img, label_image, config=blend_config
    )
    
    if annotations: 
        current_img = draw_labels(
            current_img, annotations, label_style
        )
    
    if title is not None:
        current_img = add_title(
            current_img, title, title_config
        )
    
    # --- 5. Convert to PIL Image (BGR -> RGB) ---
    if isinstance(current_img, np.ndarray):
        # Convert BGR to RGB for correct color display in PIL
        if current_img.ndim == 3 and current_img.shape[2] == 3:
            current_img = cv2.cvtColor(current_img, cv2.COLOR_BGR2RGB)
        
        current_img = Image.fromarray(current_img)

    # --- 6. Output Handling ---
    if savepath:
        current_img.save(savepath)
        
    if show:
        current_img.show()
        
    return current_img
