"""High-level visualization pipeline for the imtools package.

Provides :func:`overlay_visualize` (main entry point), helpers for
generating colored label overlays, and a label drawing utility.
"""

from __future__ import annotations

import cv2
import logging
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from imtools.viz.colors import generate_colors
from imtools.core.types import BlendConfig, LabelStyle, TitleConfig
from imtools.viz.compose import add_title


logger = logging.getLogger(__name__)

# ── Private Helpers ─────────────────────────────────────────────────────────────

def _ensure_rgba(img: Image.Image) -> Image.Image:
    """Return ``img`` converted to RGBA mode; no-op if already RGBA."""
    return img.convert('RGBA') if img.mode != 'RGBA' else img

def _load_font(name: Optional[str], size: int) -> Any:
    """Load a TrueType font with graceful fallback.

    Attempts fonts in this order:
    1. The requested ``name`` (e.g. ``'times.ttf'``).
    2. Common cross-platform fonts: ``arial.ttf``, ``DejaVuSans.ttf``,
       ``FreeSans.ttf``, ``LiberationSans-Regular.ttf``.
    3. PIL's built-in default bitmap font (last resort).
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
    style: Optional[LabelStyle] = None,
) -> Image.Image:
    """Draw text annotations onto a PIL image.

    Optionally renders a semi-transparent filled rectangle behind each label
    for legibility.  All drawing happens on an RGBA overlay that is composited
    onto the source image.

    Args:
        img: Source PIL image.  Converted to RGBA internally if necessary.
        labels: List of annotation dictionaries with keys:

            - ``'text'`` (:class:`str`) — label string.
            - ``'x'`` (:class:`int`) — centroid x-coordinate.
            - ``'y'`` (:class:`int`) — centroid y-coordinate.
        style: :class:`~imtools.core.types.LabelStyle` configuration.
            Defaults to :class:`~imtools.core.types.LabelStyle` with
            standard settings if ``None``.

    Returns:
        New PIL image (RGBA) with labels drawn at the specified positions.

    Raises:
        TypeError: If ``img`` is not a PIL :class:`~PIL.Image.Image`.

    Example:
        >>> from PIL import Image
        >>> img = Image.new('RGB', (200, 200), 'white')
        >>> annotations = [{'text': 'cat 0.9', 'x': 100, 'y': 100}]
        >>> result = draw_labels(img, annotations)
    """
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


def create_label_overlay_from_labelimg(
    img: np.ndarray,
    label_image: np.ndarray,
    config: Optional[BlendConfig] = None,
    **kwargs: Any,
) -> Image.Image:
    """Generate a blended color overlay from a label image.

    Assigns a unique color to each integer label in ``label_image`` using
    :func:`~imtools.viz.colors.generate_colors`, builds a look-up table,
    and blends the colored overlay with the source image using
    :func:`cv2.addWeighted`.

    Args:
        img: Base BGR image of shape ``(H, W, 3)`` and dtype ``uint8``.
        label_image: 2-D integer label map of shape ``(H, W)`` where ``0``
            is background and positive integers identify regions.
        config: :class:`~imtools.core.types.BlendConfig` specifying the
            alpha factor and color-generation method.  If ``None``, a
            default config is built from any remaining ``**kwargs``.
        **kwargs: Forwarded to :class:`~imtools.core.types.BlendConfig.from_params`
            when ``config`` is ``None`` (e.g. ``alpha=0.4``, ``method='hsv'``).

    Returns:
        Blended PIL RGB image of the same spatial size as ``img``.

    Example:
        >>> import numpy as np
        >>> img = np.zeros((64, 64, 3), dtype=np.uint8)
        >>> lbl = np.zeros((64, 64), dtype=np.int32)
        >>> lbl[10:30, 10:30] = 1
        >>> overlay = create_label_overlay_from_labelimg(img, lbl)
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


def overlay_visualize(
    img: np.ndarray | Image.Image,
    label_image: np.ndarray | Image.Image,
    annotations: Optional[List[Dict]] = None,
    blend_config: Optional[BlendConfig] = None,
    label_style: Optional[LabelStyle] = None,
    title_config: Optional[TitleConfig] = None,
    title: Optional[str] = None,
    savepath: Optional[str] = None,
    show: bool = False,
) -> Image.Image:
    """Compose a label overlay, optional text annotations, and an optional title.

    This is the main high-level entry point for the visualization pipeline:

    1. Apply default configs if none are supplied.
    2. Normalise inputs (PIL → NumPy BGR, dimension mismatch → resize).
    3. Generate the colored label overlay via
       :func:`create_label_overlay_from_labelimg`.
    4. Optionally draw text annotations with :func:`draw_labels`.
    5. Optionally prepend a title bar with
       :func:`~imtools.viz.compose.add_title`.
    6. Optionally save and/or display the result.

    Args:
        img: Source image as a NumPy BGR array or PIL Image.
        label_image: Integer label map as a NumPy array or PIL Image.
        annotations: Optional list of annotation dicts (``'text'``, ``'x'``,
            ``'y'`` keys) produced by
            :func:`~imtools.annotations.parsers.yolo_to_annotations` or
            :func:`~imtools.annotations.parsers.label_image_to_annotations`.
        blend_config: Blend style.  Defaults to viridis colormap at
            ``alpha=0.6``.
        label_style: Text label style.  Defaults to a readable preset.
        title_config: Title bar style.  Defaults to a gray-background preset.
        title: Title string.  Pass ``None`` (default) to omit the title bar.
        savepath: File path to save the result.  Skipped if ``None``.
        show: If ``True``, open the image in the OS default viewer.

    Returns:
        Final composed PIL Image (RGB or RGBA mode).

    Example:
        >>> import numpy as np
        >>> img = np.zeros((128, 128, 3), dtype=np.uint8)
        >>> lbl = np.zeros((128, 128), dtype=np.int32)
        >>> lbl[30:90, 30:90] = 1
        >>> result = overlay_visualize(img, lbl, title='My segmentation')
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
