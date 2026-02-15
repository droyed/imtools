from PIL import Image, ImageDraw, ImageFont
from typing import List, Optional, Union

from .formats import to_pil_image
from .common import TitleConfig


def text_on_canvas(
    text: Union[str, List[str]],
    padding: int = 20,
    min_width: Optional[int] = None,
    bg_color=(30, 30, 30, 255),
    text_color=(255, 255, 255),
    text_opacity: int = 255,
    font_path="DejaVuSans.ttf",
    font_size: int = 40,
    line_spacing: int = 10,
    align: str = "left",        # "left", "center", "right"
    crop_to_text: bool = False,
    transparent_text_only: bool = False,
    stroke_width: int = 0,
    stroke_color: Optional[tuple] = None,
    output_rgb: bool = False,
):
    """
    Draw an auto-sized canvas based on text metrics (no wrapping).

    Features:
    - Width grows to fit the widest line
    - Optional minimum width
    - Optional crop tightly to text bounds
    - Optional text-only transparent output
    """

    # Convert string to list if necessary
    if isinstance(text, str):
        text = text.split('\n')

    assert align in {"left", "center", "right"}
    assert 0 <= text_opacity <= 255

    font = ImageFont.truetype(font_path, font_size)
    ascent, descent = font.getmetrics()
    line_advance = ascent + descent

    # Temporary draw context for measurement
    tmp_img = Image.new("RGBA", (1, 1))
    tmp_draw = ImageDraw.Draw(tmp_img)

    fill = (*text_color, text_opacity)

    # -----------------
    # Measure text
    # -----------------
    line_metrics = []
    max_line_width = 0

    for line in text:
        bbox = tmp_draw.textbbox((0, 0), line, font=font, stroke_width=stroke_width)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        line_metrics.append((w, h))
        max_line_width = max(max_line_width, w)

    # -----------------
    # Compute canvas size
    # -----------------
    content_width = max_line_width + padding * 2
    canvas_width = max(content_width, min_width) if min_width else content_width

    total_height = sum(h for _, h in line_metrics)
    total_height += line_spacing * (len(line_metrics) - 1)
    canvas_height = total_height + padding * 2 + descent

    # -----------------
    # Create layers
    # -----------------
    if transparent_text_only:
        base_image = Image.new("RGBA", (canvas_width, canvas_height), (0, 0, 0, 0))
    else:
        base_image = Image.new("RGBA", (canvas_width, canvas_height), bg_color)

    text_layer = Image.new("RGBA", (canvas_width, canvas_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(text_layer)

    # -----------------
    # Draw text
    # -----------------
    y = padding
    min_x, max_x = canvas_width, 0
    min_y, max_y = y, y

    for (line, (line_w, line_h)) in zip(text, line_metrics):

        if align == "left":
            x = padding
        elif align == "center":
            x = (canvas_width - line_w) // 2
        elif align == "right":
            x = canvas_width - padding - line_w

        draw.text((x, y), line, font=font, fill=fill, stroke_width=stroke_width, stroke_fill=stroke_color)

        min_x = min(min_x, x)
        max_x = max(max_x, x + line_w)
        max_y = max(max_y, y + line_advance)

        y += line_h + line_spacing

    # Composite
    result = Image.alpha_composite(base_image, text_layer)

    # -----------------
    # Crop to text
    # -----------------
    if crop_to_text:
        margin = max(2, descent // 2)
        left = max(int(min_x) - margin, 0)
        upper = max(int(min_y) - margin, 0)
        right = min(int(max_x) + margin, canvas_width)
        lower = min(int(max_y) + margin, canvas_height)
        result = result.crop((left, upper, right, lower))

    # -----------------
    # Convert to RGB if requested
    # -----------------
    if output_rgb:
        # Flatten alpha onto background if needed
        if result.mode == "RGBA":
            # If background is transparent, choose how to flatten
            background = Image.new("RGB", result.size, bg_color[:3])
            background.paste(result, mask=result.split()[3])
            return background
        return result.convert("RGB")

    return result


def stack_images(images, align="start", mode="RGBA", direction="vertical"):
    """
    Stack multiple images vertically or horizontally.
    
    Args:
        images: List of image sources to stack. Inputs can be:
            - PIL.Image.Image: Passed through unchanged
            - numpy.ndarray: Converted using opencv_to_pil
            - str or Path: Loaded from file path
        align: Alignment option:
               - "start": Align to start (left for vertical, top for horizontal)
               - "end": Align to end (right for vertical, bottom for horizontal)
               - "center": Center alignment
               - "resize" or "resize_to_max": Resize all images to match the largest dimension
               - "resize_to_min": Resize all images to match the smallest dimension
               - "resize_to_mean": Resize all images to match the mean dimension
        mode: Image mode for the output canvas. Options:
              - "auto": Automatically detect from input images (uses RGBA if any has alpha)
              - "1": 1-bit pixels, black and white
              - "L": 8-bit pixels, grayscale
              - "P": 8-bit pixels, palette-mapped
              - "RGB": 3x8-bit pixels, true color
              - "RGBA": 4x8-bit pixels, true color with transparency
              - "CMYK": 4x8-bit pixels, color separation
              - "YCbCr": 3x8-bit pixels, color video format
              - "LAB": 3x8-bit pixels, L*a*b color space
              - "HSV": 3x8-bit pixels, Hue, Saturation, Value
              - "I": 32-bit signed integer pixels
              - "F": 32-bit floating point pixels
        direction: Stacking direction - "vertical" (or "v") or "horizontal" (or "h")
    
    Returns:
        Combined Image.Image object
    """
    # Input validation
    if not isinstance(images, (list, tuple)) or len(images) < 2:
        raise ValueError("images must be a list or tuple with at least 2 images")
    
    valid_alignments = {"start", "end", "center", "resize", "resize_to_max", "resize_to_min", "resize_to_mean"}
    if align not in valid_alignments:
        raise ValueError(f"Invalid align value '{align}'. Must be one of: {valid_alignments}")
    
    valid_modes = {"auto", "1", "L", "P", "RGB", "RGBA", "CMYK", "YCbCr", "LAB", "HSV", "I", "F"}
    if mode not in valid_modes:
        raise ValueError(f"Invalid mode value '{mode}'. Must be one of: {valid_modes}")
    
    valid_directions = {"vertical", "v", "horizontal", "h"}
    if direction not in valid_directions:
        raise ValueError(f"Invalid direction value '{direction}'. Must be one of: {valid_directions}")
    
    is_vertical = direction in {"vertical", "v"}
    resize_options = {"resize", "resize_to_max", "resize_to_min", "resize_to_mean"}

    # Convert images to PIL images
    images = [to_pil_image(img) for img in images]
    
    # Collect relevant dimensions
    if is_vertical:
        dimensions = [img.width for img in images]
    else:
        dimensions = [img.height for img in images]
    
    # Calculate target dimension for resize options
    if align in resize_options:
        if align in {"resize", "resize_to_max"}:
            target_dimension = max(dimensions)
        elif align == "resize_to_min":
            target_dimension = min(dimensions)
        elif align == "resize_to_mean":
            target_dimension = int(sum(dimensions) / len(dimensions))
    
    # Handle resize options (pre-processing)
    if align in resize_options:
        processed_images = []
        for img in images:
            if is_vertical and img.width != target_dimension:
                new_height = int(img.height * (target_dimension / img.width))
                img = img.resize((target_dimension, new_height), Image.LANCZOS)
            elif not is_vertical and img.height != target_dimension:
                new_width = int(img.width * (target_dimension / img.height))
                img = img.resize((new_width, target_dimension), Image.LANCZOS)
            processed_images.append(img)
    else:
        processed_images = list(images)
    
    # Determine output mode
    if mode == "auto":
        has_alpha = any(img.mode == "RGBA" for img in processed_images)
        output_mode = "RGBA" if has_alpha else processed_images[0].mode
    else:
        output_mode = mode
    
    # Calculate canvas dimensions
    if is_vertical:
        canvas_width = max(img.width for img in processed_images)
        canvas_height = sum(img.height for img in processed_images)
    else:
        canvas_width = sum(img.width for img in processed_images)
        canvas_height = max(img.height for img in processed_images)
    
    # Calculate positions for each image
    positions = []
    current_pos = 0
    for img in processed_images:
        if is_vertical:
            if align in resize_options or align == "start":
                x = 0
            elif align == "end":
                x = canvas_width - img.width
            elif align == "center":
                x = (canvas_width - img.width) // 2
            positions.append((x, current_pos))
            current_pos += img.height
        else:
            if align in resize_options or align == "start":
                y = 0
            elif align == "end":
                y = canvas_height - img.height
            elif align == "center":
                y = (canvas_height - img.height) // 2
            positions.append((current_pos, y))
            current_pos += img.width
    
    # Create and compose final image
    stacked = Image.new(output_mode, (canvas_width, canvas_height))
    for img, pos in zip(processed_images, positions):
        stacked.paste(img, pos)
    
    return stacked


def add_title(
    image: Image.Image,
    title: str,
    config: Optional[TitleConfig] = None,
) -> Image.Image:
    """
    Composite a title bar above a rendered image.

    Parameters
    ----------
    image : PIL.Image.Image
        The image to add the title to.
    title : str
        Title text.
    config : TitleConfig, optional
        Title styling.  Defaults are used if not supplied.

    Returns
    -------
    PIL.Image.Image
        A new image with the title bar stacked on top.
    """
    config = config or TitleConfig()

    title_frame = text_on_canvas(
        title,
        padding=config.padding,
        font_size=config.font_size,
        text_color=config.text_color,
        bg_color=config.bg_color,
        align=config.align,
        min_width=image.size[0],
        line_spacing=config.line_spacing,
        font_path=config.font_path,
    )

    return stack_images([title_frame, image], "resize_to_max", direction="v")

