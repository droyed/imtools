

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import List, Optional, Union
from .read_write import load_image


def draw_mask_overlays(image, masks, scores=None, output_path=None, title_prefix=None):
    """
    Draw mask overlays on an image and optionally save to file.

    Args:
        image: PIL Image or array
        masks: Tensor of segmentation masks
        scores: Optional tensor of confidence scores. If provided, used for labeling.
        output_path: Optional path to save the output image. If None, plot is not saved or closed.
        title_prefix: Optional string to prepend to the title. If provided, creates a two-line title.
    """
    fig, ax = plt.subplots(1, figsize=(12, 9))
    image = load_image(image, convert_to_rgb=True)
    ax.imshow(image)

    # Create a color map for different masks
    colors = plt.cm.rainbow(np.linspace(0, 1, len(masks)))

    # Initialize scores as list of None if not provided
    if scores is None:
        scores = [None] * len(masks)

    # Overlay each mask with a different color
    for i, (mask, score, color) in enumerate(zip(masks, scores, colors)):
        # Normalize mask           
        mask_np = load_image(mask)

        # Create a colored overlay
        colored_mask = np.zeros((*mask_np.shape, 4))
        colored_mask[mask_np > 0] = [*color[:3], 0.5]  # RGB + alpha

        ax.imshow(colored_mask)

        # Add label with index and score (if available)
        y_coords, x_coords = np.where(mask_np > 0)
        if len(y_coords) > 0:
            center_y, center_x = y_coords.mean(), x_coords.mean()
            if score is not None:
                label_text = f'#{i+1}\n{score:.2f}'
            else:
                label_text = f'#{i+1}'
            ax.text(center_x, center_y, label_text,
                   bbox=dict(facecolor='white', alpha=0.7, edgecolor=color, linewidth=2),
                   fontsize=10, color='black', ha='center', va='center',
                   weight='bold')

    ax.axis('off')

    # Create title with optional prefix
    if title_prefix is not None:
        title = f'{title_prefix}\nMask Overlays (found: {len(masks)})'
    else:
        title = f'Mask Overlays (found: {len(masks)})'
    plt.title(title)

    # Save the mask overlay debug image if output_path is provided
    if output_path is not None:
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        print(f"Mask overlay debug image saved to: {output_path}")
        plt.close()
    else:
        plt.show()

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
        text = [text]

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


