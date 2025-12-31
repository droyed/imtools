# imtools

A comprehensive Python package for image loading, manipulation, drawing, and mask processing.

## Features

- **Image Loading**: Load images from file paths, PIL Images, NumPy arrays, or PyTorch tensors
- **Image Drawing**: Draw mask overlays and text on canvas with extensive customization
- **Mask Utilities**: Extract largest blobs, compute bounding boxes, and fill holes in binary masks
- **Image Layout**: Stack images horizontally or vertically with automatic resizing
- **Format Conversion**: Flexible RGB/RGBA/grayscale conversion with normalization
- **Type Support**: Multiple data types with optional uint8 conversion

## Installation

### From GitHub

Install directly from GitHub:

```bash
pip install git+https://github.com/droyed/imtools.git
```

### Local Installation

Install the package in development mode:

```bash
pip install -e .
```

Or install normally:

```bash
pip install .
```

Install with development dependencies:

```bash
pip install -e ".[dev]"
```

## Usage

### Image Loading (`load_image`)

Load images from various sources with flexible format conversion.

```python
from imtools import load_image
import numpy as np
import torch
from PIL import Image

# Load from file path
img = load_image("path/to/image.png")

# Load and convert to RGB
img = load_image("path/to/image.png", convert_to_rgb=True, normalize_dims=True)

# Load from PIL Image
pil_img = Image.open("path/to/image.png")
img = load_image(pil_img)

# Load from NumPy array
array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
img = load_image(array)

# Load from PyTorch tensor
tensor = torch.randint(0, 255, (3, 100, 100), dtype=torch.uint8)
img = load_image(tensor, normalize_dims=True)  # Converts CHW to HWC

# Force uint8 output
img = load_image("image.png", force_uint8=True)

# Drop alpha channel
img = load_image("image_with_alpha.png", drop_alpha=True)
```

**Parameters:**
- `image`: Input image (file path, PIL Image, NumPy array, or PyTorch tensor)
- `drop_alpha`: Drop alpha channel if present (default: False)
- `normalize_dims`: Normalize shape to (H, W, C) format (default: False)
- `force_uint8`: Convert output to uint8 dtype (default: False)
- `convert_to_rgb`: Convert output to RGB with exactly 3 channels (default: False)

### Drawing Functions

#### Draw Mask Overlays (`draw_mask_overlays`)

Visualize segmentation masks with colored overlays and scores.

```python
from imtools import draw_mask_overlays
import numpy as np

# Create sample masks
masks = [
    np.random.randint(0, 2, (100, 100), dtype=np.uint8),
    np.random.randint(0, 2, (100, 100), dtype=np.uint8),
]
scores = [0.95, 0.87]  # Optional confidence scores

# Draw overlays and display
draw_mask_overlays("image.png", masks, scores=scores)

# Save to file with custom title
draw_mask_overlays(
    "image.png",
    masks,
    scores=scores,
    output_path="output.png",
    title_prefix="Segmentation Results"
)
```

**Parameters:**
- `image`: PIL Image or array
- `masks`: List of segmentation masks
- `scores`: Optional list of confidence scores for labeling
- `output_path`: Optional path to save the output image
- `title_prefix`: Optional string to prepend to the title

#### Text on Canvas (`text_on_canvas`)

Create auto-sized text images with extensive customization options.

```python
from imtools import text_on_canvas

# Simple text
img = text_on_canvas("Hello World")

# Multiple lines with custom styling
img = text_on_canvas(
    ["Line 1", "Line 2", "Line 3"],
    font_size=50,
    text_color=(255, 255, 0),
    bg_color=(0, 0, 128, 255),
    align="center"
)

# Text with stroke/outline
img = text_on_canvas(
    "Bold Text",
    stroke_width=3,
    stroke_color=(0, 0, 0, 255),
    text_color=(255, 255, 255)
)

# Cropped to text bounds with transparent background
img = text_on_canvas(
    "Compact",
    crop_to_text=True,
    transparent_text_only=True
)

# RGB output (no alpha channel)
img = text_on_canvas(
    "RGB Image",
    output_rgb=True,
    bg_color=(255, 255, 255, 255)
)

# Single string input (automatically converted to list)
img = text_on_canvas("Single line text")
```

**Parameters:**
- `text`: String or list of strings to render
- `padding`: Padding around text in pixels (default: 20)
- `min_width`: Minimum canvas width (default: None)
- `bg_color`: Background color as RGBA tuple (default: (30, 30, 30, 255))
- `text_color`: Text color as RGB tuple (default: (255, 255, 255))
- `text_opacity`: Text opacity 0-255 (default: 255)
- `font_path`: Path to TrueType font (default: "DejaVuSans.ttf")
- `font_size`: Font size in pixels (default: 40)
- `line_spacing`: Spacing between lines (default: 10)
- `align`: Text alignment - "left", "center", or "right" (default: "left")
- `crop_to_text`: Crop canvas tightly to text bounds (default: False)
- `transparent_text_only`: Create transparent background (default: False)
- `stroke_width`: Outline/stroke width in pixels (default: 0)
- `stroke_color`: Stroke color as RGBA tuple (default: None)
- `output_rgb`: Convert output to RGB mode (default: False)

### Mask Utilities

#### Get Biggest Blob (`get_biggest_blob`)

Extract the largest connected component from a binary mask.

```python
from imtools import get_biggest_blob
import numpy as np

# Create a mask with multiple blobs
mask = np.array([
    [1, 1, 0, 0, 0],
    [1, 0, 0, 1, 1],
    [0, 0, 0, 1, 1],
    [0, 0, 0, 0, 0],
], dtype=np.uint8)

# Extract only the largest blob
largest = get_biggest_blob(mask)
# Returns mask with only the 4-pixel blob on the right
```

**Parameters:**
- `mask`: Binary NumPy array with values 0 and 1

**Returns:**
- Binary mask containing only the largest connected component

#### Bounding Box from Mask (`bbox_from_mask`)

Compute the axis-aligned bounding box of a binary mask.

```python
from imtools import bbox_from_mask
import numpy as np

# Create a boolean mask
mask = np.array([
    [False, False, False, False],
    [False, True,  True,  False],
    [False, True,  True,  False],
    [False, False, False, False],
], dtype=bool)

# Get bounding box [col_start, row_start, col_end, row_end]
bbox = bbox_from_mask(mask)
# Returns [1, 1, 3, 3] - exclusive end coordinates
```

**Parameters:**
- `mask`: 2D boolean NumPy array

**Returns:**
- List `[col_start, row_start, col_end, row_end]` with exclusive end coordinates

#### Fill Holes in Mask (`fill_holes_mask`)

Process binary masks by applying morphological closing and filling internal holes.

```python
from imtools import fill_holes_mask
import numpy as np

# Create a mask with holes
mask = np.array([
    [1, 1, 1, 1, 1],
    [1, 0, 0, 0, 1],
    [1, 0, 0, 0, 1],
    [1, 0, 0, 0, 1],
    [1, 1, 1, 1, 1],
], dtype=np.uint8)

# Fill holes
filled = fill_holes_mask(mask)

# Custom parameters
filled = fill_holes_mask(
    mask,
    apply_closing=True,
    kernel_size=3,
    pad_length=10
)
```

**Parameters:**
- `mask`: Binary mask (NumPy array)
- `apply_closing`: Whether to apply morphological closing (default: True)
- `kernel_size`: Size of the kernel for closing operation (default: 5)
- `pad_length`: Padding length for flood fill operation (default: 5)

**Returns:**
- Processed binary mask with holes filled

### Image Layout

#### Stack Images (`stack_images`)

Stack multiple images horizontally or vertically with automatic resizing.

```python
from imtools import stack_images

# Stack images horizontally
result = stack_images(
    ["image1.png", "image2.png", "image3.png"],
    order='horizontal'
)

# Stack images vertically
result = stack_images(
    ["image1.png", "image2.png"],
    order='vertical'
)

# Images are automatically resized to match the first image's dimensions
# - Horizontal: heights are matched, widths scale proportionally
# - Vertical: widths are matched, heights scale proportionally
```

**Parameters:**
- `image_paths`: List of image file paths
- `order`: Stacking order - 'horizontal' or 'vertical' (default: 'horizontal')

**Returns:**
- Stacked image as NumPy array

## Development

### Testing

The package includes a comprehensive test suite with 95 tests covering all functionality.

Run all tests:

```bash
pytest
```

Run tests with verbose output:

```bash
pytest -v
```

Run tests with coverage report:

```bash
pytest --cov=imtools --cov-report=term-missing
```

#### Test Files

- `tests/test_load_image.py` - tests for image loading functionality
- `tests/test_draw.py` - tests for drawing functions (text_on_canvas, draw_mask_overlays)
- `tests/test_image_layout.py` - tests for image stacking
- `tests/test_mask_utils.py` - tests for mask utilities (get_biggest_blob, bbox_from_mask, fill_holes_mask)

Each test file includes comprehensive coverage of:
- Normal operation with various input types
- Edge cases and boundary conditions
- Error handling and input validation
- Parameter combinations and options

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
