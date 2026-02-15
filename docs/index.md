# imtools

Image processing and visualization toolkit for mask operations, annotation parsing, and visualization.

## Installation

```bash
pip install -e .
```

For development:
```bash
pip install -e ".[dev]"
```

For documentation:
```bash
pip install -e ".[docs]"
```

## Package Overview

| Subpackage | Description |
|------------|-------------|
| [`core`](api/core.md) | Data types and image format converters (PIL ↔ OpenCV) |
| [`annotations`](api/annotations.md) | YOLO annotation parsing and label formatting |
| [`masks`](api/masks.md) | Mask operations and label image converters |
| [`viz`](api/viz.md) | Visualization, color generation, and display utilities |
| [`benchmarks`](api/benchmarks.md) | Performance benchmarks for mask converters |

## Quick Start

### Image Format Conversion

```python
from imtools.core.formats import pil_to_opencv, opencv_to_pil, imwrite

# PIL -> OpenCV
cv_img = pil_to_opencv(pil_image)

# OpenCV -> PIL
pil_img = opencv_to_pil(cv_img)

# Save images
imwrite(image_array, "output.png")
```

### Mask Operations

```python
from imtools.masks.ops import get_biggest_blob, bbox_from_mask, fill_holes_mask
from imtools.masks.converters import masks_to_label_image, binary_mask_to_label_image

# Get largest connected component
largest_mask = get_biggest_blob(binary_mask)

# Compute bounding box
bbox = bbox_from_mask(mask)  # [col_start, row_start, col_end, row_end]

# Fill holes in mask
filled = fill_holes_mask(mask)

# Convert masks to label image (N, H, W) -> (H, W)
label_img = masks_to_label_image(masks_array)
```

### Annotation Parsing

```python
from imtools.annotations.parsers import yolo_to_annotations, label_image_to_annotations
from imtools.annotations.labels import LabelFormat, LabelContext, resolve_label

# Parse YOLO results
annotations = yolo_to_annotations(yolo_result, conf_thresh=0.5)

# Parse label image
annotations = label_image_to_annotations(label_image, class_name="car")

# Custom label formatting
ctx = LabelContext(class_name="person", conf=0.95, cx=100, cy=200)
label = resolve_label(LabelFormat.PIPE_FULL, ctx)
# -> "#1 | person | 0.95 | 100x200"
```

### Visualization

```python
from imtools.viz.pipeline import overlay_visualize
from imtools.viz.colors import generate_colors
from imtools.viz.display import show_cv2, show_mpl
from imtools.core.types import BlendConfig

# Generate color palette
colors = generate_colors(10, method='preset')

# Create overlay visualization
result = overlay_visualize(
    img,
    label_image,
    annotations=annotations,
    blend_config=BlendConfig.Presets.categorical(),
    title="Detection Results"
)
```

For a detailed guide on `overlay_visualize`, see the [Usage](USAGE_overlay_visualize.md) page.

## Benchmarks

Run performance benchmarks:

```bash
python -m imtools.benchmarks
```

## Dependencies

- numpy, opencv-python, torch, pillow, matplotlib
- scikit-image, scipy, pandas
- dearpygui (for interactive overlay viewer)
- pyyaml

## License

MIT
