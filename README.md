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

## Package Overview

| Subpackage | Description |
|------------|-------------|
| `core` | Data types and image format converters (PIL <-> OpenCV) |
| `annotations` | YOLO annotation parsing and label formatting |
| `masks` | Mask operations and label image converters |
| `viz` | Visualization, color generation, and display utilities |
| `benchmarks` | Performance benchmarks for mask converters |

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

# Convert binary mask to connected components
label_img = binary_mask_to_label_image(mask)
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
from imtools.viz.colors import generate_colors, create_color_palette_image
from imtools.viz.compose import text_on_canvas, stack_images, add_title
from imtools.viz.display import show_cv2, show_mpl

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

# Interactive overlay viewer (requires dearpygui)
from imtools.viz.overlay_viewer import run_overlay_viewer
blended_img, settings = run_overlay_viewer(image, label_image)
```

For detailed documentation on `overlay_visualize`, see [docs/USAGE_overlay_visualize.md](docs/USAGE_overlay_visualize.md) and [notebooks/imtools_overlay_guide.ipynb](notebooks/imtools_overlay_guide.ipynb).

### Configuration

```python
from imtools.core.types import BlendConfig, LabelStyle, TitleConfig

# Blend configuration with presets
config = BlendConfig.Presets.categorical()
config = BlendConfig.Presets.vibrant()
config = BlendConfig.Presets.pastel()
config = BlendConfig.Presets.colorblind_safe()

# Label styling
style = LabelStyle.Presets.dense_detection()
style = LabelStyle.Presets.segmentation_mask()

# Title configuration
title_cfg = TitleConfig.Presets.presentation()
title_cfg = TitleConfig.Presets.dark_mode()
```

## Run Demos

Run the demo scripts:

```bash
python -m demos
```

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

## Attributions

Test images located at `assets/` are from the [COCO val2017 dataset](https://cocodataset.org/) (Lin et al., 2015), licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
