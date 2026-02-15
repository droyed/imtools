# imtools

![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Version](https://img.shields.io/badge/version-0.1.2-green.svg)

Toolkit for computer vision segmentation visualization — convert mask arrays to publication-ready annotated overlay images.

## Features

- PIL ↔ OpenCV format conversions (11+ image modes)
- Binary mask operations (blobs, bounding boxes, hole filling)
- 3D mask → 2D label image converters (PyTorch GPU/CPU + NumPy backends)
- Distinct color generation (Kelly palette, HSV, golden ratio, K-means, matplotlib colormaps)
- 39 annotation label formats (standard, geometric, positional, JSON/CSV, etc.)
- Full overlay visualization pipeline with configurable blending & styling
- OpenCV/matplotlib display utilities
- Image composition (stacking, title bars, text on canvas)
- Performance benchmarks: GPU vectorized achieves ~1722 FPS for 640×640

## Installation

```bash
git clone <repo>
cd imtools
pip install -e ".[dev,test]"
# or via make
make install
```

## Quick Start

YOLO segmentation workflow:

```python
from imtools import yolo_to_label_image, yolo_to_annotations, overlay_visualize
from imtools.common import BlendConfig, LabelStyle
from imtools.label_formats import LabelFormat
from ultralytics import YOLO

model = YOLO('yolov8n-seg.pt')
results = model('image.jpg')[0]

label_image = yolo_to_label_image(results)
annotations = yolo_to_annotations(results, label_format=LabelFormat.PIPE_FULL)

overlay = overlay_visualize(
    img=results.orig_img,
    label_image=label_image,
    annotations=annotations,
    blend_config=BlendConfig(alpha=0.6, method='golden_ratio'),
    label_style=LabelStyle.Presets.high_contrast(),
    title="Segmentation Results"
)
overlay.save('output.png')
```

## Module Overview

| Module | Purpose |
|---|---|
| `formats.py` | PIL ↔ OpenCV conversion, image I/O |
| `mask_utils.py` | Binary mask operations |
| `converters.py` | 3D masks → 2D label images (multi-backend) |
| `color_gen.py` | Distinct color generation algorithms |
| `annotations.py` | YOLO/label image → annotation objects |
| `label_formats.py` | 39 label format variants |
| `viz_pipeline.py` | Full overlay visualization pipeline |
| `compose.py` | Image composition and stacking |
| `display_utils.py` | OpenCV/matplotlib display helpers |
| `common.py` | Configuration dataclasses |
| `viz/overlay_viewer.py` | Interactive GUI viewer |

## Running Demos

```bash
python -m demos          # Run all demos
make demo                # Alternative shortcut
```

Demo scripts in `demos/`:

- `demo_overlay_yolo.py` — YOLO segmentation visualization
- `demo_overlay_sam3.py` — SAM3 results visualization
- `demo_overlay_mask.py` — Binary mask overlay
- `demo_color_gen.py` — Color generation showcase
- `demo_conversions.py` — Format conversion examples

## Benchmarks

```
>>> python -m imtools.benchmarks
# Scenario: N=10, H=640, W=640
Torch GPU | Vectorized (No Transfer)  1721.9 FPS
Torch GPU | Loop (Force CUDA)         1163.1 FPS
NumPy | Vectorized                     136.9 FPS
NumPy | Loop                            68.4 FPS
```

## Development

```bash
make lint    # Run ruff linter
make test    # Run pytest with coverage
make clean   # Remove build artifacts and outputs
```

## Documentation

- `docs/imtools_API_Feb9.md` — Comprehensive 2600+ line API reference
- `docs/OVERLAY_README.md` — Overlay API overview
- `docs/imtools_overlay_guide.ipynb` — Jupyter notebook guide

## Attributions

Test images located at `assets/` are from the [COCO val2017 dataset](https://cocodataset.org/) (Lin et al., 2015), licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.