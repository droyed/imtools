# imtools Package Reorganization Plan

## Overview

Reorganize the flat `imtools` package into a logical subpackage structure grouped by domain. Old modules are deleted outright — no backward-compatibility shims. The top-level `src/imtools/__init__.py` re-exports the most commonly used symbols so users can do `from imtools import overlay_visualize` without deep paths.

## Current vs Proposed Structure

```
CURRENT                                  PROPOSED
src/imtools/                             src/imtools/
├── common.py                            ├── core/
├── formats.py                           │   ├── __init__.py
├── mask_utils.py                        │   ├── types.py          ← common.py
│                                        │   └── formats.py        ← formats.py (including imwrite)
├── converters.py                        ├── masks/
├── annotations.py                       │   ├── __init__.py
├── label_formats.py                     │   ├── ops.py            ← mask_utils.py
├── color_gen.py                         │   └── converters.py     ← converters.py
├── compose.py                           ├── annotations/
├── display_utils.py                     │   ├── __init__.py
├── viz_pipeline.py                      │   ├── parsers.py        ← annotations.py
├── viz/                                 │   ├── labels.py         ← label_formats.py
│   ├── __init__.py                      │   └── yolo.py           ← yolo_to_label_image from converters.py
│   └── overlay_viewer.py                ├── viz/
├── benchmarks/                          │   ├── __init__.py
│   ├── __init__.py                      │   ├── colors.py         ← color_gen.py
│   ├── __main__.py                      │   ├── pipeline.py       ← viz_pipeline.py
│   └── benchmark_converters.py          │   ├── compose.py        ← compose.py
└── __init__.py                          │   ├── display.py        ← display_utils.py
                                         │   └── overlay_viewer.py ← viz/overlay_viewer.py (moved up)
                                         ├── benchmarks/           (unchanged)
                                         │   ├── __init__.py
                                         │   ├── __main__.py
                                         │   └── benchmark_converters.py
                                         └── __init__.py
```

## Migration Steps

Execute these steps **in order**. Each step is independently testable.

---

### Step 1: Create `core/` subpackage

Create `src/imtools/core/` and move foundational types and format converters into it.

**Files to create:**

- `src/imtools/core/__init__.py`
- `src/imtools/core/types.py` — move all contents of `common.py` here
- `src/imtools/core/formats.py` — move all contents of `formats.py` here (including `imwrite`)

**Re-exports in `src/imtools/core/__init__.py`:**

```python
from imtools.core.types import Annotation, BlendConfig, TitleConfig, LabelStyle
from imtools.core.formats import pil_to_opencv, opencv_to_pil, to_pil_image, to_numpy_image, imwrite
```

**Delete old files:**

- `src/imtools/common.py`
- `src/imtools/formats.py`

**Update internal imports:** Any file that does `from imtools.common import ...` or `from imtools.formats import ...` should be updated to import from `imtools.core.types` or `imtools.core.formats`.

---

### Step 2: Create `masks/` subpackage

Create `src/imtools/masks/` for mask operations and mask-level converters.

**Files to create:**

- `src/imtools/masks/__init__.py`
- `src/imtools/masks/ops.py` — move all contents of `mask_utils.py` here
- `src/imtools/masks/converters.py` — move these functions from `converters.py`:
  - `masks_to_label_image_torch_vectorized`
  - `masks_to_label_image_torch_loop`
  - `masks_to_label_image_numpy_loop`
  - `masks_to_label_image_numpy_vectorized`
  - `binary_mask_to_label_image`
  - `masks_to_label_image`

**Re-exports in `src/imtools/masks/__init__.py`:**

```python
from imtools.masks.ops import get_biggest_blob, bbox_from_mask, fill_holes_mask, extract_region_metadata
from imtools.masks.converters import masks_to_label_image, binary_mask_to_label_image
```

**Delete old file:**

- `src/imtools/mask_utils.py`

Note: `src/imtools/converters.py` still holds `yolo_to_label_image` — it will be deleted in Step 3 after that function is moved.

---

### Step 3: Create `annotations/` subpackage

Create `src/imtools/annotations/` for annotation parsing, label formatting, and YOLO conversions.

**Files to create:**

- `src/imtools/annotations/__init__.py`
- `src/imtools/annotations/parsers.py` — move all contents of `annotations.py` here
- `src/imtools/annotations/labels.py` — move all contents of `label_formats.py` here
- `src/imtools/annotations/yolo.py` — move `yolo_to_label_image` from `converters.py` here

**Re-exports in `src/imtools/annotations/__init__.py`:**

```python
from imtools.annotations.parsers import yolo_to_annotations, label_image_to_annotations
from imtools.annotations.labels import LabelFormat, LabelContext, resolve_label
from imtools.annotations.yolo import yolo_to_label_image
```

**Delete old files:**

- `src/imtools/annotations.py`
- `src/imtools/label_formats.py`
- `src/imtools/converters.py` (now fully split across `masks/converters.py` and `annotations/yolo.py`)

---

### Step 4: Reorganize `viz/` subpackage

Consolidate all visualization code under `src/imtools/viz/`.

**Files to create / move:**

- `src/imtools/viz/colors.py` — move all contents of `color_gen.py` here
- `src/imtools/viz/pipeline.py` — move all contents of `viz_pipeline.py` here
- `src/imtools/viz/compose.py` — move all contents of `compose.py` here
- `src/imtools/viz/display.py` — move all contents of `display_utils.py` here
- `src/imtools/viz/overlay_viewer.py` — already exists, no move needed

**Update `src/imtools/viz/__init__.py`:**

```python
from imtools.viz.colors import generate_colors, create_color_palette_image
from imtools.viz.pipeline import overlay_visualize, draw_labels, create_label_overlay_from_labelimg
from imtools.viz.compose import text_on_canvas, stack_images, add_title
from imtools.viz.display import show_cv2, show_mpl, show_cv2_fullscreen
from imtools.viz.overlay_viewer import OverlayViewer, run_overlay_viewer
```

**Delete old files:**

- `src/imtools/color_gen.py`
- `src/imtools/viz_pipeline.py`
- `src/imtools/compose.py`
- `src/imtools/display_utils.py`

---

### Step 5: Update all internal cross-imports

Grep the entire codebase for old import paths and update them to the new canonical locations. Key patterns to find-and-replace:

| Old import | New import |
|---|---|
| `from imtools.common import` | `from imtools.core.types import` |
| `from imtools.formats import` | `from imtools.core.formats import` |
| `from imtools.mask_utils import` | `from imtools.masks.ops import` |
| `from imtools.converters import` | `from imtools.masks.converters import` or `from imtools.annotations.yolo import` |
| `from imtools.color_gen import` | `from imtools.viz.colors import` |
| `from imtools.display_utils import` | `from imtools.viz.display import` |
| `from imtools.viz_pipeline import` | `from imtools.viz.pipeline import` |
| `from imtools.compose import` | `from imtools.viz.compose import` |
| `from imtools.label_formats import` | `from imtools.annotations.labels import` |
| `from imtools.annotations import` | `from imtools.annotations.parsers import` |

Also update `src/imtools/benchmarks/benchmark_converters.py` imports accordingly.

---

### Step 6: Update top-level `__init__.py`

Rewrite `src/imtools/__init__.py` to re-export the most commonly used symbols. This lets users write `from imtools import overlay_visualize, BlendConfig` without deep paths, while less common utilities remain accessible via their subpackage.

```python
# --- Core types & format converters ---
from imtools.core.types import Annotation, BlendConfig, TitleConfig, LabelStyle
from imtools.core.formats import (
    pil_to_opencv,
    opencv_to_pil,
    to_pil_image,
    to_numpy_image,
    imwrite,
)

# --- Mask operations & converters ---
from imtools.masks.ops import get_biggest_blob, bbox_from_mask, fill_holes_mask
from imtools.masks.converters import masks_to_label_image, binary_mask_to_label_image

# --- Annotation parsing & label formatting ---
from imtools.annotations.parsers import yolo_to_annotations, label_image_to_annotations
from imtools.annotations.labels import LabelFormat, LabelContext, resolve_label
from imtools.annotations.yolo import yolo_to_label_image

# --- Visualization ---
from imtools.viz.pipeline import overlay_visualize, draw_labels, create_label_overlay_from_labelimg
from imtools.viz.colors import generate_colors, create_color_palette_image
from imtools.viz.compose import text_on_canvas, stack_images, add_title
from imtools.viz.display import show_cv2, show_mpl, show_cv2_fullscreen
from imtools.viz.overlay_viewer import OverlayViewer, run_overlay_viewer
```

**What stays subpackage-only (not re-exported at top level):**

These are internal/specialized and should be imported from their subpackage directly:

- `imtools.masks.ops.extract_region_metadata`
- `imtools.masks.converters.masks_to_label_image_torch_vectorized`
- `imtools.masks.converters.masks_to_label_image_torch_loop`
- `imtools.masks.converters.masks_to_label_image_numpy_loop`
- `imtools.masks.converters.masks_to_label_image_numpy_vectorized`
- `imtools.viz.colors.generate_distinct_colors_hsv`
- `imtools.viz.colors.generate_distinct_colors_golden_ratio`
- `imtools.viz.colors.generate_distinct_colors_kmeans`
- `imtools.viz.colors.generate_distinct_colors_preset`
- `imtools.viz.colors.generate_colors_from_colormap`
- `imtools.viz.colors.generate_colors_from_colormap_extended`
- `imtools.viz.display.block_until_closed`
- `imtools.viz.display.get_screen_resolution`
- `imtools.viz.display.scale_image_to_fit`

---

## Validation Checklist

After each step, verify:

- [ ] All existing tests pass (run full test suite)
- [ ] `python -c "import imtools"` works without errors
- [ ] Spot-check key imports: `from imtools import overlay_visualize, BlendConfig, masks_to_label_image`
- [ ] Benchmarks still run: `python -m imtools.benchmarks`
- [ ] No circular imports (test with `python -c "from imtools.core import types; from imtools.viz import pipeline"`)
- [ ] No leftover old modules: `find src/imtools -maxdepth 1 -name '*.py' ! -name '__init__.py'` should return nothing

## Notes

- **Circular imports**: `viz/pipeline.py` likely imports from `core/types.py` and `viz/colors.py`. Ensure `core/` never imports from `viz/` or `masks/`. The dependency graph should be: `core` ← `masks` ← `annotations` ← `viz`.
- **`converters.py` split**: `yolo_to_label_image` depends on YOLO result objects and semantically belongs with annotations, while the `masks_to_label_image_*` variants are pure mask operations. Splitting them across `masks/converters.py` and `annotations/yolo.py` reflects this.
