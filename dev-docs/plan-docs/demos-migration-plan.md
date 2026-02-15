# demos/ Migration Plan

## Overview

Update all imports in `demos/` to work with the reorganized `imtools` package. Since the top-level `src/imtools/__init__.py` re-exports most commonly used symbols, many imports can be simplified to `from imtools import ...`.

## File-by-File Changes

---

### 1. `demos/demo_conversions.py`

**No change needed** — already uses a top-level re-export.

```python
# line 4 (unchanged)
from imtools import imwrite
```

---

### 2. `demos/setup_demo_data.py`

**No change needed** — all three imports resolve via top-level re-exports.

```python
# line 12 (unchanged — fill_holes_mask is re-exported at top level)
from imtools import fill_holes_mask

# line 41, 58 (unchanged)
from imtools import to_numpy_image
```

Wait — line 12 currently reads:

```python
from imtools.mask_utils import fill_holes_mask
```

This **must** change because `mask_utils.py` no longer exists.

| Line | Old import | New import |
|------|-----------|------------|
| 12 | `from imtools.mask_utils import fill_holes_mask` | `from imtools import fill_holes_mask` |
| 41 | `from imtools import to_numpy_image` | *(no change)* |
| 58 | `from imtools import to_numpy_image` | *(no change)* |

---

### 3. `demos/demo_overlay_mask.py`

All five old-style imports need updating. Every symbol is available at the top level, so all can become short imports.

| Line | Old import | New import |
|------|-----------|------------|
| 15 | `from imtools.converters import binary_mask_to_label_image` | `from imtools import binary_mask_to_label_image` |
| 16 | `from imtools.annotations import label_image_to_annotations` | `from imtools import label_image_to_annotations` |
| 17 | `from imtools.label_formats import LabelFormat` | `from imtools import LabelFormat` |
| 18 | `from imtools.common import BlendConfig, TitleConfig, LabelStyle` | `from imtools import BlendConfig, TitleConfig, LabelStyle` |
| 19 | `from imtools import overlay_visualize` | *(no change)* |

**Simplified result** — lines 15–19 collapse into two lines:

```python
from imtools import (
    binary_mask_to_label_image, label_image_to_annotations,
    LabelFormat, BlendConfig, TitleConfig, LabelStyle, overlay_visualize,
)
```

---

### 4. `demos/demo_overlay_yolo.py`

| Line | Old import | New import |
|------|-----------|------------|
| 9 | `from imtools.label_formats import LabelFormat` | `from imtools import LabelFormat` |
| 10 | `from imtools.common import BlendConfig, TitleConfig, LabelStyle` | `from imtools import BlendConfig, TitleConfig, LabelStyle` |
| 11 | `from imtools import overlay_visualize` | *(no change)* |
| 13 | `from imtools.annotations import yolo_to_annotations` | `from imtools import yolo_to_annotations` |
| 14 | `from imtools.converters import yolo_to_label_image` | `from imtools import yolo_to_label_image` |

**Simplified result** — lines 9–14 collapse into:

```python
from imtools import (
    LabelFormat, BlendConfig, TitleConfig, LabelStyle,
    overlay_visualize, yolo_to_annotations, yolo_to_label_image,
)
```

---

### 5. `demos/demo_overlay_sam3.py`

Identical import pattern to `demo_overlay_yolo.py`.

| Line | Old import | New import |
|------|-----------|------------|
| 9 | `from imtools.label_formats import LabelFormat` | `from imtools import LabelFormat` |
| 10 | `from imtools.common import BlendConfig, TitleConfig, LabelStyle` | `from imtools import BlendConfig, TitleConfig, LabelStyle` |
| 11 | `from imtools import overlay_visualize` | *(no change)* |
| 13 | `from imtools.annotations import yolo_to_annotations` | `from imtools import yolo_to_annotations` |
| 14 | `from imtools.converters import yolo_to_label_image` | `from imtools import yolo_to_label_image` |

**Simplified result:**

```python
from imtools import (
    LabelFormat, BlendConfig, TitleConfig, LabelStyle,
    overlay_visualize, yolo_to_annotations, yolo_to_label_image,
)
```

---

### 6. `demos/demo_color_gen.py`

| Line | Old import | New import |
|------|-----------|------------|
| 3 | `from imtools.color_gen import generate_colors, create_color_palette_image` | `from imtools import generate_colors, create_color_palette_image` |

---

### 7. `demos/demo_config.py`

Not listed in the grep output — inspect this file for any `imtools` imports. If it only reads `config.yaml` and has no imtools imports, no changes needed.

---

## Execution Steps

### Step 1: Update `setup_demo_data.py`

Replace the single broken import on line 12:

```bash
sed -i 's/from imtools.mask_utils import fill_holes_mask/from imtools import fill_holes_mask/' demos/setup_demo_data.py
```

### Step 2: Update `demo_overlay_mask.py`

Replace lines 15–19 with the consolidated import block:

```python
from imtools import (
    binary_mask_to_label_image, label_image_to_annotations,
    LabelFormat, BlendConfig, TitleConfig, LabelStyle, overlay_visualize,
)
```

### Step 3: Update `demo_overlay_yolo.py`

Replace lines 9–14 with:

```python
from imtools import (
    LabelFormat, BlendConfig, TitleConfig, LabelStyle,
    overlay_visualize, yolo_to_annotations, yolo_to_label_image,
)
```

### Step 4: Update `demo_overlay_sam3.py`

Replace lines 9–14 with:

```python
from imtools import (
    LabelFormat, BlendConfig, TitleConfig, LabelStyle,
    overlay_visualize, yolo_to_annotations, yolo_to_label_image,
)
```

### Step 5: Update `demo_color_gen.py`

Replace line 3:

```bash
sed -i 's/from imtools.color_gen import generate_colors, create_color_palette_image/from imtools import generate_colors, create_color_palette_image/' demos/demo_color_gen.py
```

### Step 6: Inspect `demo_config.py`

Check for any imtools imports and update if needed:

```bash
grep -n "imtools" demos/demo_config.py
```

### Step 7: Validate

```bash
# Syntax check all demo files
python -m py_compile demos/demo_conversions.py
python -m py_compile demos/setup_demo_data.py
python -m py_compile demos/demo_overlay_mask.py
python -m py_compile demos/demo_overlay_yolo.py
python -m py_compile demos/demo_overlay_sam3.py
python -m py_compile demos/demo_color_gen.py

# Confirm no old module references remain
grep -rn "imtools\.common\|imtools\.formats\|imtools\.mask_utils\|imtools\.converters\|imtools\.color_gen\|imtools\.label_formats" demos/
```

The grep should return **no results**. If it does, those lines still reference deleted modules and need updating.
