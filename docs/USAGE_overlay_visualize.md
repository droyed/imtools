## Usage - `overlay_visualize`

The simplest way to visualize an overlay:

```python
overlay_visualize(img, label_image)
```

| Parameter | Description |
|-----------|-------------|
| `img` | Image path or image array |
| `label_image` | Labeled image with indices where each index represents a region |

## Creating a Label Image

Choose the method that matches your input:

**From a binary mask:**

```python
from imtools.masks.converters import binary_mask_to_label_image

label_image = binary_mask_to_label_image(mask)
```

**From YOLO/SAM results** (`ultralytics.engine.results.Results`):

```python
from imtools.annotations.yolo import yolo_to_label_image

label_image = yolo_to_label_image(results)
```

## Adding Annotations

Annotations enrich your overlay with additional context.

**Generic (from label image):**

```python
from imtools.annotations import label_image_to_annotations

annotations = label_image_to_annotations(label_image)
```

**From YOLO/SAM results** (includes class names, confidence, etc.):

```python
from imtools.annotations import yolo_to_annotations

annotations = yolo_to_annotations(results)
```

**Then visualize:**

```python
overlay_visualize(img, label_image, annotations)
```
