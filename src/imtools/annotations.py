import numpy as np
from typing import Dict, List, Union
from skimage.measure import regionprops
from .label_formats import LabelFormat, LabelContext, resolve_label


# ── Public adapter functions ────────────────────────────────────────────────

def yolo_to_annotations(
    result,
    conf_thresh: float = 0.0,
    label_format: Union[LabelFormat, str] = LabelFormat.DEFAULT,
) -> List[Dict[str, Union[str, int, float]]]:
    """
    Convert a YOLO segmentation result into a list of annotation dictionaries.

    Parameters
    ----------
    result
        YOLO result object with masks, boxes, and names attributes.
    conf_thresh : float, optional
        Confidence threshold for filtering detections. Defaults to 0.0.
    label_format : Union[LabelFormat, str], optional
        Format for generating labels. Defaults to LabelFormat.DEFAULT.

    Returns
    -------
    List[Dict[str, Union[str, int, float]]]
        List of annotation dictionaries with keys: 'text', 'x', 'y', 'score'.
    """
    annotations = []

    if result.masks is None:
        return annotations

    masks_tensor = result.masks.data
    boxes = result.boxes
    boxes_xywh = boxes.xywh.cpu().numpy()  # Center x, Center y, Width, Height

    for i, mask_tensor in enumerate(masks_tensor):
        conf = float(boxes.conf[i])
        if conf < conf_thresh:
            continue

        # Extract mask
        mask_np = mask_tensor.cpu().numpy().astype(bool)

        # Extract class and geometry data
        cls_id = int(boxes.cls[i])
        class_name = result.names[cls_id]
        cx, cy, w, h = boxes_xywh[i]

        # Create context object
        ctx = LabelContext(
            class_name=class_name,
            conf=conf,
            index=i,
            cx=int(cx),
            cy=int(cy),
            width=int(w),
            height=int(h),
            area=int(mask_np.sum())
        )

        # Generate label using the resolver
        label_text = resolve_label(label_format, ctx)

        info = {
            'text': label_text,
            'x': ctx.cx,
            'y': ctx.cy,
            'score': conf,
        }
        annotations.append(info)

    return annotations

def label_image_to_annotations(
    label_image: np.ndarray,
    class_name: str = '',
    label_format: Union[LabelFormat, str] = LabelFormat.DEFAULT,
) -> List[Dict[str, Union[str, int, float]]]:
    """
    Parse a label image (integer mask) and convert regions into annotation dictionaries.

    Parameters
    ----------
    label_image : np.ndarray
        Integer mask where each unique value represents a distinct region.
    class_name : str, optional
        Class name to use for all regions. Defaults to 'NA'.
    label_format : Union[LabelFormat, str], optional
        Format for generating labels. Defaults to LabelFormat.DEFAULT.

    Returns
    -------
    List[Dict[str, Union[str, int, float]]]
        List of annotation dictionaries with keys: 'text', 'x', 'y', 'score'.
    """
    annotations = []

    # regionprops iterates through every unique integer in the label_image
    for i, region in enumerate(regionprops(label_image)):
        # Geometry calculation
        # regionprops.centroid returns (row, col) -> flip to (y, x)
        cy, cx = region.centroid

        # regionprops.bbox returns (min_row, min_col, max_row, max_col)
        min_row, min_col, max_row, max_col = region.bbox

        width = max_col - min_col
        height = max_row - min_row
        area = region.area

        # Create context object
        ctx = LabelContext(
            class_name=class_name,  # Uses the argument provided (default 'NA')
            conf=1.0,               # Label images are ground truth, so conf is 1.0
            index=i,                # The loop index
            cx=int(cx),
            cy=int(cy),
            width=int(width),
            height=int(height),
            area=int(area)
        )

        # Generate label using the resolver
        label_text = resolve_label(label_format, ctx)

        info = {
            'text': label_text,
            'x': ctx.cx,
            'y': ctx.cy,
            'score': ctx.conf,
        }
        annotations.append(info)

    return annotations
