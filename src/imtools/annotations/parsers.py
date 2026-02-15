"""Annotation conversion utilities for the imtools package.

Provides adapters for converting YOLO segmentation results and integer
label images into a common list-of-dict annotation format consumed by
the visualization pipeline.
"""

from __future__ import annotations

import numpy as np
from typing import Any, Dict, List, Union
from skimage.measure import regionprops
from imtools.annotations.labels import LabelFormat, LabelContext, resolve_label


# ── Public adapter functions ────────────────────────────────────────────────

def yolo_to_annotations(
    result: Any,
    conf_thresh: float = 0.0,
    label_format: Union[LabelFormat, str] = LabelFormat.DEFAULT,
) -> List[Dict[str, Union[str, int, float]]]:
    """Convert a YOLO segmentation result into a list of annotation dicts.

    Iterates over all detections in a YOLO result object, optionally filters
    by confidence threshold, and converts each detection to the standard
    annotation dictionary format used by
    :func:`~imtools.viz.pipeline.draw_labels`.

    Args:
        result: Ultralytics YOLO ``Results`` object with ``masks``,
            ``boxes``, and ``names`` attributes.  If ``result.masks`` is
            ``None`` an empty list is returned immediately.
        conf_thresh: Minimum confidence score for a detection to be included.
            Detections with ``conf < conf_thresh`` are skipped.
        label_format: :class:`~imtools.annotations.labels.LabelFormat` enum
            member or a custom template string for generating label text.

    Returns:
        List of annotation dictionaries, each with the keys:

        - ``'text'`` (:class:`str`) — formatted label string.
        - ``'x'`` (:class:`int`) — centroid x-coordinate.
        - ``'y'`` (:class:`int`) — centroid y-coordinate.
        - ``'score'`` (:class:`float`) — detection confidence.

    Example:
        >>> # Assuming `result` is a YOLO Results object
        >>> anns = yolo_to_annotations(result, conf_thresh=0.5)
        >>> anns[0].keys()
        dict_keys(['text', 'x', 'y', 'score'])
    """
    annotations: List[Dict[str, Union[str, int, float]]] = []

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

        info: Dict[str, Union[str, int, float]] = {
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
    """Parse an integer label image into a list of annotation dicts.

    Applies :func:`~skimage.measure.regionprops` to ``label_image`` and
    converts each detected region into the standard annotation dictionary
    format.  All regions receive a confidence score of ``1.0`` (ground truth).

    Args:
        label_image: 2-D integer array of shape ``(H, W)`` where each
            unique non-zero value identifies a distinct region.
        class_name: Class label assigned to every region.  Defaults to an
            empty string; pass a meaningful name (e.g. ``'cell'``) when the
            image contains a single object type.
        label_format: :class:`~imtools.annotations.labels.LabelFormat` enum
            member or a custom template string for label text generation.

    Returns:
        List of annotation dictionaries, each with the keys:

        - ``'text'`` (:class:`str`) — formatted label string.
        - ``'x'`` (:class:`int`) — centroid x-coordinate (column).
        - ``'y'`` (:class:`int`) — centroid y-coordinate (row).
        - ``'score'`` (:class:`float`) — always ``1.0``.

    Example:
        >>> import numpy as np
        >>> from skimage.measure import label
        >>> lbl = label(np.eye(3, dtype=bool))
        >>> anns = label_image_to_annotations(lbl, class_name='dot')
        >>> len(anns)
        3
    """
    annotations: List[Dict[str, Union[str, int, float]]] = []

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

        info: Dict[str, Union[str, int, float]] = {
            'text': label_text,
            'x': ctx.cx,
            'y': ctx.cy,
            'score': ctx.conf,
        }
        annotations.append(info)

    return annotations
