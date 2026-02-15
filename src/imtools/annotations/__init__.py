"""Annotations subpackage for imtools.

Re-exports label formatting, annotation parsing, and YOLO conversion
utilities.

Public API:
    - :class:`~imtools.annotations.labels.LabelFormat`
    - :class:`~imtools.annotations.labels.LabelContext`
    - :func:`~imtools.annotations.labels.resolve_label`
    - :func:`~imtools.annotations.parsers.yolo_to_annotations`
    - :func:`~imtools.annotations.parsers.label_image_to_annotations`
    - :func:`~imtools.annotations.yolo.yolo_to_label_image`
"""

from imtools.annotations.parsers import yolo_to_annotations, label_image_to_annotations
from imtools.annotations.labels import LabelFormat, LabelContext, resolve_label
from imtools.annotations.yolo import yolo_to_label_image
