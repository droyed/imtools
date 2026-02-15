"""Masks subpackage for imtools.

Re-exports the primary mask operation and conversion utilities.

Public API:
    - :func:`~imtools.masks.ops.get_biggest_blob`
    - :func:`~imtools.masks.ops.bbox_from_mask`
    - :func:`~imtools.masks.ops.fill_holes_mask`
    - :func:`~imtools.masks.ops.extract_region_metadata`
    - :func:`~imtools.masks.converters.masks_to_label_image`
    - :func:`~imtools.masks.converters.binary_mask_to_label_image`
"""

from imtools.masks.ops import get_biggest_blob, bbox_from_mask, fill_holes_mask, extract_region_metadata
from imtools.masks.converters import masks_to_label_image, binary_mask_to_label_image
