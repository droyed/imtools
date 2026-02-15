"""Binary mask operations for the imtools package.

Provides utilities for extracting the largest connected component,
computing bounding boxes, filling holes, and extracting region
metadata from masks.
"""

from __future__ import annotations

import numpy as np
from skimage.measure import label, regionprops
from scipy.ndimage import binary_closing
from skimage.segmentation import flood_fill
from typing import Any


def _extract_properties(regionprops_object: Any) -> list[str]:
    """Return a sorted list of non-callable, non-private attribute names."""
    properties = []
    for attr in dir(regionprops_object):
        if attr.startswith('_'):
            continue
        try:
            value = getattr(regionprops_object, attr)
        except AttributeError:
            continue
        if callable(value):
            continue
        properties.append(attr)
    sorted_properties = sorted(properties)
    return sorted_properties

def _extract_info(regionprops_object: Any) -> dict[str, Any]:
    """Return a dictionary mapping every available property name to its value."""
    available_props = _extract_properties(regionprops_object)
    largest_region_props = {
        prop: getattr(regionprops_object, prop)
        for prop in available_props
    }
    return largest_region_props

def get_biggest_blob(
    mask: np.ndarray,
    return_props: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    """Return a binary mask containing only the largest connected component.

    Uses scikit-image :func:`~skimage.measure.label` with 8-connectivity
    (``connectivity=2``) to identify connected components and keeps the one
    with the greatest pixel area.

    Args:
        mask: 2-D binary array with values ``{0, 1}``.
        return_props: If ``True``, also return a dictionary of
            :func:`~skimage.measure.regionprops` attributes for the
            largest region.

    Returns:
        - If ``return_props`` is ``False`` (default): a boolean NumPy array
          of the same shape as ``mask`` with only the largest blob set to
          ``True``.
        - If ``return_props`` is ``True``: a ``(largest_blob_mask, props)``
          tuple where ``props`` is a :class:`dict` mapping property names
          to their values.
        - In both cases, an all-zero array is returned when ``mask`` is
          empty (no foreground pixels).

    Raises:
        AssertionError: If ``mask`` is not a NumPy array or contains values
            other than ``0`` and ``1``.

    Example:
        >>> import numpy as np
        >>> mask = np.array([[1, 1, 0, 0],
        ...                  [1, 0, 0, 1],
        ...                  [0, 0, 1, 1]])
        >>> get_biggest_blob(mask)
        array([[True, True, False, False],
               [True, False, False, False],
               [False, False, False, False]])
    """
    # Input validation
    assert isinstance(mask, np.ndarray), "Input must be a numpy array"
    assert np.all(np.isin(mask, [0, 1])), "Input array must contain only binary values (0 and 1)"

    # Handle edge case: empty mask (all zeros)
    if not mask.any():
        return mask.copy()

    # Label connected components with 8-connectivity
    labeled_mask = label(mask, connectivity=2)

    # Extract region properties
    regions = regionprops(labeled_mask)

    # Handle edge case: no regions found (shouldn't happen if mask.any() is True, but safe check)
    if len(regions) == 0:
        return np.zeros_like(mask)

    # Find the region with the largest area
    largest_region = max(regions, key=lambda r: r.area)

    # Create output mask with only the largest blob
    output_mask = labeled_mask == largest_region.label

    if return_props:
        # Create output mask with only the largest blob
        props = _extract_info(largest_region)
        return output_mask, props
    else:
        return output_mask

def bbox_from_mask(mask: np.ndarray) -> list[int]:
    """Compute the axis-aligned bounding box of a 2-D boolean mask.

    The bounding box convention is ``[col_start, row_start, col_end, row_end]``
    where ``col_end`` and ``row_end`` are **exclusive** (one past the last
    ``True`` pixel along each axis).

    Args:
        mask: 2-D boolean array.  ``True`` pixels are considered foreground.

    Returns:
        Bounding box as ``[col_start, row_start, col_end, row_end]`` using
        0-based indexing and exclusive end coordinates.

        Note:
            If ``mask`` contains **no** ``True`` pixels the function returns
            ``[0, 0, n, m]`` (full-image box) because :func:`numpy.argmax`
            returns ``0`` for an all-``False`` array — caller should guard
            against this edge case.

    Raises:
        AssertionError: If ``mask`` is not a NumPy array, is not 2-D, or
            does not have dtype ``bool``.

    Example:
        >>> import numpy as np
        >>> m = np.zeros((10, 10), dtype=bool)
        >>> m[2:5, 3:7] = True
        >>> bbox_from_mask(m)
        [3, 2, 7, 5]
    """
    assert isinstance(mask, np.ndarray), "Input must be a numpy array"
    assert mask.ndim == 2, "Input must be a 2D numpy array"
    # NOTE: `np.bool` is deprecated; boolean arrays use dtype `np.bool_` / `bool`.
    assert mask.dtype == np.bool_, "Input must be a boolean numpy array"

    m,n = mask.shape
    mask0,mask1 = mask.any(0),mask.any(1)
    col_start,col_end = mask0.argmax(),n-mask0[::-1].argmax()
    row_start,row_end = mask1.argmax(),m-mask1[::-1].argmax()
    bbox = [col_start,row_start,col_end,row_end]
    return bbox

def fill_holes_mask(
    mask: np.ndarray,
    apply_closing: bool = True,
    kernel_size: int = 5,
    pad_length: int = 5,
) -> np.ndarray:
    """Fill internal holes in a binary mask using morphological closing and flood fill.

    Algorithm:
        1. (Optional) Apply binary morphological closing to bridge small gaps.
        2. Pad the mask with zeros on all sides.
        3. Flood-fill the exterior from the top-left corner — any pixel that
           remains un-filled after this step is an enclosed hole.
        4. Combine the original mask with the un-filled (hole) regions.
        5. Crop the output back to the original dimensions.

    Args:
        mask: 2-D binary array (values ``0`` / ``1`` or boolean).
        apply_closing: If ``True`` (default), apply morphological closing
            before flood-filling to bridge narrow gaps.
        kernel_size: Edge length of the square structuring element used for
            morphological closing.
        pad_length: Width of the zero-padding applied before flood-filling.
            Larger values ensure the flood fill reaches all exterior regions.

    Returns:
        A binary mask of the same spatial shape as ``mask`` with internal
        holes filled.

    Example:
        >>> import numpy as np
        >>> ring = np.zeros((10, 10), dtype=np.uint8)
        >>> ring[2:8, 2:8] = 1
        >>> ring[4:6, 4:6] = 0   # punch a hole
        >>> filled = fill_holes_mask(ring, apply_closing=False)
        >>> filled[5, 5]  # previously a hole
        True
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    if apply_closing:
        mask = binary_closing(mask, structure=kernel)

    # Pad the mask to enable flood fill from edges
    mask_padded = np.pad(mask, ((pad_length, pad_length), (pad_length, pad_length)), mode='constant', constant_values=0)

    # Convert to uint8 and apply flood fill from corner
    mask_uint8 = mask_padded.astype(np.uint8) * 255
    filled_uint8 = flood_fill(mask_uint8, (0, 0), 255)

    # Combine original mask with inverted flood fill to get holes filled
    holes_filled_mask = mask_padded | (filled_uint8 == 0)

    # Crop back to original size
    output_mask = holes_filled_mask[pad_length:-pad_length, pad_length:-pad_length].copy()

    return output_mask

def _get_regions(mask: np.ndarray) -> list:
    """Return regionprops for all foreground regions in a binary mask."""
    labeled_mask = label(mask, background=0)
    return regionprops(labeled_mask)

def extract_region_metadata(
    mask: np.ndarray,
    item_size_threshold: int = 10,
) -> list[dict[str, Any]]:
    """Extract scalar regionprops metadata for every region in a mask.

    Iterates over all connected regions detected in ``mask`` and returns
    their :func:`~skimage.measure.regionprops` attributes, filtering out
    large array-valued properties (e.g. ``image``, ``coords``) that would
    consume excessive memory.

    Args:
        mask: 2-D integer label image or binary mask.  Each unique non-zero
            value is treated as a separate region.
        item_size_threshold: Maximum number of elements allowed for
            array-valued properties.  Properties whose ``ndarray.size``
            meets or exceeds this threshold are discarded.

    Returns:
        A list of dictionaries, one per region, where each dictionary maps
        property name (``str``) to property value.  Large array values are
        excluded.

    Example:
        >>> import numpy as np
        >>> from skimage.measure import label
        >>> mask = label(np.array([[0, 1, 1], [0, 1, 0], [1, 1, 0]]))
        >>> meta = extract_region_metadata(mask)
        >>> [d['area'] for d in meta]
        [3, 2]
    """
    regions = _get_regions(mask)
    metadata = []

    for region in regions:
        per_region_info = {}
        # Assuming imtools extracts heavy data
        info = _extract_info(region)

        for item_name, item in info.items():
            # Keep item if it is NOT an array, OR if it is a small array
            if not isinstance(item, np.ndarray) or item.size < item_size_threshold:
                per_region_info[item_name] = item

        metadata.append(per_region_info)

    return metadata
