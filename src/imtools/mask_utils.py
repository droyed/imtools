import numpy as np
import cv2
from skimage.measure import label, regionprops
from scipy.ndimage import binary_closing
from skimage.segmentation import flood_fill


def generate_label_image(mask, connectivity=8):
    """
    Generates a label image from a binary mask using connected components.
    
    Args:
        mask (np.ndarray): Binary mask to segment.
        connectivity (int): Pixel connectivity (4 or 8).
    
    Returns:
        np.ndarray: Label image where each connected component has a unique integer label.
    """
    # Ensure mask is uint8 for OpenCV
    mask_uint8 = mask.astype(np.uint8)
    _, label_image = cv2.connectedComponents(mask_uint8, connectivity=connectivity)
    
    return label_image

def _extract_properties(regionprops_object):
    """
    Extract properties from a regionprops object
    """
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

def _extract_info(regionprops_object):
    """
    Extract information from a regionprops object
    """
    available_props = _extract_properties(regionprops_object)
    largest_region_props = {
        prop: getattr(regionprops_object, prop)
        for prop in available_props
    }
    return largest_region_props
    
def get_biggest_blob(mask: np.ndarray, return_props: bool = False) -> np.ndarray:
    """
    Returns a binary mask containing only the largest blob.

    Uses scikit-image regionprops with 8-connectivity to identify connected
    components and extracts the blob with the largest area (pixel count).

    Parameters:
    -----------
    mask : np.ndarray
        Binary mask array with values 0 and 1
    return_largest_region_props : bool
        If True, returns the properties of the largest blob
    Returns:
    --------
    np.ndarray
        Binary mask with only the largest blob (same shape as input).
        Returns array of zeros if input mask is empty.

    Raises:
    -------
    AssertionError
        If input is not a numpy array or contains non-binary values

    Examples:
    ---------
    >>> mask = np.array([[1, 1, 0, 0],
    ...                  [1, 0, 0, 1],
    ...                  [0, 0, 1, 1]])
    >>> result = get_biggest_blob(mask)
    >>> result
    array([[1, 1, 0, 0],
           [1, 0, 0, 0],
           [0, 0, 0, 0]])
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
    """
    Compute the axis-aligned bounding box of a 2D boolean mask.

    The returned bounding box follows the convention:
    `[col_start, row_start, col_end, row_end]`, where `col_end` and `row_end`
    are **exclusive** (i.e., one past the last True pixel along each axis).

    Parameters:
    -----------
    mask : np.ndarray
        2D boolean array. True values indicate the foreground.

    Returns:
    --------
    list[int]
        Bounding box as `[col_start, row_start, col_end, row_end]` using
        0-based indexing and exclusive end coordinates.

        Notes on edge cases:
        - If `mask` contains no True pixels, this implementation returns the
          full-image box `[0, 0, n, m]` (because `argmax()` returns 0 on an
          all-False array).

    Raises:
    -------
    AssertionError
        If `mask` is not a numpy array, is not 2D, or is not of boolean dtype.
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

def fill_holes_mask(mask, apply_closing=True, kernel_size=5, pad_length=5):
    """
    Process a binary mask by applying morphological closing and filling holes.

    Args:
        mask: Binary mask (numpy array)
        apply_closing: Whether to apply morphological closing (default: True)
        kernel_size: Size of the kernel for closing operation (default: 5)
        pad_length: Padding length for flood fill operation (default: 5)

    Returns:
        output_mask: Processed binary mask with holes filled
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