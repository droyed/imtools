"""Mask-to-label-image converters for the imtools package.

Provides multiple backends (NumPy and PyTorch, loop and vectorised) for
converting a stack of binary masks ``(N, H, W)`` into a single 2-D label
image ``(H, W)``, plus a connected-component converter for single masks.
"""

from __future__ import annotations

import numpy as np
import torch
import cv2
from typing import Optional


# --- PyTorch Backend Converters ---

def masks_to_label_image_torch_vectorized(
    masks_array: torch.Tensor | np.ndarray,
    device: Optional[torch.device | str] = None,
) -> np.ndarray:
    """Convert a mask stack to a label image using PyTorch broadcasting (GPU-optimised).

    Each pixel is assigned the label index of the highest-indexed mask that
    covers it.  When multiple masks overlap the last one (highest index) wins.

    Args:
        masks_array: Boolean or numeric mask stack of shape ``(N, H, W)``.
            Accepts either a :class:`torch.Tensor` or a :class:`numpy.ndarray`.
        device: Target computation device.  If ``None``, the tensor's current
            device is preserved; if input is a NumPy array it defaults to CPU.

    Returns:
        2-D ``int32`` NumPy array of shape ``(H, W)`` where each element is
        the 1-based label index of the last overlapping mask (0 = background).

    Example:
        >>> import numpy as np
        >>> masks = np.zeros((3, 4, 4), dtype=bool)
        >>> masks[0, :2, :2] = True
        >>> masks[1, 1:3, 1:3] = True
        >>> lbl = masks_to_label_image_torch_vectorized(masks)
        >>> lbl.shape
        (4, 4)
    """
    # 1. Prepare Tensor
    if isinstance(masks_array, torch.Tensor):
        if device is not None:
            masks_tensor = masks_array.to(device)
        else:
            masks_tensor = masks_array  # Keep existing device
    else:
        # If input is NumPy and device is None, defaults to CPU
        masks_tensor = torch.tensor(masks_array, device=device)

    # 2. Resolve Actual Device
    # We must use the tensor's actual device for the indices to avoid mismatches
    actual_device = masks_tensor.device

    n, h, w = masks_tensor.shape

    # 3. Create indices on the SAME device
    indices = torch.arange(1, n + 1, dtype=torch.int32, device=actual_device).view(n, 1, 1)

    # 4. Broadcast & Reduce
    # (N, H, W) * (N, 1, 1) -> (N, H, W). Max over dim 0 gives the highest index (label).
    label_image = (masks_tensor * indices).max(dim=0).values

    return label_image.cpu().numpy()


def masks_to_label_image_torch_loop(
    masks_array: torch.Tensor | np.ndarray,
    device: Optional[torch.device | str] = None,
) -> np.ndarray:
    """Convert a mask stack to a label image using a PyTorch loop.

    Iterates over masks one at a time, writing each mask's 1-based index into
    the output tensor.  Useful as a memory-efficient fallback when the
    vectorised variant would allocate a prohibitively large intermediate tensor.

    Args:
        masks_array: Boolean or numeric mask stack of shape ``(N, H, W)``.
            Accepts either a :class:`torch.Tensor` or a :class:`numpy.ndarray`.
        device: Target computation device.  ``None`` preserves the input
            tensor's device (or defaults to CPU for NumPy input).

    Returns:
        2-D ``int32`` NumPy array of shape ``(H, W)``.  Later masks overwrite
        earlier ones in overlapping regions.

    Example:
        >>> import torch
        >>> masks = torch.zeros(2, 4, 4, dtype=torch.bool)
        >>> masks[0, :2, :2] = True
        >>> lbl = masks_to_label_image_torch_loop(masks)
        >>> lbl[0, 0]
        1
    """
    # 1. Prepare Tensor
    if isinstance(masks_array, torch.Tensor):
        if device is not None:
            masks = masks_array.to(device)
        else:
            masks = masks_array
    else:
        masks = torch.as_tensor(masks_array, device=device)

    actual_device = masks.device
    n, h, w = masks.shape

    # Create result tensor on the correct device
    label_image = torch.zeros((h, w), dtype=torch.int32, device=actual_device)

    for i in range(n):
        mask_slice = masks[i]
        if mask_slice.dtype != torch.bool:
             mask_slice = mask_slice.bool()
        label_image[mask_slice] = i + 1

    return label_image.cpu().numpy()


# --- NumPy Backend Converters ---

def masks_to_label_image_numpy_loop(masks_3d: np.ndarray | torch.Tensor) -> np.ndarray:
    """Convert a mask stack to a label image using a NumPy loop.

    Args:
        masks_3d: Boolean or numeric mask stack of shape ``(N, H, W)``.
            A :class:`torch.Tensor` is automatically moved to CPU and
            converted to a NumPy array.

    Returns:
        2-D ``int32`` NumPy array of shape ``(H, W)``.  Later masks
        overwrite earlier ones in overlapping regions.

    Example:
        >>> import numpy as np
        >>> masks = np.eye(3, dtype=bool)[..., np.newaxis].transpose(2, 0, 1)
        >>> masks_to_label_image_numpy_loop(masks).shape
        (3, 1)
    """
    if isinstance(masks_3d, torch.Tensor):
        masks_3d = masks_3d.detach().cpu().numpy()

    n_masks, h, w = masks_3d.shape
    label_image = np.zeros((h, w), dtype=np.int32)
    masks_bool = masks_3d.astype(bool, copy=False)

    for i in range(n_masks):
        label_image[masks_bool[i]] = i + 1

    return label_image


def masks_to_label_image_numpy_vectorized(masks_3d: np.ndarray | torch.Tensor) -> np.ndarray:
    """Convert a mask stack to a label image using NumPy broadcasting.

    Creates an ``(N, H, W)`` intermediate array by multiplying the boolean
    masks by their 1-based indices, then takes the element-wise maximum
    over the N dimension.

    Args:
        masks_3d: Boolean or numeric mask stack of shape ``(N, H, W)``.
            A :class:`torch.Tensor` is automatically moved to CPU and
            converted to a NumPy array.

    Returns:
        2-D ``int32`` NumPy array of shape ``(H, W)``.

    Example:
        >>> import numpy as np
        >>> masks = np.zeros((2, 3, 3), dtype=bool)
        >>> masks[0, 0, :] = True
        >>> masks[1, 1, :] = True
        >>> masks_to_label_image_numpy_vectorized(masks)[0, 0]
        1
    """
    if isinstance(masks_3d, torch.Tensor):
        masks_3d = masks_3d.detach().cpu().numpy()

    label_image = (masks_3d * np.arange(1, len(masks_3d)+1)[:, None, None]).max(axis=0)
    return label_image.astype(np.int32)


# --- Single Mask Converter ---

def binary_mask_to_label_image(mask: np.ndarray, connectivity: int = 8) -> np.ndarray:
    """Convert a single binary mask into a label image via connected components.

    Uses :func:`cv2.connectedComponents` to assign a unique integer label to
    each connected component of the foreground.

    Args:
        mask: 2-D binary array of shape ``(H, W)`` with foreground pixels
            set to any non-zero value.
        connectivity: Pixel connectivity — ``4`` (edge-connected) or
            ``8`` (edge- and corner-connected).

    Returns:
        2-D NumPy integer array of shape ``(H, W)`` where background pixels
        are ``0`` and each connected component has a unique label
        ``1, 2, …, K``.

    Example:
        >>> import numpy as np
        >>> mask = np.array([[1, 0, 1], [0, 0, 1], [1, 0, 0]], dtype=np.uint8)
        >>> lbl = binary_mask_to_label_image(mask)
        >>> len(np.unique(lbl)) - 1  # number of components (excl. background)
        3
    """
    # Ensure mask is uint8 for OpenCV
    mask_uint8 = mask.astype(np.uint8, copy=False)
    _, label_image = cv2.connectedComponents(mask_uint8, connectivity=connectivity)

    return label_image


# --- Main Dispatchers ---

def masks_to_label_image(
    masks_array: np.ndarray | torch.Tensor,
    use_loop: bool = True,
    use_numpy: bool = True,
    device: Optional[torch.device | str] = None,
) -> np.ndarray:
    """Dispatch mask-stack conversion to the appropriate backend and strategy.

    Args:
        masks_array: Boolean or numeric mask stack of shape ``(N, H, W)``.
        use_loop: If ``True`` (default), use the sequential loop
            implementation.  If ``False``, use the vectorised (broadcasting)
            implementation.
        use_numpy: If ``True`` (default), use the NumPy backend and ignore
            ``device``.  If ``False``, use the PyTorch backend.
        device: Target device for the PyTorch backend only.
            ``None`` — keep tensor on its current device (or CPU for NumPy
            input).  ``'cpu'`` / ``'cuda'`` — force the specified device.

    Returns:
        2-D ``int32`` NumPy array of shape ``(H, W)``.

    Example:
        >>> import numpy as np
        >>> masks = (np.random.rand(5, 64, 64) > 0.5)
        >>> lbl = masks_to_label_image(masks)
        >>> lbl.shape
        (64, 64)
    """
    if use_numpy:
        # NumPy Backend
        if use_loop:
            return masks_to_label_image_numpy_loop(masks_array)
        else:
            return masks_to_label_image_numpy_vectorized(masks_array)
    else:
        # PyTorch Backend
        if use_loop:
            return masks_to_label_image_torch_loop(masks_array, device=device)
        else:
            return masks_to_label_image_torch_vectorized(masks_array, device=device)
