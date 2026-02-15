"""YOLO-specific conversion utilities for the imtools package.

Provides :func:`yolo_to_label_image` which converts an Ultralytics YOLO
``Results`` object into a 2-D integer label image compatible with the
rest of the imtools visualization pipeline.
"""

from __future__ import annotations

import numpy as np
import torch
from typing import Any, Optional
from imtools.masks.converters import masks_to_label_image


def yolo_to_label_image(
    results: Any,
    use_loop: bool = True,
    use_numpy: Optional[bool] = None,
    device: Optional[str] = None,
) -> np.ndarray | torch.Tensor:
    """Convert a YOLO segmentation ``Results`` object to a 2-D label image.

    Extracts the boolean mask stack from ``results.masks.data`` and
    dispatches to :func:`~imtools.masks.converters.masks_to_label_image`
    with automatic backend selection.

    Args:
        results: Ultralytics YOLO ``Results`` object.  Must have
            ``masks``, ``orig_shape``, and ``boxes`` attributes.
        use_loop: If ``True`` (default), use the sequential loop
            implementation inside the converter.
        use_numpy: Backend selection:

            - ``None`` (default) — automatically choose based on the device
              of ``results.masks.data``: NumPy for CPU masks, PyTorch for
              GPU masks.
            - ``True`` — force NumPy backend (always returns a NumPy array).
            - ``False`` — force PyTorch backend.
        device: Target device for the PyTorch backend.  ``None`` keeps
            masks on their current device; ``'cpu'`` / ``'cuda'`` forces
            transfer.

    Returns:
        2-D integer array of shape ``(H, W)`` where ``0`` is background and
        each positive integer is a detection's 1-based index.  Returns a
        NumPy ``int32`` array when ``use_numpy`` is ``True`` or ``None``
        with CPU masks; returns a :class:`torch.Tensor` otherwise.

    Example:
        >>> # Assuming `results` is a YOLO Results object
        >>> lbl = yolo_to_label_image(results)
        >>> lbl.shape == results.orig_shape
        True
    """
    # 1. Safety Check: Handle No Detections
    if results.masks is None:
        h, w = results.orig_shape
        # If use_numpy is None (auto), default to NumPy for empty results
        # Usually, returning a CPU numpy array is safest if we don't know the device.
        if use_numpy is True or use_numpy is None:
            return np.zeros((h, w), dtype=np.int32)
        else:
            target_device = device if device else 'cpu'
            return torch.zeros((h, w), device=target_device, dtype=torch.int32)

    # 2. Get 3D masks
    masks_tensor = results.masks.data

    # 2.1. Determine Backend Automatically
    if use_numpy is None:
        # If masks are on CPU, use NumPy (faster loop).
        # If masks are on GPU, use PyTorch (avoid transfer).
        use_numpy = (masks_tensor.device.type == 'cpu')

    # 3. Conditional Conversion
    if masks_tensor.dtype != torch.bool:
        masks_tensor = masks_tensor.bool()

    # 4. Call Dispatcher
    return masks_to_label_image(
        masks_tensor,
        use_loop=use_loop,
        use_numpy=use_numpy,
        device=device
    )
