import os
import cv2
import numpy as np
import torch
from PIL import Image
from typing import Union


def load_image(
    image: Union[str, Image.Image, torch.Tensor, np.ndarray],
    *,
    drop_alpha: bool = False,
    normalize_dims: bool = False,
    force_uint8: bool = False,
    convert_to_rgb: bool = False,
) -> np.ndarray:
    """
    Load image from multiple input types.

    Parameters
    ----------
    image:
        Image path, PIL Image, NumPy array, or Torch tensor
    drop_alpha:
        Drop alpha channel if present (ignored if convert_to_rgb=True)
    normalize_dims:
        Normalize shape to (H, W, C). If False, preserve original dimensions.
    force_uint8:
        If True, convert output to np.uint8.
        If False, preserve original dtype.
    convert_to_rgb:
        If True, output will be RGB with exactly 3 channels.

    Returns
    -------
    np.ndarray
    """

    # ---------- Load to NumPy ----------
    if isinstance(image, str):
        if not os.path.exists(image):
            raise FileNotFoundError(f"Image file not found: {image}")

        img = cv2.imread(image, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Failed to read image: {image}")

        # OpenCV loads as BGR / BGRA
        if convert_to_rgb:
            if img.ndim == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            elif img.ndim == 3 and img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)

    elif isinstance(image, Image.Image):
        if convert_to_rgb:
            img = np.array(image.convert("RGB"))
        else:
            img = np.array(image)

    elif isinstance(image, torch.Tensor):
        if image.is_cuda:
            image = image.cpu()

        img = image.detach().numpy()

        # (C, H, W) -> (H, W, C)
        if normalize_dims and img.ndim == 3 and img.shape[0] in (1, 3, 4):
            img = np.transpose(img, (1, 2, 0))

    elif isinstance(image, np.ndarray):
        img = image.copy()

    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    # ---------- Normalize shape ----------
    if normalize_dims:
        # Grayscale -> (H, W, 1)
        if img.ndim == 2:
            img = img[..., None]

        # Enforce RGB output
        if convert_to_rgb:
            if img.ndim == 3 and img.shape[2] == 4:
                img = img[..., :3]
            elif img.ndim == 3 and img.shape[2] == 1:
                img = np.repeat(img, 3, axis=2)

        # Optional alpha drop
        elif img.ndim == 3 and img.shape[2] == 4 and drop_alpha:
            img = img[..., :3]

    # ---------- Normalize dtype ----------
    if force_uint8:
        if img.dtype == np.bool_:
            img = img.astype(np.uint8) * 255

        elif np.issubdtype(img.dtype, np.floating):
            img = np.clip(img, 0.0, 1.0)
            img = (img * 255).round().astype(np.uint8)

        elif img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)

    return img
