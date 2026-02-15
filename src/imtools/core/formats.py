"""Image format conversion utilities for the imtools package.

Provides bidirectional conversion between PIL Images, OpenCV-compatible
NumPy arrays, and file paths, plus a unified image-load helper and a
NumPy-to-disk writer.
"""

from __future__ import annotations

import numpy as np
from PIL import Image
from pathlib import Path
import cv2
import os
from typing import Union


def pil_to_opencv(pil_image: Image.Image, ensure_contiguous: bool = True) -> np.ndarray:
    """
    Convert a PIL Image to an OpenCV-compatible NumPy array.

    Standardizes all PIL image modes into OpenCV-compatible formats:

    Output Formats:
        - BGR (uint8): For color images (RGB, P, CMYK, YCbCr, etc.)
        - BGRA (uint8): For images with transparency (RGBA, PA, LA)
        - Grayscale (uint8): For L and 1-bit binary modes
        - Grayscale (uint16): For 16-bit modes (I;16, I;16L, I;16B)
        - Grayscale (int32): For mode 'I' (32-bit signed integer)
        - Grayscale (float32): For mode 'F' (32-bit floating point)

    Args:
        pil_image: PIL Image object to convert.
        ensure_contiguous: If True (default), guarantees the output array is
            C-contiguous, which is required by many OpenCV functions. If False,
            may return a non-contiguous view for RGB images (faster, uses less
            memory, but some OpenCV operations may fail or produce unexpected
            results).

    Returns:
        NumPy array in OpenCV-compatible format (BGR/BGRA color ordering).

    Raises:
        TypeError: If input is not a PIL Image.

    Examples:
        >>> from PIL import Image
        >>> img = Image.open('photo.jpg')
        >>> cv_img = pil_to_opencv(img)
        >>> cv_img.shape
        (480, 640, 3)

        >>> # For performance-critical code where you know the array
        >>> # will only be used with compatible operations:
        >>> cv_img = pil_to_opencv(img, ensure_contiguous=False)

    Notes:
        - Mode '1' (1-bit binary): Converted to uint8 grayscale (0 -> 0, 1 -> 255)
        - Mode 'P' (palette): Palette is resolved to actual RGB/RGBA colors
        - Mode 'PA' (palette + alpha): Converted to BGRA
        - Mode 'LA' (grayscale + alpha): Expanded to BGRA since OpenCV doesn't
          natively support 2-channel grayscale+alpha
        - Modes 'CMYK', 'YCbCr', 'LAB', 'HSV': Converted to BGR via RGB
        - The `ensure_contiguous` parameter only affects modes where a
          zero-copy view is possible (RGB, I;16). Other modes always return
          a new array due to necessary conversions or data type changes.
    """

    if not isinstance(pil_image, Image.Image):
        raise TypeError(f"Input must be a PIL Image object, got {type(pil_image)}")

    mode = pil_image.mode
    w, h = pil_image.size

    # --- Group 1: 16-bit Unsigned Modes ---
    # These require raw byte access since np.array() doesn't handle them correctly.
    # I;16L = little-endian, I;16B = big-endian, I;16 = native byte order
    if mode == 'I;16':
        raw_data = pil_image.tobytes()
        arr = np.frombuffer(raw_data, dtype=np.uint16).reshape((h, w))
        return arr.copy() if ensure_contiguous else arr
    elif mode == 'I;16L':
        raw_data = pil_image.tobytes()
        return np.frombuffer(raw_data, dtype='<u2').reshape((h, w)).astype(np.uint16)
    elif mode == 'I;16B':
        raw_data = pil_image.tobytes()
        return np.frombuffer(raw_data, dtype='>u2').reshape((h, w)).astype(np.uint16)

    # --- Group 2: 32-bit Scientific Modes ---
    # np.array() handles these correctly, preserving the data type.
    if mode in ('I', 'F'):
        return np.array(pil_image)

    # --- Group 3: Palette Modes (P and PA) ---
    # Palette images store indices into a color lookup table.
    # We must resolve these to actual color values.
    if mode == 'P':
        if 'transparency' in pil_image.info:
            pil_image = pil_image.convert('RGBA')
            mode = 'RGBA'
        else:
            pil_image = pil_image.convert('RGB')
            mode = 'RGB'
    elif mode == 'PA':
        pil_image = pil_image.convert('RGBA')
        mode = 'RGBA'

    # --- Group 4: 1-bit Binary ---
    # Convert to 8-bit grayscale. PIL's convert() scales 0/1 to 0/255.
    if mode == '1':
        pil_image = pil_image.convert('L')
        mode = 'L'

    # --- Group 5: Grayscale + Alpha ---
    # OpenCV doesn't have a native grayscale+alpha format, so expand to BGRA.
    if mode == 'LA':
        pil_image = pil_image.convert('RGBA')
        mode = 'RGBA'

    # --- Group 6: Exotic Color Spaces ---
    # Convert to RGB first, then we'll swap to BGR below.
    if mode in ('CMYK', 'YCbCr', 'LAB', 'HSV', 'RGBX', 'RGBa'):
        pil_image = pil_image.convert('RGB')
        mode = 'RGB'

    # --- Final Conversion to NumPy ---
    cv_image = np.array(pil_image)

    # --- Channel Reordering (RGB -> BGR, RGBA -> BGRA) ---
    # OpenCV uses BGR ordering, PIL uses RGB.
    if mode == 'RGB':
        cv_image = cv_image[:, :, ::-1]
        if ensure_contiguous:
            cv_image = np.ascontiguousarray(cv_image)
    elif mode == 'RGBA':
        # Fancy indexing always returns a copy, so this is already contiguous
        cv_image = cv_image[:, :, [2, 1, 0, 3]]
    # Mode 'L' passes through unchanged (already grayscale, no channel swap needed)

    return cv_image


def opencv_to_pil(cv_image: np.ndarray, channel_order: str = 'BGR') -> Image.Image:
    """
    Convert an OpenCV-compatible NumPy array to a PIL Image.

    Automatically determines the appropriate PIL mode based on array dtype and shape:

    Input → Output Mapping:
        - BGR/RGB (uint8, 3 channels) → RGB
        - BGRA/RGBA (uint8, 4 channels) → RGBA
        - 2-channel (uint8) → LA (grayscale + alpha)
        - Grayscale (uint8, 2D) → L
        - Grayscale (uint16, 2D) → I;16
        - Grayscale (int32, 2D) → I
        - Grayscale (float32, 2D) → F
        - Boolean (2D) → 1 (1-bit binary)

    Args:
        cv_image: NumPy array from OpenCV or similar source. Can be 2D (grayscale)
            or 3D (color/multi-channel).
        channel_order: Channel ordering for 3/4-channel uint8 images.
            - 'BGR': OpenCV default (converts to RGB/RGBA for PIL)
            - 'RGB': Already in PIL order (no conversion needed)
            Ignored for grayscale and non-uint8 images.

    Returns:
        PIL Image object with appropriate mode for the input data.

    Raises:
        TypeError: If input is not a NumPy array.
        ValueError: If array has unsupported dtype, shape, or channel count.

    Examples:
        >>> import cv2
        >>> cv_img = cv2.imread('photo.jpg')
        >>> pil_img = opencv_to_pil(cv_img)
        >>> pil_img.mode
        'RGB'

        >>> # For arrays already in RGB order (e.g., from matplotlib):
        >>> pil_img = opencv_to_pil(rgb_array, channel_order='RGB')

        >>> # Grayscale images:
        >>> gray = cv2.imread('photo.jpg', cv2.IMREAD_GRAYSCALE)
        >>> pil_img = opencv_to_pil(gray)
        >>> pil_img.mode
        'L'

        >>> # 16-bit depth images:
        >>> depth = cv2.imread('depth.png', cv2.IMREAD_UNCHANGED)  # uint16
        >>> pil_img = opencv_to_pil(depth)
        >>> pil_img.mode
        'I;16'

    Notes:
        - Boolean arrays are converted to mode '1' (1-bit binary).
        - 2-channel uint8 arrays are interpreted as grayscale + alpha (LA mode).
        - For 3/4-channel non-uint8 arrays, a ValueError is raised since these
          don't have a standard PIL equivalent.
        - Non-contiguous arrays are handled automatically.
        - float64 arrays must be converted to float32 before calling this function.
    """

    if not isinstance(cv_image, np.ndarray):
        raise TypeError(f"Input must be a NumPy array, got {type(cv_image)}")

    if channel_order not in ('BGR', 'RGB'):
        raise ValueError(f"channel_order must be 'BGR' or 'RGB', got '{channel_order}'")

    # Ensure we have a contiguous array for PIL
    if not cv_image.flags['C_CONTIGUOUS']:
        cv_image = np.ascontiguousarray(cv_image)

    ndim = cv_image.ndim
    dtype = cv_image.dtype

    # --- 2D Arrays (Grayscale / Single-channel) ---
    if ndim == 2:
        if dtype == np.uint8:
            return Image.fromarray(cv_image, mode='L')

        elif dtype == np.uint16:
            # PIL's I;16 mode expects little-endian uint16
            return Image.fromarray(cv_image, mode='I;16')

        elif dtype == np.int32:
            return Image.fromarray(cv_image, mode='I')

        elif dtype == np.float32:
            return Image.fromarray(cv_image, mode='F')

        elif dtype == np.bool_:
            # Convert boolean to PIL mode '1' (1-bit pixels)
            return Image.fromarray(cv_image, mode='1')

        else:
            raise ValueError(
                f"Unsupported dtype '{dtype}' for 2D array. "
                f"Supported: uint8 (L), uint16 (I;16), int32 (I), float32 (F), bool (1)"
            )

    # --- 3D Arrays (Multi-channel) ---
    elif ndim == 3:
        height, width, channels = cv_image.shape

        if dtype == np.uint8:
            if channels == 3:
                # BGR → RGB or RGB → RGB
                if channel_order == 'BGR':
                    rgb_image = cv_image[:, :, ::-1]
                    rgb_image = np.ascontiguousarray(rgb_image)
                else:
                    rgb_image = cv_image
                return Image.fromarray(rgb_image, mode='RGB')

            elif channels == 4:
                # BGRA → RGBA or RGBA → RGBA
                if channel_order == 'BGR':
                    rgba_image = cv_image[:, :, [2, 1, 0, 3]]
                else:
                    rgba_image = cv_image
                return Image.fromarray(rgba_image, mode='RGBA')

            elif channels == 2:
                # Grayscale + Alpha → LA
                return Image.fromarray(cv_image, mode='LA')

            elif channels == 1:
                # Single-channel 3D array → squeeze to 2D grayscale
                return Image.fromarray(cv_image[:, :, 0], mode='L')

            else:
                raise ValueError(
                    f"Unsupported channel count {channels} for uint8 array. "
                    f"Supported: 1 (L), 2 (LA), 3 (RGB), 4 (RGBA)"
                )

        elif dtype in (np.uint16, np.int32, np.float32):
            # For non-uint8, only single-channel makes sense
            if channels == 1:
                return opencv_to_pil(cv_image[:, :, 0], channel_order)
            else:
                raise ValueError(
                    f"Multi-channel arrays with dtype '{dtype}' are not supported. "
                    f"Only single-channel (grayscale) arrays are valid for this dtype."
                )

        else:
            raise ValueError(
                f"Unsupported dtype '{dtype}' for 3D array. "
                f"Supported: uint8, uint16, int32, float32"
            )

    else:
        raise ValueError(
            f"Array must be 2D or 3D, got {ndim}D array with shape {cv_image.shape}"
        )


def to_pil_image(source: Union[str, Path, Image.Image, np.ndarray]) -> Image.Image:
    """Convert various image sources to a PIL Image.

    Args:
        source: Image source — one of:

            - :class:`PIL.Image.Image`: Passed through unchanged.
            - :class:`numpy.ndarray`: Converted via :func:`opencv_to_pil`.
            - :class:`str` / :class:`pathlib.Path`: Loaded from file path.

    Returns:
        PIL Image object.

    Raises:
        TypeError: If ``source`` is not a supported type.
        FileNotFoundError: If a path-based source does not exist.

    Example:
        >>> from PIL import Image
        >>> img = to_pil_image("photo.jpg")
        >>> isinstance(img, Image.Image)
        True
    """
    # Already a PIL Image - pass through
    if isinstance(source, Image.Image):
        return source

    # Numpy array - use custom converter
    if isinstance(source, np.ndarray):
        return opencv_to_pil(source)

    # File path
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        return Image.open(path)

    raise TypeError(
        f"Unsupported image source type: {type(source).__name__}. "
        f"Expected PIL.Image, numpy.ndarray, or path string."
    )

def to_numpy_image(
    source: Union[str, Path, Image.Image, np.ndarray],
    *,
    normalize_dims: bool = False,
    force_3d: bool = False,
    drop_alpha: bool = False,
    force_uint8: bool = False,
    force_copy: bool = False,
) -> np.ndarray:
    """Load an image as a standard RGB/RGBA NumPy array with optional transforms.

    All file-path sources are loaded with :func:`cv2.imread` (``UNCHANGED``),
    then converted from OpenCV BGR/BGRA ordering to standard RGB/RGBA so
    the returned array follows the NumPy/PIL convention throughout.

    Args:
        source: Image source — one of:

            - ``str`` / ``pathlib.Path``: File path loaded via OpenCV.
            - ``PIL.Image.Image``: Converted to a NumPy array directly.
            - ``numpy.ndarray``: Used as-is (assumed to be in RGB order).
        normalize_dims: If ``True``, expand a 2-D ``(H, W)`` grayscale
            array to ``(H, W, 1)`` before any further processing.
        force_3d: If ``True``, replicate grayscale channels so the output
            has exactly 3 channels: ``(H, W) → (H, W, 3)`` and
            ``(H, W, 1) → (H, W, 3)``.
        drop_alpha: If ``True``, discard the alpha channel from a 4-channel
            ``(H, W, 4)`` array, yielding ``(H, W, 3)``.
        force_uint8: If ``True``, convert the array to ``uint8``:

            - float in ``[0, 1]`` → scaled to ``[0, 255]``.
            - ``uint16`` → shifted by 8 bits (``// 256``).
            - other integer types → clipped to ``[0, 255]``.
        force_copy: If ``True``, always return a new array (safe for
            in-place modification regardless of the source).

    Returns:
        NumPy array in RGB (or RGBA) channel order.  Shape and dtype depend
        on the source and the flags passed.

    Raises:
        FileNotFoundError: If a path-based source does not exist.
        TypeError: If ``source`` is not a supported type.
        ValueError: If OpenCV fails to decode the file.

    Example:
        >>> img = to_numpy_image("photo.jpg", force_3d=True, force_uint8=True)
        >>> img.shape  # (H, W, 3)
        >>> img.dtype
        dtype('uint8')
    """
    img = None

    # --- 1. Load Source (Standardize to RGB/RGBA) ---
    if isinstance(source, (str, Path)):
        path = str(source)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image not found: {path}")

        # Load UNCHANGED to preserve Alpha/Bit-depth initially
        # We handle channel conversions manually later
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Failed to read: {path}")

        # Convert OpenCV BGR/BGRA -> Standard RGB/RGBA
        if img.ndim == 3:
            if img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            elif img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)

    elif isinstance(source, Image.Image):
        # PIL is already RGB/RGBA
        img = np.array(source)

    elif isinstance(source, np.ndarray):
        img = source  # Assume user provided standard RGB layout

    else:
        raise TypeError(f"Unsupported type: {type(source)}")

    # --- 2. Normalize Dimensions (Expansion) ---
    # (H, W) -> (H, W, 1)
    # We do this FIRST so subsequent steps can assume at least 3 dims
    if normalize_dims and img.ndim == 2:
        img = img[..., None]

    # --- 3. Ensure 3 Channels (Replication) ---
    # Handles: (H, W) -> (H, W, 3) AND (H, W, 1) -> (H, W, 3)
    if force_3d:
        # Case A: 2D Grayscale (H, W) -> Stack to (H, W, 3)
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)

        # Case B: 3D Grayscale (H, W, 1) -> Repeat to (H, W, 3)
        elif img.ndim == 3 and img.shape[2] == 1:
            img = np.repeat(img, 3, axis=2)

        # Case C: Already 3+ channels -> Do nothing

    # --- 4. Drop Alpha ---
    # Standard (H, W, 4) -> (H, W, 3)
    if drop_alpha and img.ndim == 3 and img.shape[2] == 4:
        img = img[..., :3]

    # --- 5. Force Uint8 ---
    if force_uint8 and img.dtype != np.uint8:
        # Float [0.0, 1.0] -> [0, 255]
        if np.issubdtype(img.dtype, np.floating):
            img = np.clip(img, 0.0, 1.0)
            img = (img * 255).round().astype(np.uint8)
        # Uint16 [0, 65535] -> [0, 255]
        elif img.dtype == np.uint16:
            img = (img // 256).astype(np.uint8)
        # Other types -> Clip
        else:
            img = np.clip(img, 0, 255).astype(np.uint8)

    # --- 6. Force Copy ---
    if force_copy:
        # Handles views, non-contiguous arrays, or user request for safety
        img = img.copy()

    return img


def imwrite(
    image_arr: np.ndarray,
    file_path: Union[str, Path],
    input_channel_order: str = 'RGB',
    quality: int = 85,
    optimize: bool = True,
) -> None:
    """Save a NumPy image array to disk using Pillow.

    Handles dtype normalisation (``bool``, ``float``, ``uint16``, ``uint8``)
    and channel-order rewriting (BGR/BGRA → RGB/RGBA) before saving.

    Args:
        image_arr: Image array to save.  Supported dtypes and shapes:

            - ``bool`` — converted to ``uint8`` ``{0, 255}``.
            - ``float32`` / ``float64`` — must be in ``[0.0, 1.0]``; scaled
              to ``uint8``.
            - ``uint16`` — down-scaled to ``uint8`` (via ``// 256``) unless
              the output format is PNG or TIFF *and* the array is 2-D.
            - ``uint8`` — used as-is.
        file_path: Destination path.  The extension determines the file
            format (``'.jpg'``, ``'.jpeg'``, ``'.png'``, ``'.tiff'``, …).
        input_channel_order: Channel ordering of ``image_arr``.
            Use ``'BGR'`` for OpenCV arrays (channels are reversed before
            saving); ``'RGB'`` (default) leaves channels unchanged.
        quality: JPEG compression quality in ``[0, 100]``.  Ignored for
            non-JPEG formats.
        optimize: Whether to apply format-specific optimisation.  For JPEG
            this enables Huffman table optimisation; for PNG it enables
            compression optimisation.

    Raises:
        ValueError: If a float array contains values outside ``[0.0, 1.0]``,
            or if the array has an unsupported number of channels.

    Example:
        >>> import numpy as np
        >>> img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        >>> imwrite(img, "output.png")
        Saved to output.png
    """

    # --- 1. Datatype Normalization ---
    # Handle Bool
    if image_arr.dtype == bool:
        image_arr = image_arr.astype(np.uint8) * 255

    # Handle Float (Strict 0.0-1.0)
    elif np.issubdtype(image_arr.dtype, np.floating):
        if image_arr.min() < 0.0 or image_arr.max() > 1.0:
            raise ValueError("Float images must be strictly in 0.0-1.0 range.")
        image_arr = (image_arr * 255).astype(np.uint8)

    # Handle Uint16 (Downscale for compatibility if not PNG/TIFF)
    elif image_arr.dtype == np.uint16:
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in ['.png', '.tiff', '.tif'] or image_arr.ndim == 3:
             image_arr = (image_arr // 256).astype(np.uint8)

    # --- 2. Channel Reordering (BGR/BGRA -> RGB/RGBA) ---
    if input_channel_order.upper() == 'BGR' and image_arr.ndim == 3:
        channels = image_arr.shape[2]
        if channels == 3:
            # BGR -> RGB (Reverse all)
            image_arr = image_arr[..., ::-1]
        elif channels == 4:
            # BGRA -> RGBA (Swap B and R, keep Alpha last)
            # [0,1,2,3] -> [2,1,0,3]
            image_arr = image_arr[..., [2, 1, 0, 3]]

    # --- 3. Determine Mode ---
    if image_arr.ndim == 2:
        mode = 'L'
    elif image_arr.shape[2] == 3:
        mode = 'RGB'
    elif image_arr.shape[2] == 4:
        mode = 'RGBA'
    else:
        raise ValueError(f"Unsupported channel count: {image_arr.shape[2]}")

    # --- 4. Create Pillow Image ---
    img = Image.fromarray(image_arr, mode=mode)

    # --- 5. Save ---
    ext = os.path.splitext(file_path)[1].lower()

    if ext in ['.jpg', '.jpeg']:
        if mode == 'RGBA':
            # Composite on white for JPEG
            bg = Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[3])
            img = bg
        img.save(file_path, quality=quality, optimize=optimize)

    elif ext == '.png':
        img.save(file_path, optimize=optimize)

    else:
        img.save(file_path)

    print(f"Saved to {file_path}")
