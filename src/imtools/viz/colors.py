"""Color generation utilities for label-overlay visualization.

Provides multiple algorithms for generating ``N`` perceptually distinct
RGB colors, plus a unified dispatcher and a palette visualization helper.
"""

from __future__ import annotations

import numpy as np
import cv2
import matplotlib.pyplot as plt
import colorsys
from matplotlib import cm
from typing import Any


def generate_distinct_colors_hsv(
    n: int,
    saturation: float = 0.9,
    value: float = 0.9,
    shuffle: bool = True,
) -> np.ndarray:
    """Generate ``n`` distinct colors by evenly distributing hues in HSV space.

    Args:
        n: Number of colors to generate.
        saturation: HSV saturation in ``[0, 1]``.  Higher values yield more
            vivid colors; lower values give pastels.
        value: HSV brightness/value in ``[0, 1]``.
        shuffle: If ``True`` (default), reorder the generated colors using
            a step of 2 to reduce visual adjacency of similar hues.

    Returns:
        ``uint8`` NumPy array of shape ``(n, 3)`` with RGB values in
        ``[0, 255]``.

    Example:
        >>> colors = generate_distinct_colors_hsv(5, saturation=0.7)
        >>> colors.shape
        (5, 3)
    """
    color_list = []

    for i in range(n):
        # Distribute hues evenly around color wheel
        hue = i / n

        # Convert HSV to RGB
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)

        # Convert to 0-255 range
        rgb_255 = np.array([int(c * 255) for c in rgb])
        color_list.append(rgb_255)

    colors = np.array(color_list)

    # Shuffle to avoid adjacent similar colors
    if shuffle and n > 1:
        indices = np.arange(n)
        # Use a golden ratio shuffle for better distribution
        indices = (indices * 2) % n
        colors = colors[indices]

    return colors


def generate_distinct_colors_golden_ratio(
    n: int,
    saturation: float = 0.9,
    value: float = 0.9,
) -> np.ndarray:
    """Generate ``n`` distinct colors using the golden-ratio hue step.

    Advances the hue by ``0.618…`` (the golden-ratio conjugate) at each step,
    starting from a random initial hue.  This spacing maximises perceptual
    distance between adjacent colors.

    Args:
        n: Number of colors to generate.
        saturation: HSV saturation in ``[0, 1]``.
        value: HSV brightness/value in ``[0, 1]``.

    Returns:
        ``uint8`` NumPy array of shape ``(n, 3)`` with RGB values in
        ``[0, 255]``.

    Example:
        >>> colors = generate_distinct_colors_golden_ratio(8)
        >>> colors.dtype
        dtype('int64')
    """
    golden_ratio_conjugate = 0.618033988749895
    color_list = []

    hue = np.random.rand()  # Start with random hue

    for i in range(n):
        # Use golden ratio to generate well-distributed hues
        hue = (hue + golden_ratio_conjugate) % 1.0
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        rgb_255 = np.array([int(c * 255) for c in rgb])
        color_list.append(rgb_255)

    return np.array(color_list)


def generate_distinct_colors_kmeans(
    n: int,
    n_candidates: int = 1000,
    saturation: float = 0.9,
    value: float = 0.9,
) -> np.ndarray:
    """Generate ``n`` maximally distinct colors using k-means clustering.

    Samples ``n_candidates`` evenly-spaced HSV hues as candidate colors,
    then applies :func:`cv2.kmeans` with k-means++ initialisation to find
    ``n`` cluster centres that are maximally separated in RGB space.

    Args:
        n: Number of colors to generate.
        n_candidates: Number of candidate colors to cluster.  Higher values
            improve quality but increase computation time.
        saturation: HSV saturation in ``[0, 1]`` for candidate generation.
        value: HSV brightness/value in ``[0, 1]`` for candidate generation.

    Returns:
        ``uint8`` NumPy array of shape ``(n, 3)`` with RGB values in
        ``[0, 255]``.

    Example:
        >>> colors = generate_distinct_colors_kmeans(10, n_candidates=500)
        >>> colors.shape
        (10, 3)
    """
    # Generate many candidate colors
    hues = np.linspace(0, 1, n_candidates, endpoint=False)
    candidates_list = []

    for hue in hues:
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        candidates_list.append(rgb)

    candidates = (np.array(candidates_list) * 255).astype(np.float32)

    # Use k-means to find n distinct colors
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    best_labels = np.empty((n_candidates, 1), dtype=np.int32)
    _, labels, centers = cv2.kmeans(
        candidates,
        n,
        best_labels,
        criteria,
        10,
        cv2.KMEANS_PP_CENTERS
    )

    return centers.astype(np.uint8)


def generate_distinct_colors_preset(n: int) -> np.ndarray:
    """Generate ``n`` distinct colors from Kelly's 21 maximally-contrast colors.

    For ``n ≤ 21``, returns the first ``n`` entries from Kelly's hand-curated
    palette.  For ``n > 21``, appends additional colors generated by
    :func:`generate_distinct_colors_golden_ratio`.

    Args:
        n: Number of colors to generate.  Returns an empty array for ``n ≤ 0``.

    Returns:
        NumPy array of shape ``(n, 3)`` with ``uint8`` RGB values in
        ``[0, 255]``.  Empty array (shape ``(0,)``) when ``n ≤ 0``.

    Example:
        >>> colors = generate_distinct_colors_preset(3)
        >>> colors[0].tolist()
        [255, 179, 0]
    """
    if n <= 0:
        return np.array([])

    # Predefined highly distinct colors (Kelly's 22 colors of maximum contrast)
    kelly_colors_list = [
        [255, 179, 0],      # Vivid Yellow
        [128, 62, 117],     # Strong Purple
        [255, 104, 0],      # Vivid Orange
        [166, 189, 215],    # Very Light Blue
        [193, 0, 32],       # Vivid Red
        [206, 162, 98],     # Grayish Yellow
        [129, 112, 102],    # Medium Gray
        [0, 125, 52],       # Vivid Green
        [246, 118, 142],    # Strong Purplish Pink
        [0, 83, 138],       # Strong Blue
        [255, 122, 92],     # Strong Yellowish Pink
        [83, 55, 122],      # Strong Violet
        [255, 142, 0],      # Vivid Orange Yellow
        [179, 40, 81],      # Strong Purplish Red
        [244, 200, 0],      # Vivid Greenish Yellow
        [127, 24, 13],      # Strong Reddish Brown
        [147, 170, 0],      # Vivid Yellowish Green
        [89, 51, 21],       # Deep Yellowish Brown
        [241, 58, 19],      # Vivid Reddish Orange
        [35, 44, 22],       # Dark Olive Green
        [0, 161, 194],      # Vivid Blue
    ]

    kelly_colors = np.array(kelly_colors_list)

    if n <= len(kelly_colors):
        return kelly_colors[:n]
    else:
        # For more colors, combine preset with generated
        additional_needed = n - len(kelly_colors)
        additional_colors = generate_distinct_colors_golden_ratio(additional_needed)
        return np.vstack([kelly_colors, additional_colors])


def generate_colors_from_colormap(
    n: int,
    colormap: str = 'tab20',
    return_format: str = 'uint8',
) -> np.ndarray:
    """Sample ``n`` colors sequentially from a Matplotlib colormap (basic version).

    Creates a colormap re-sampled to exactly ``n`` entries and samples
    sequentially from index ``0`` to ``n-1``.  For categorical maps with
    fewer than ``n`` native colors, Matplotlib will interpolate or repeat
    entries.  Use :func:`generate_colors_from_colormap_extended` for
    cycle-aware sampling.

    Args:
        n: Number of colors to generate.
        colormap: Name of any Matplotlib colormap (e.g. ``'tab20'``,
            ``'viridis'``, ``'Set2'``).
        return_format: Output dtype:

            - ``'uint8'`` (default) — values in ``[0, 255]``.
            - any other string — values in ``[0.0, 1.0]`` as ``float64``.

    Returns:
        NumPy array of shape ``(n, 3)`` in the requested dtype.

    Example:
        >>> colors = generate_colors_from_colormap(5, colormap='Set2')
        >>> colors.shape
        (5, 3)
    """
    cmap = cm.get_cmap(colormap, n)

    color_list = []
    for idx in range(n):
        color = cmap(idx)[:3]  # Takes RGB, ignores alpha
        color_list.append(color)

    colors = np.array(color_list)

    if return_format == 'uint8':
        colors = (colors * 255).astype(np.uint8)

    return colors


def generate_colors_from_colormap_extended(
    n: int,
    colormap: str = 'tab20',
    return_format: str = 'uint8',
) -> np.ndarray:
    """Sample ``n`` colors from a Matplotlib colormap with colormap-type awareness.

    Improves on :func:`generate_colors_from_colormap` by distinguishing
    categorical from continuous colormaps:

    - **Categorical** (e.g. ``tab20``, ``Set1``, ``Paired``): cycles through
      the colormap's native palette entries when ``n`` exceeds the capacity,
      preserving distinct colors rather than interpolating.
    - **Continuous** (e.g. ``viridis``, ``coolwarm``): samples evenly across
      the full ``[0, 1]`` range.

    Args:
        n: Number of colors to generate.
        colormap: Name of any Matplotlib colormap.
        return_format: Output dtype:

            - ``'uint8'`` (default) — values in ``[0, 255]``.
            - any other string — values in ``[0.0, 1.0]`` as ``float64``.

    Returns:
        NumPy array of shape ``(n, 3)`` in the requested dtype.

    Example:
        >>> colors = generate_colors_from_colormap_extended(30, colormap='tab20')
        >>> colors.shape
        (30, 3)
    """
    # Dictionary of categorical colormaps and their native capacities
    categorical_cmaps = {
        'tab20': 20, 'tab20b': 20, 'tab20c': 20,
        'Set1': 9, 'Set2': 8, 'Set3': 12,
        'Paired': 12, 'Accent': 8, 'Pastel1': 9, 'Pastel2': 8
    }

    color_list = []

    if colormap in categorical_cmaps:
        max_colors = categorical_cmaps[colormap]
        cmap = cm.get_cmap(colormap)  # Get without specifying n

        for idx in range(n):
            # Cycle through available colors
            color_idx = idx % max_colors
            # Normalize to [0, 1] range
            normalized_idx = color_idx / max_colors
            color = cmap(normalized_idx)[:3]
            color_list.append(color)
    else:
        # For continuous colormaps
        cmap = cm.get_cmap(colormap)
        for idx in range(n):
            # Sample evenly across the range
            normalized_idx = idx / max(1, n - 1) if n > 1 else 0
            color = cmap(normalized_idx)[:3]
            color_list.append(color)

    colors = np.array(color_list)

    if return_format == 'uint8':
        colors = (colors * 255).astype(np.uint8)

    return colors


def generate_colors(n: int, method: str = 'preset', **kwargs: Any) -> np.ndarray:
    """
    Unified function to generate N distinct colors using various methods.

    Parameters:
    -----------
    n : int
        Number of colors to generate
    method : str
        Color generation method. Options:
        - 'preset': Kelly's colors + golden ratio (default)
        - 'hsv': HSV color space with even hue distribution
        - 'golden_ratio': Golden ratio hue distribution
        - 'kmeans': K-means clustering for maximal distinction
        - 'colormap': Basic matplotlib colormap sampling
        - 'colormap_extended': Intelligent colormap sampling
    **kwargs : dict
        Additional arguments passed to the underlying function:
        - saturation (float): For hsv, golden_ratio, kmeans (default=0.9)
        - value (float): For hsv, golden_ratio, kmeans (default=0.9)
        - shuffle (bool): For hsv only (default=True)
        - n_candidates (int): For kmeans only (default=1000)
        - colormap (str): For colormap methods (default='tab20')
        - return_format (str): For colormap methods (default='uint8')

    Returns:
    --------
    colors : numpy.ndarray
        Array of shape (n, 3) with RGB colors in range [0, 255]

    Examples:
    ---------
    >>> colors = generate_colors(10)  # Uses preset method
    >>> colors = generate_colors(10, method='hsv', saturation=0.8)
    >>> colors = generate_colors(10, method='colormap', colormap='viridis')
    """
    method = method.lower().strip()

    # Method aliases for convenience
    method_aliases = {
        'preset': 'preset',
        'kelly': 'preset',
        'hsv': 'hsv',
        'golden': 'golden_ratio',
        'golden_ratio': 'golden_ratio',
        'goldenratio': 'golden_ratio',
        'kmeans': 'kmeans',
        'k-means': 'kmeans',
        'colormap': 'colormap',
        'cmap': 'colormap',
        'colormap_extended': 'colormap_extended',
        'cmap_extended': 'colormap_extended',
        'extended': 'colormap_extended',
    }

    # Normalize method name
    if method in method_aliases:
        method = method_aliases[method]
    else:
        valid_methods = list(set(method_aliases.values()))
        raise ValueError(f"Unknown method '{method}'. Valid methods: {valid_methods}")

    # Route to appropriate function with relevant kwargs
    if method == 'preset':
        return generate_distinct_colors_preset(n)

    elif method == 'hsv':
        hsv_kwargs = {
            'saturation': kwargs.get('saturation', 0.9),
            'value': kwargs.get('value', 0.9),
            'shuffle': kwargs.get('shuffle', True),
        }
        return generate_distinct_colors_hsv(n, **hsv_kwargs)

    elif method == 'golden_ratio':
        gr_kwargs = {
            'saturation': kwargs.get('saturation', 0.9),
            'value': kwargs.get('value', 0.9),
        }
        return generate_distinct_colors_golden_ratio(n, **gr_kwargs)

    elif method == 'kmeans':
        km_kwargs = {
            'n_candidates': kwargs.get('n_candidates', 1000),
            'saturation': kwargs.get('saturation', 0.9),
            'value': kwargs.get('value', 0.9),
        }
        return generate_distinct_colors_kmeans(n, **km_kwargs)

    elif method == 'colormap':
        cmap_kwargs = {
            'colormap': kwargs.get('colormap', 'tab20'),
            'return_format': kwargs.get('return_format', 'uint8'),
        }
        return generate_colors_from_colormap(n, **cmap_kwargs)

    elif method == 'colormap_extended':
        cmap_ext_kwargs = {
            'colormap': kwargs.get('colormap', 'tab20'),
            'return_format': kwargs.get('return_format', 'uint8'),
        }
        return generate_colors_from_colormap_extended(n, **cmap_ext_kwargs)

    # Unreachable: ValueError is raised above for unrecognized methods
    return np.array([])  # pragma: no cover


def create_color_palette_image(
    colors: np.ndarray,
    cell_size: int = 50,
    labels: bool = True,
    palette_type: str = 'grid',
    show: bool = False,
) -> np.ndarray:
    """Create a visual palette image showing all colors in a grid or strip layout.

    Args:
        colors: Array of shape ``(n, 3)`` with ``uint8`` RGB values in
            ``[0, 255]``.
        cell_size: Side length in pixels of each color swatch.
        labels: If ``True`` (default), overlay each swatch with its
            1-based index using white text with a black outline.
        palette_type: Layout style:

            - ``'grid'`` (default) — square-ish grid arrangement.
            - ``'strip'`` — single horizontal row.
        show: If ``True``, display the palette image via Matplotlib before
            returning it.

    Returns:
        ``uint8`` NumPy array of shape ``(H, W, 3)`` containing the RGB
        palette image.

    Example:
        >>> colors = generate_colors(10)
        >>> palette = create_color_palette_image(colors, cell_size=60)
        >>> palette.shape[2]
        3
    """
    n_colors = len(colors)

    if palette_type == 'strip':
        # Horizontal strip
        palette = np.zeros((cell_size, cell_size * n_colors, 3), dtype=np.uint8)

        for i, color in enumerate(colors):
            x_start = i * cell_size
            x_end = (i + 1) * cell_size
            palette[:, x_start:x_end] = color

            if labels:
                # Add label
                label_text = str(i + 1)
                text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                text_x = x_start + (cell_size - text_size[0]) // 2
                text_y = (cell_size + text_size[1]) // 2

                # White text with black outline for visibility
                cv2.putText(palette, label_text, (text_x, text_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
                cv2.putText(palette, label_text, (text_x, text_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    else:  # grid
        # Calculate grid dimensions
        cols = int(np.ceil(np.sqrt(n_colors)))
        rows = int(np.ceil(n_colors / cols))

        palette = np.full((rows * cell_size, cols * cell_size, 3), 255, dtype=np.uint8)

        for i, color in enumerate(colors):
            row = i // cols
            col = i % cols

            y_start = row * cell_size
            y_end = (row + 1) * cell_size
            x_start = col * cell_size
            x_end = (col + 1) * cell_size

            palette[y_start:y_end, x_start:x_end] = color

            if labels:
                # Add label
                label_text = str(i + 1)
                text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                text_x = x_start + (cell_size - text_size[0]) // 2
                text_y = y_start + (cell_size + text_size[1]) // 2

                # White text with black outline
                cv2.putText(palette, label_text, (text_x, text_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
                cv2.putText(palette, label_text, (text_x, text_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    if show:
        # Show the palette using matplotlib
        fig = plt.figure(figsize=(10, 10))
        plt.imshow(palette)
        plt.axis('off')
        plt.show()

    return palette
