import numpy as np
import cv2
import matplotlib.pyplot as plt
import colorsys
from matplotlib import cm


def generate_distinct_colors_hsv(n, saturation=0.9, value=0.9, shuffle=True):
    """
    Generate N distinct colors using HSV color space.
    
    Parameters:
    -----------
    n : int
        Number of colors to generate
    saturation : float
        Saturation value (0-1), default=0.9
    value : float
        Value/brightness (0-1), default=0.9
    shuffle : bool
        Whether to shuffle colors for better visual separation
    
    Returns:
    --------
    colors : numpy.ndarray
        Array of shape (n, 3) with RGB colors in range [0, 255]
    """
    colors = []
    
    for i in range(n):
        # Distribute hues evenly around color wheel
        hue = i / n
        
        # Convert HSV to RGB
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        
        # Convert to 0-255 range
        rgb_255 = np.array([int(c * 255) for c in rgb])
        colors.append(rgb_255)
    
    colors = np.array(colors)
    
    # Shuffle to avoid adjacent similar colors
    if shuffle and n > 1:
        indices = np.arange(n)
        # Use a golden ratio shuffle for better distribution
        indices = (indices * 2) % n
        colors = colors[indices]
    
    return colors


def generate_distinct_colors_golden_ratio(n, saturation=0.9, value=0.9):
    """
    Generate N distinct colors using golden ratio for hue distribution.
    This method provides better perceptual separation.
    
    Parameters:
    -----------
    n : int
        Number of colors to generate
    saturation : float
        Saturation value (0-1)
    value : float
        Value/brightness (0-1)
    
    Returns:
    --------
    colors : numpy.ndarray
        Array of shape (n, 3) with RGB colors in range [0, 255]
    """
    golden_ratio_conjugate = 0.618033988749895
    colors = []
    
    hue = np.random.rand()  # Start with random hue
    
    for i in range(n):
        # Use golden ratio to generate well-distributed hues
        hue = (hue + golden_ratio_conjugate) % 1.0
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        rgb_255 = np.array([int(c * 255) for c in rgb])
        colors.append(rgb_255)
    
    return np.array(colors)


def generate_distinct_colors_kmeans(n, n_candidates=1000, saturation=0.9, value=0.9):
    """
    Generate N maximally distinct colors using k-means clustering in color space.
    
    Parameters:
    -----------
    n : int
        Number of colors to generate
    n_candidates : int
        Number of candidate colors to cluster
    saturation : float
        Saturation value (0-1)
    value : float
        Value/brightness (0-1)
    
    Returns:
    --------
    colors : numpy.ndarray
        Array of shape (n, 3) with RGB colors in range [0, 255]
    """
    # Generate many candidate colors
    hues = np.linspace(0, 1, n_candidates, endpoint=False)
    candidates = []
    
    for hue in hues:
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        candidates.append(rgb)
    
    candidates = np.array(candidates) * 255
    
    # Use k-means to find n distinct colors
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(
        candidates.astype(np.float32),
        n,
        None,
        criteria,
        10,
        cv2.KMEANS_PP_CENTERS
    )
    
    return centers.astype(np.uint8)


def generate_distinct_colors_preset(n):
    """
    Generate N distinct colors using preset color palettes.
    Combines multiple strategies for different ranges of N.
    
    Parameters:
    -----------
    n : int
        Number of colors to generate
    
    Returns:
    --------
    colors : numpy.ndarray
        Array of shape (n, 3) with RGB colors in range [0, 255]
    """
    if n <= 0:
        return np.array([])
    
    # Predefined highly distinct colors (Kelly's 22 colors of maximum contrast)
    kelly_colors = [
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
    
    kelly_colors = np.array(kelly_colors)
    
    if n <= len(kelly_colors):
        return kelly_colors[:n]
    else:
        # For more colors, combine preset with generated
        additional_needed = n - len(kelly_colors)
        additional_colors = generate_distinct_colors_golden_ratio(additional_needed)
        return np.vstack([kelly_colors, additional_colors])


def generate_colors_from_colormap(n, colormap='tab20', return_format='uint8'):
    """
    BASIC VERSION: Simple and straightforward color generation.
    
    How it works:
    - Creates a colormap with exactly N discrete colors
    - Samples colors sequentially from index 0 to n-1
    - Each call to cmap(idx) internally normalizes idx/(n-1)
    
    Limitations:
    - Doesn't know about colormap's native capacity
    - For categorical maps with n > capacity, colors get interpolated/repeated
    - May produce similar-looking colors for large N on categorical maps
    """
    cmap = cm.get_cmap(colormap, n)
    
    colors = []
    for idx in range(n):
        color = cmap(idx)[:3]  # Takes RGB, ignores alpha
        colors.append(color)
    
    colors = np.array(colors)
    
    if return_format == 'uint8':
        colors = (colors * 255).astype(np.uint8)
    
    return colors


def generate_colors_from_colormap_extended(n, colormap='tab20', return_format='uint8'):
    """
    EXTENDED VERSION: Intelligent color generation with awareness of colormap type.
    
    How it works:
    - Knows the native capacity of categorical colormaps
    - For categorical: cycles through available colors if n > capacity
    - For continuous: samples evenly across the full colormap range
    
    Advantages:
    - Preserves distinct colors from categorical maps
    - Better distribution for continuous maps
    - Handles large N more gracefully
    """
    # Dictionary of categorical colormaps and their native capacities
    categorical_cmaps = {
        'tab20': 20, 'tab20b': 20, 'tab20c': 20,
        'Set1': 9, 'Set2': 8, 'Set3': 12,
        'Paired': 12, 'Accent': 8, 'Pastel1': 9, 'Pastel2': 8
    }
    
    if colormap in categorical_cmaps:
        max_colors = categorical_cmaps[colormap]
        cmap = cm.get_cmap(colormap)  # Get without specifying n
        
        colors = []
        for idx in range(n):
            # Cycle through available colors
            color_idx = idx % max_colors
            # Normalize to [0, 1] range
            normalized_idx = color_idx / max_colors
            color = cmap(normalized_idx)[:3]
            colors.append(color)
    else:
        # For continuous colormaps
        cmap = cm.get_cmap(colormap)
        colors = []
        for idx in range(n):
            # Sample evenly across the range
            normalized_idx = idx / max(1, n - 1) if n > 1 else 0
            color = cmap(normalized_idx)[:3]
            colors.append(color)
    
    colors = np.array(colors)
    
    if return_format == 'uint8':
        colors = (colors * 255).astype(np.uint8)
    
    return colors


def generate_colors(n, method='preset', **kwargs):
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


def create_color_palette_image(colors, cell_size=50, labels=True, palette_type='grid', show=False):
    """
    Create a visual palette showing all colors.
    
    Parameters:
    -----------
    colors : numpy.ndarray
        Array of shape (n, 3) with RGB colors in range [0, 255]
    cell_size : int
        Size of each color cell in pixels
    labels : bool
        Whether to add numbered labels
    palette_type : str
        'grid' or 'strip' layout
    show : bool
        Whether to show the palette image
    
    Returns:
    --------
    palette_image : numpy.ndarray
        RGB image showing the color palette
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
        
        palette = np.ones((rows * cell_size, cols * cell_size, 3), dtype=np.uint8) * 255
        
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


def demo_generate_color_methods(n, **kwargs):
    """
    Demonstrate different color generation methods by displaying them side by side.
    
    Parameters:
    -----------
    n : int
        Number of colors to generate for each method
    **kwargs : dict
        Additional arguments passed to generate_colors (e.g., saturation, colormap)

    Examples
    --------
    Basic usage with default parameters:

    >>> demo_generate_color_methods(50)
    >>> demo_generate_color_methods(10)  # Fewer colors for cleaner visualization

    Custom saturation (affects hsv, golden_ratio, kmeans methods):

    >>> demo_generate_color_methods(30, saturation=0.3)  # Very desaturated/grayish
    >>> demo_generate_color_methods(30, saturation=0.5)  # Muted/pastel colors
    >>> demo_generate_color_methods(30, saturation=1.0)  # Fully saturated colors

    Custom value/brightness (affects hsv, golden_ratio, kmeans methods):

    >>> demo_generate_color_methods(30, value=0.5)  # Darker colors
    >>> demo_generate_color_methods(30, value=1.0)  # Brightest colors

    Combined saturation and value:

    >>> demo_generate_color_methods(30, saturation=0.6, value=0.8)  # Soft, muted palette
    >>> demo_generate_color_methods(30, saturation=0.4, value=1.0)  # Pastel/light palette

    Different colormaps (affects colormap and colormap_extended methods):

    >>> demo_generate_color_methods(20, colormap='Set1')      # Bold, distinct colors
    >>> demo_generate_color_methods(20, colormap='Paired')    # Paired color scheme
    >>> demo_generate_color_methods(20, colormap='viridis')   # Perceptually uniform
    >>> demo_generate_color_methods(20, colormap='coolwarm')  # Blue to red diverging
    >>> demo_generate_color_methods(15, colormap='Blues')     # Monochromatic blue shades

    Method-specific parameters:

    >>> demo_generate_color_methods(20, n_candidates=2000)  # K-means: better distribution
    >>> demo_generate_color_methods(20, shuffle=False)      # HSV: sequential hues

    Theme-based examples:

    >>> demo_generate_color_methods(20, saturation=1.0, value=1.0)   # Neon/vibrant theme
    >>> demo_generate_color_methods(20, saturation=0.3, value=0.95)  # Pastel/soft theme

    Notes
    -----
    **Quick reference for parameter effects:**

    | Parameter | Affects Methods | Low Values | High Values |
    |-----------|-----------------|------------|-------------|
    | `saturation` | hsv, golden_ratio, kmeans | Grayish/muted | Vivid/bold |
    | `value` | hsv, golden_ratio, kmeans | Darker | Brighter |
    | `shuffle` | hsv | Sequential hues | Mixed hues |
    | `n_candidates` | kmeans | Faster, less optimal | Slower, better distribution |
    | `colormap` | colormap, colormap_extended | - | - |
    """
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    fig.suptitle(f'{n} Distinct Colors')
    fig.tight_layout()
    
    methods = [
        ('preset', 'Preset'),
        ('hsv', 'HSV'),
        ('golden_ratio', 'Golden Ratio'),
        ('kmeans', 'K-Means'),
        ('colormap', 'Colormap'),
        ('colormap_extended', 'Colormap Extended')
    ]

    for ax, (method, title) in zip(axes.flat, methods):
        colors = generate_colors(n, method=method, **kwargs)
        palette_grid = create_color_palette_image(colors, cell_size=80, palette_type='grid')
        ax.imshow(palette_grid)
        ax.set_title(title)
        ax.axis('off')

    plt.show()


# Example usage
if __name__ == "__main__":
    # Example 1: Generate and display colors using different methods
    demo_generate_color_methods(50)
