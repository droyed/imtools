import os
import matplotlib.pyplot as plt
from imtools.color_gen import generate_colors, create_color_palette_image
from demo_config import get_output_dir


# Create output directory
output_dir = get_output_dir('color_gen')
os.makedirs(output_dir, exist_ok=True)

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

    output_path = os.path.join(output_dir, 'color_palette_demo.png')

    # Assert that the output directory exists
    assert os.path.exists(output_dir), f"Output directory not found at {output_dir}"
    print(f"Output directory created at {output_dir}")


    plt.savefig(output_path, dpi=150, bbox_inches='tight')

    # Assert that the output file exists
    assert os.path.exists(output_path), f"Output file not found at {output_path}"
    print(f"Output file created at {output_path}")


# Example usage
if __name__ == "__main__":
    # Example 1: Generate and display colors using different methods
    demo_generate_color_methods(50)
