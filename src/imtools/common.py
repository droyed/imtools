import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union


@dataclass
class Annotation:
    """
    Standardized representation of a single annotated object.

    Attributes
    ----------
    mask : np.ndarray
        2D boolean array. True where the object exists.
    text : str
        Display string rendered at the centroid.
    centroid : (int, int) or (float, float)
        Position as (x, y) — i.e. (column, row) — in image coordinates.
    score : float
        Detection confidence. Defaults to 1.0 for non-detection sources.
    """

    mask: np.ndarray
    text: str
    centroid: Union[Tuple[int, int], Tuple[float, float]]
    score: float = 1.0


@dataclass
class BlendConfig:
    """Configuration for blending operations.
    
    Attributes:
        alpha: Blending factor (0.0-1.0).
        method: Blending method name.
        params: Additional parameters for the blending method.
    """
    alpha: float = 0.3
    method: str = 'preset'
    params: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_params(cls, alpha: float = 0.3, method: str = 'preset', **kwargs) -> 'BlendConfig':
        """Create a config, capturing all remaining kwargs into params.
        
        Args:
            alpha: Blending factor (0.0-1.0).
            method: Blending method name.
            **kwargs: Additional parameters to store in params.
            
        Returns:
            A new BlendConfig instance.
        """
        return cls(alpha=alpha, method=method, params=kwargs)

    # --- The Presets Namespace ---
    class Presets:
        """Collection of standard blending configurations for color generation."""

        @staticmethod
        def categorical() -> 'BlendConfig':
            """Default preset for distinct categorical labels (e.g., object classes).
            
            Uses Kelly's maximally distinct colors + golden ratio extension.
            Good general-purpose choice for most labeling tasks.
            """
            return BlendConfig.from_params(
                alpha=0.3,
                method='preset'
            )

        @staticmethod
        def vibrant() -> 'BlendConfig':
            """Eye-catching, saturated colors for visualizations that need to pop.
            
            Best for: marketing materials, dashboards, attention-grabbing displays.
            """
            return BlendConfig.from_params(
                alpha=0.4,
                method='hsv',
                saturation=0.95,
                value=0.95,
                shuffle=True
            )

        @staticmethod
        def pastel() -> 'BlendConfig':
            """Soft, muted colors for subtle overlays and annotations.
            
            Best for: background regions, non-distracting highlights, elegant UIs.
            """
            return BlendConfig.from_params(
                alpha=0.5,
                method='hsv',
                saturation=0.4,
                value=0.95,
                shuffle=True
            )

        @staticmethod
        def high_distinction() -> 'BlendConfig':
            """Maximum color separation for many classes using k-means clustering.
            
            Best for: datasets with 20+ classes, dense segmentation maps.
            Note: Slower to compute but produces optimally distinct colors.
            """
            return BlendConfig.from_params(
                alpha=0.35,
                method='kmeans',
                n_candidates=2000,
                saturation=0.85,
                value=0.85
            )

        @staticmethod
        def subtle_overlay() -> 'BlendConfig':
            """Very light blending for non-distracting background visualization.
            
            Best for: overlays on detailed images, preserving original content visibility.
            """
            return BlendConfig.from_params(
                alpha=0.15,
                method='golden_ratio',
                saturation=0.6,
                value=0.9
            )

        @staticmethod
        def publication() -> 'BlendConfig':
            """Print-friendly colors that work well in grayscale and color.
            
            Best for: academic papers, reports, formal documentation.
            Uses Set2 colormap which is designed for clarity in print.
            """
            return BlendConfig.from_params(
                alpha=0.4,
                method='colormap',
                colormap='Set2',
                return_format='uint8'
            )

        @staticmethod
        def colorblind_safe() -> 'BlendConfig':
            """Accessible color palette safe for common color vision deficiencies.
            
            Best for: public-facing applications, accessibility-compliant outputs.
            Uses cividis colormap designed for color vision deficiency accessibility.
            """
            return BlendConfig.from_params(
                alpha=0.4,
                method='colormap_extended',
                colormap='cividis',
                return_format='uint8'
            )

        @staticmethod
        def sequential() -> 'BlendConfig':
            """Ordered color progression for ranked or continuous data.
            
            Best for: heatmaps, intensity maps, ordered categories.
            Uses viridis which is perceptually uniform and widely recognized.
            """
            return BlendConfig.from_params(
                alpha=0.5,
                method='colormap_extended',
                colormap='viridis',
                return_format='uint8'
            )

        @staticmethod
        def diverging() -> 'BlendConfig':
            """Colors diverging from a neutral center for bipolar data.
            
            Best for: difference maps, sentiment analysis, deviation from baseline.
            Uses coolwarm which clearly shows positive/negative divergence.
            """
            return BlendConfig.from_params(
                alpha=0.5,
                method='colormap_extended',
                colormap='coolwarm',
                return_format='uint8'
            )

        @staticmethod
        def bold() -> 'BlendConfig':
            """Strong, opaque colors for maximum visual impact.
            
            Best for: annotations that must stand out, presentation graphics.
            """
            return BlendConfig.from_params(
                alpha=0.6,
                method='hsv',
                saturation=0.9,
                value=0.85,
                shuffle=True
            )


@dataclass
class TitleConfig:
    """Configuration for title bar rendering."""

    # Typography
    font_path: str = "DejaVuSans-Bold.ttf"
    font_size: int = 12
    line_spacing: int = 2

    # Colors
    text_color: Union[Tuple[int, int, int], str] = (0, 0, 0)
    bg_color: Union[Tuple[int, int, int], str] = (255, 255, 255)

    # Layout
    padding: int = 15
    align: str = "left"

    # --- The Presets Namespace ---
    class Presets:
        """Collection of standard title bar configurations."""

        @staticmethod
        def publication() -> 'TitleConfig':
            """Formal style for academic papers and reports.
            
            Serif font, centered alignment, moderate size.
            Best for: journal figures, technical documentation.
            """
            return TitleConfig(
                font_path="times.ttf",
                font_size=14,
                line_spacing=3,
                text_color=(0, 0, 0),
                bg_color=(255, 255, 255),
                padding=12,
                align="center"
            )

        @staticmethod
        def presentation() -> 'TitleConfig':
            """Large, bold style for slides and large displays.
            
            High contrast, generous padding for readability at distance.
            Best for: presentations, posters, digital signage.
            """
            return TitleConfig(
                font_path="arialbd.ttf",
                font_size=24,
                line_spacing=4,
                text_color=(255, 255, 255),
                bg_color=(30, 30, 30),
                padding=25,
                align="left"
            )

        @staticmethod
        def minimal() -> 'TitleConfig':
            """Subtle, unobtrusive style for thumbnails and previews.
            
            Small font, tight spacing, muted gray tones.
            Best for: image galleries, grid layouts, subtle annotations.
            """
            return TitleConfig(
                font_path="arial.ttf",
                font_size=9,
                line_spacing=1,
                text_color=(80, 80, 80),
                bg_color=(245, 245, 245),
                padding=6,
                align="left"
            )

        @staticmethod
        def dark_mode() -> 'TitleConfig':
            """Light text on dark background for dark-themed interfaces.
            
            Best for: night mode UIs, dark backgrounds, media applications.
            """
            return TitleConfig(
                font_path="DejaVuSans.ttf",
                font_size=12,
                line_spacing=2,
                text_color=(230, 230, 230),
                bg_color=(35, 35, 35),
                padding=15,
                align="left"
            )

        @staticmethod
        def compact() -> 'TitleConfig':
            """Space-efficient style for constrained layouts.
            
            Smallest readable font, minimal padding.
            Best for: dense grids, comparison views, small containers.
            """
            return TitleConfig(
                font_path="arial.ttf",
                font_size=8,
                line_spacing=1,
                text_color=(0, 0, 0),
                bg_color=(255, 255, 255),
                padding=4,
                align="left"
            )

        @staticmethod
        def banner() -> 'TitleConfig':
            """Prominent header style for hero sections.
            
            Very large, centered, bold appearance.
            Best for: cover images, section headers, featured content.
            """
            return TitleConfig(
                font_path="arialbd.ttf",
                font_size=32,
                line_spacing=6,
                text_color=(255, 255, 255),
                bg_color=(45, 55, 72),
                padding=35,
                align="center"
            )

        @staticmethod
        def high_contrast() -> 'TitleConfig':
            """Maximum visibility for accessibility needs.
            
            Large font, generous padding, black on white.
            Best for: low-vision users, accessibility compliance, clear signage.
            """
            return TitleConfig(
                font_path="arialbd.ttf",
                font_size=18,
                line_spacing=4,
                text_color=(0, 0, 0),
                bg_color=(255, 255, 255),
                padding=20,
                align="left"
            )


@dataclass
class LabelStyle:
    """Configuration for label appearance."""
    font_size: int = 12
    padding: int = 5
    alpha: int = 100            # Box opacity (0-255)
    show_boxes: bool = True
    text_color: str = 'black'
    box_fill: tuple = (255, 255, 255)
    box_outline: str = 'black'
    box_outline_width: int = 2
    font_name: Optional[str] = "arial.ttf"

    # --- The Presets Namespace ---
    class Presets:
        """Collection of standard style configurations."""
    
        @staticmethod
        def dense_detection() -> 'LabelStyle':
            """Best for many small objects close together (e.g., crowd counting, cells)."""
            return LabelStyle(
                font_size=8,
                padding=1,
                alpha=80,
                show_boxes=True,
                text_color='white',
                box_fill=(0, 0, 0),
                box_outline='white',
                box_outline_width=1,
                font_name="arial.ttf"
            )

        @staticmethod
        def segmentation_mask() -> 'LabelStyle':
            """Best for large regions where labels should be subtle (e.g., land cover)."""
            return LabelStyle(
                font_size=14,
                padding=4,
                alpha=160,
                show_boxes=True,
                text_color='black',
                box_fill=(255, 255, 255),
                box_outline='black',
                box_outline_width=2,
                font_name="arial.ttf"
            )

        @staticmethod
        def publication() -> 'LabelStyle':
            """Minimalist style for papers/reports (Serif font, no boxes)."""
            return LabelStyle(
                font_size=16,
                padding=0,
                alpha=255,
                show_boxes=False,
                text_color='black',
                box_outline_width=0,
                font_name="times.ttf"  # Formal Serif look
            )

        @staticmethod
        def high_contrast() -> 'LabelStyle':
            """Maximum visibility for dark or noisy images (Bold font)."""
            return LabelStyle(
                font_size=12,
                padding=6,
                alpha=255,  # Fully opaque
                show_boxes=True,
                text_color='yellow',
                box_fill=(0, 0, 0),
                box_outline='yellow',
                box_outline_width=3,
                font_name="arialbd.ttf"  # Arial Bold
            )