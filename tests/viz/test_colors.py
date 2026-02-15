"""
Tests for imtools.viz.colors module.
"""
import numpy as np
import pytest

from imtools.viz.colors import (
    generate_distinct_colors_hsv,
    generate_distinct_colors_golden_ratio,
    generate_distinct_colors_kmeans,
    generate_distinct_colors_preset,
    generate_colors_from_colormap,
    generate_colors_from_colormap_extended,
    generate_colors,
    create_color_palette_image,
)


class TestGenerateDistinctColorsHsv:
    """Tests for HSV color generation."""

    def test_generate_colors(self):
        """Test basic color generation."""
        colors = generate_distinct_colors_hsv(5)

        assert colors.shape == (5, 3)
        # Check values are in valid range regardless of dtype
        assert colors.min() >= 0
        assert colors.max() <= 255

    def test_color_range(self):
        """Test colors are in valid range."""
        colors = generate_distinct_colors_hsv(10)

        assert colors.min() >= 0
        assert colors.max() <= 255

    def test_no_shuffle(self):
        """Test without shuffle."""
        colors = generate_distinct_colors_hsv(5, shuffle=False)

        assert colors.shape == (5, 3)

    def test_zero_colors(self):
        """Test with n=0."""
        colors = generate_distinct_colors_hsv(0)

        assert len(colors) == 0


class TestGenerateDistinctColorsGoldenRatio:
    """Tests for golden ratio color generation."""

    def test_generate_colors(self):
        """Test basic color generation."""
        colors = generate_distinct_colors_golden_ratio(5)

        assert colors.shape == (5, 3)

    def test_color_range(self):
        """Test colors are in valid range."""
        colors = generate_distinct_colors_golden_ratio(10)

        assert colors.min() >= 0
        assert colors.max() <= 255


class TestGenerateDistinctColorsKmeans:
    """Tests for k-means color generation."""

    def test_generate_colors(self):
        """Test basic color generation."""
        colors = generate_distinct_colors_kmeans(5)

        assert colors.shape == (5, 3)

    def test_color_range(self):
        """Test colors are in valid range."""
        colors = generate_distinct_colors_kmeans(10)

        assert colors.min() >= 0
        assert colors.max() <= 255


class TestGenerateDistinctColorsPreset:
    """Tests for preset color generation."""

    def test_generate_colors(self):
        """Test basic color generation."""
        colors = generate_distinct_colors_preset(5)

        assert colors.shape == (5, 3)

    def test_more_colors_than_preset(self):
        """Test generating more colors than preset has."""
        colors = generate_distinct_colors_preset(30)

        assert colors.shape == (30, 3)

    def test_zero_colors(self):
        """Test with n=0."""
        colors = generate_distinct_colors_preset(0)

        assert len(colors) == 0


class TestGenerateColorsFromColormap:
    """Tests for colormap-based color generation."""

    def test_generate_colors(self):
        """Test basic color generation."""
        colors = generate_colors_from_colormap(5)

        assert colors.shape == (5, 3)

    def test_float_format(self):
        """Test with float return format."""
        colors = generate_colors_from_colormap(5, return_format='float')

        assert colors.min() >= 0.0
        assert colors.max() <= 1.0


class TestGenerateColorsFromColormapExtended:
    """Tests for extended colormap color generation."""

    def test_generate_colors(self):
        """Test basic color generation."""
        colors = generate_colors_from_colormap_extended(5)

        assert colors.shape == (5, 3)

    def test_categorical_colormap(self):
        """Test with categorical colormap."""
        colors = generate_colors_from_colormap_extended(25, colormap='tab20')

        assert colors.shape == (25, 3)

    def test_sequential_colormap(self):
        """Test with sequential colormap."""
        colors = generate_colors_from_colormap_extended(10, colormap='viridis')

        assert colors.shape == (10, 3)


class TestGenerateColors:
    """Tests for unified generate_colors function."""

    @pytest.mark.parametrize("method", [
        'preset', 'hsv', 'golden_ratio', 'kmeans',
        'colormap', 'colormap_extended'
    ])
    def test_various_methods(self, method):
        """Test all color generation methods."""
        colors = generate_colors(10, method=method)

        assert colors.shape == (10, 3)

    def test_method_aliases(self):
        """Test method name aliases."""
        colors1 = generate_colors(5, method='kelly')
        colors2 = generate_colors(5, method='preset')

        assert colors1.shape == colors2.shape

    def test_invalid_method_raises(self):
        """Test invalid method raises ValueError."""
        with pytest.raises(ValueError):
            generate_colors(5, method='invalid_method')


class TestCreateColorPaletteImage:
    """Tests for color palette visualization."""

    def test_grid_palette(self):
        """Test grid palette creation."""
        colors = generate_colors(4)
        palette = create_color_palette_image(colors, cell_size=20, palette_type='grid')

        assert isinstance(palette, np.ndarray)
        assert palette.dtype == np.uint8

    def test_strip_palette(self):
        """Test strip palette creation."""
        colors = generate_colors(4)
        palette = create_color_palette_image(colors, cell_size=20, palette_type='strip')

        assert palette.shape[0] == 20

    def test_with_labels(self):
        """Test palette with labels."""
        colors = generate_colors(2)
        palette = create_color_palette_image(colors, labels=True)

        assert palette is not None
