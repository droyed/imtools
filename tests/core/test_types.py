"""
Tests for imtools.core.types module.
"""
import numpy as np
import pytest
from imtools.core.types import (
    Annotation,
    BlendConfig,
    TitleConfig,
    LabelStyle,
)


class TestAnnotation:
    """Tests for Annotation dataclass."""

    def test_annotation_creation(self):
        """Test creating an Annotation with required fields."""
        mask = np.zeros((10, 10), dtype=np.bool_)
        mask[2:5, 2:5] = True

        ann = Annotation(mask=mask, text="test", centroid=(5, 5))

        assert ann.mask is mask
        assert ann.text == "test"
        assert ann.centroid == (5, 5)
        assert ann.score == 1.0  # default

    def test_annotation_with_score(self):
        """Test creating an Annotation with custom score."""
        mask = np.zeros((10, 10), dtype=np.bool_)
        ann = Annotation(mask=mask, text="test", centroid=(5, 5), score=0.85)

        assert ann.score == 0.85


class TestBlendConfig:
    """Tests for BlendConfig dataclass."""

    def test_blend_config_defaults(self):
        """Test default BlendConfig values."""
        config = BlendConfig()

        assert config.alpha == 0.3
        assert config.method == 'preset'
        assert config.params == {}

    def test_blend_config_from_params(self):
        """Test creating BlendConfig with from_params."""
        config = BlendConfig.from_params(alpha=0.5, method='hsv', saturation=0.8)

        assert config.alpha == 0.5
        assert config.method == 'hsv'
        assert config.params == {'saturation': 0.8}

    def test_blend_config_presets(self):
        """Test all BlendConfig presets."""
        presets = [
            'categorical', 'vibrant', 'pastel', 'high_distinction',
            'subtle_overlay', 'publication', 'colorblind_safe',
            'sequential', 'diverging', 'bold'
        ]

        for preset_name in presets:
            preset_func = getattr(BlendConfig.Presets, preset_name)
            config = preset_func()
            assert isinstance(config, BlendConfig)
            assert 0.0 <= config.alpha <= 1.0


class TestTitleConfig:
    """Tests for TitleConfig dataclass."""

    def test_title_config_defaults(self):
        """Test default TitleConfig values."""
        config = TitleConfig()

        assert config.font_path == "DejaVuSans-Bold.ttf"
        assert config.font_size == 12
        assert config.padding == 15
        assert config.align == "left"

    def test_title_config_presets(self):
        """Test all TitleConfig presets."""
        preset_names = [
            'publication', 'presentation', 'minimal',
            'dark_mode', 'compact', 'banner', 'high_contrast'
        ]

        for preset_name in preset_names:
            preset_func = getattr(TitleConfig.Presets, preset_name)
            config = preset_func()
            assert isinstance(config, TitleConfig)


class TestLabelStyle:
    """Tests for LabelStyle dataclass."""

    def test_label_style_defaults(self):
        """Test default LabelStyle values."""
        style = LabelStyle()

        assert style.font_size == 12
        assert style.padding == 5
        assert style.alpha == 100
        assert style.show_boxes is True

    def test_label_style_presets(self):
        """Test all LabelStyle presets."""
        preset_names = ['dense_detection', 'segmentation_mask', 'publication', 'high_contrast']

        for preset_name in preset_names:
            preset_func = getattr(LabelStyle.Presets, preset_name)
            style = preset_func()
            assert isinstance(style, LabelStyle)
