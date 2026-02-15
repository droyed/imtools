"""
Tests for imtools.viz.compose module.
"""
import numpy as np
import pytest
from PIL import Image

from imtools.viz.compose import (
    text_on_canvas,
    stack_images,
    add_title,
)
from imtools.core.types import TitleConfig


class TestTextOnCanvas:
    """Tests for text_on_canvas function."""

    def test_basic_text(self):
        """Test basic text rendering."""
        result = text_on_canvas("Hello")

        assert isinstance(result, Image.Image)
        assert result.mode == 'RGBA'

    def test_multiline_text(self):
        """Test multiline text."""
        result = text_on_canvas("Line1\nLine2")

        assert result is not None

    def test_text_list_input(self):
        """Test text as list input."""
        result = text_on_canvas(["First", "Second"])

        assert isinstance(result, Image.Image)

    def test_custom_font_size(self):
        """Test custom font size."""
        result = text_on_canvas("Test", font_size=60)

        assert result is not None

    def test_custom_colors(self):
        """Test custom text and background colors."""
        result = text_on_canvas(
            "Test",
            bg_color=(255, 0, 0, 255),
            text_color=(0, 255, 0)
        )

        assert result is not None

    def test_alignment_options(self):
        """Test different alignment options."""
        for align in ["left", "center", "right"]:
            result = text_on_canvas("Test", align=align)
            assert result is not None

    def test_invalid_alignment_raises(self):
        """Test invalid alignment raises error."""
        with pytest.raises(AssertionError):
            text_on_canvas("Test", align="invalid")

    def test_output_rgb(self):
        """Test RGB output."""
        result = text_on_canvas("Test", output_rgb=True)

        assert result.mode == 'RGB'

    def test_crop_to_text(self):
        """Test cropping to text bounds."""
        result = text_on_canvas("Test", crop_to_text=True)

        assert result is not None


class TestStackImages:
    """Tests for stack_images function."""

    def test_vertical_stack(self):
        """Test vertical stacking."""
        img1 = Image.new('RGB', (100, 50), color='red')
        img2 = Image.new('RGB', (100, 50), color='blue')

        result = stack_images([img1, img2], direction='vertical')

        assert result.height == 100
        assert result.width == 100

    def test_horizontal_stack(self):
        """Test horizontal stacking."""
        img1 = Image.new('RGB', (50, 100), color='red')
        img2 = Image.new('RGB', (50, 100), color='blue')

        result = stack_images([img1, img2], direction='horizontal')

        assert result.width == 100
        assert result.height == 100

    def test_align_start(self):
        """Test start alignment."""
        img1 = Image.new('RGB', (50, 50), color='red')
        img2 = Image.new('RGB', (100, 100), color='blue')

        result = stack_images([img1, img2], align='start')

        assert result is not None

    def test_align_center(self):
        """Test center alignment."""
        img1 = Image.new('RGB', (50, 50), color='red')
        img2 = Image.new('RGB', (100, 100), color='blue')

        result = stack_images([img1, img2], align='center')

        assert result is not None

    def test_align_end(self):
        """Test end alignment."""
        img1 = Image.new('RGB', (50, 50), color='red')
        img2 = Image.new('RGB', (100, 100), color='blue')

        result = stack_images([img1, img2], align='end')

        assert result is not None

    def test_resize_to_max(self):
        """Test resize to max dimension."""
        img1 = Image.new('RGB', (50, 50), color='red')
        img2 = Image.new('RGB', (100, 100), color='blue')

        result = stack_images([img1, img2], align='resize_to_max')

        assert result is not None

    def test_resize_to_min(self):
        """Test resize to min dimension."""
        img1 = Image.new('RGB', (50, 50), color='red')
        img2 = Image.new('RGB', (100, 100), color='blue')

        result = stack_images([img1, img2], align='resize_to_min')

        assert result is not None

    def test_invalid_direction_raises(self):
        """Test invalid direction raises error."""
        img1 = Image.new('RGB', (50, 50))
        img2 = Image.new('RGB', (50, 50))

        with pytest.raises(ValueError):
            stack_images([img1, img2], direction='invalid')

    def test_invalid_align_raises(self):
        """Test invalid alignment raises error."""
        img1 = Image.new('RGB', (50, 50))
        img2 = Image.new('RGB', (50, 50))

        with pytest.raises(ValueError):
            stack_images([img1, img2], align='invalid')

    def test_invalid_mode_raises(self):
        """Test invalid mode raises error."""
        img1 = Image.new('RGB', (50, 50))
        img2 = Image.new('RGB', (50, 50))

        with pytest.raises(ValueError):
            stack_images([img1, img2], mode='INVALID')

    def test_too_few_images_raises(self):
        """Test with fewer than 2 images raises error."""
        with pytest.raises(ValueError):
            stack_images([Image.new('RGB', (50, 50))])


class TestAddTitle:
    """Tests for add_title function."""

    def test_add_title_basic(self):
        """Test adding title to image."""
        img = Image.new('RGB', (100, 100), color='white')
        title = "Test Image"

        result = add_title(img, title)

        assert result.height > img.height
        assert result.width >= img.width  # May be larger due to padding

    def test_add_title_with_config(self):
        """Test adding title with custom config."""
        img = Image.new('RGB', (100, 100), color='white')
        config = TitleConfig(font_size=24)
        title = "Custom"

        result = add_title(img, title, config=config)

        assert result.height > img.height

    def test_title_uses_config(self):
        """Test title respects config settings."""
        img = Image.new('RGB', (200, 100), color='white')
        config = TitleConfig(padding=30)
        title = "Padded"

        result = add_title(img, title, config=config)

        assert result is not None
