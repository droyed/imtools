"""
Tests for imtools.viz.display module.

Note: Display functions that require GUI are tested for their
computational logic without actual rendering.
"""
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from imtools.viz.display import (
    get_screen_resolution,
    scale_image_to_fit,
    show_cv2,
)


class TestGetScreenResolution:
    """Tests for get_screen_resolution function."""

    def test_returns_tuple(self):
        """Test returns a tuple."""
        resolution = get_screen_resolution()

        assert isinstance(resolution, tuple)
        assert len(resolution) == 2

    def test_resolution_values(self):
        """Test resolution values are positive integers."""
        width, height = get_screen_resolution()

        assert width > 0
        assert height > 0
        assert isinstance(width, int)
        assert isinstance(height, int)


class TestScaleImageToFit:
    """Tests for scale_image_to_fit function."""

    def test_scale_with_aspect(self):
        """Test scaling while maintaining aspect ratio."""
        image = np.zeros((100, 200, 3), dtype=np.uint8)

        result, x_off, y_off = scale_image_to_fit(image, 400, 200, maintain_aspect=True)

        assert result.shape[2] == 3
        assert x_off >= 0
        assert y_off >= 0

    def test_scale_without_aspect(self):
        """Test scaling without maintaining aspect ratio."""
        image = np.zeros((100, 200, 3), dtype=np.uint8)

        result, x_off, y_off = scale_image_to_fit(image, 400, 200, maintain_aspect=False)

        assert result.shape == (200, 400, 3)

    def test_larger_image(self):
        """Test scaling down a larger image."""
        image = np.zeros((1000, 1000, 3), dtype=np.uint8)

        result, x_off, y_off = scale_image_to_fit(image, 500, 500, maintain_aspect=True)

        assert result.shape[0] <= 500
        assert result.shape[1] <= 500


class TestShowCv2:
    """Tests for show_cv2 function."""

    def test_invalid_dtype_raises(self):
        """Test that non-uint8 images raise ValueError."""
        image = np.zeros((100, 100, 3), dtype=np.float32)

        with pytest.raises(ValueError):
            show_cv2(image, block=False)

    def test_valid_uint8_image(self):
        """Test valid uint8 image doesn't crash."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        # Should not raise when block=False
        show_cv2(image, block=False)

    def test_grayscale_uint8(self):
        """Test grayscale uint8 image."""
        image = np.zeros((100, 100), dtype=np.uint8)

        show_cv2(image, block=False)

    def test_close_key_types(self):
        """Test different close key types."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        # Should not raise for various key types
        show_cv2(image, close_key='q', block=False)
        show_cv2(image, close_key='esc', block=False)
        show_cv2(image, close_key=27, block=False)


class TestBlockUntilClosed:
    """Tests for block_until_closed - mocked since it requires window."""

    @patch('cv2.waitKey')
    @patch('cv2.destroyWindow')
    def test_wait_for_key(self, mock_destroy, mock_wait):
        """Test waiting for key press."""
        from imtools.viz.display import block_until_closed

        mock_wait.return_value = ord('q')

        # Should not hang
        block_until_closed("test_window", 'q')

        mock_destroy.assert_called_once()
