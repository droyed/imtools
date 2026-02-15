"""
Tests for imtools.core.formats module.
"""
import numpy as np
import pytest
from pathlib import Path
from PIL import Image
import tempfile

from imtools.core.formats import (
    pil_to_opencv,
    opencv_to_pil,
    to_pil_image,
    to_numpy_image,
    imwrite,
)


class TestPilToOpencv:
    """Tests for pil_to_opencv function."""

    def test_pil_rgb_to_opencv(self):
        """Test converting PIL RGB to OpenCV BGR."""
        pil_img = Image.new('RGB', (10, 10), color=(255, 0, 0))
        cv_img = pil_to_opencv(pil_img)

        assert isinstance(cv_img, np.ndarray)
        assert cv_img.shape == (10, 10, 3)
        assert cv_img.dtype == np.uint8

    def test_pil_rgba_to_opencv(self):
        """Test converting PIL RGBA to OpenCV BGRA."""
        pil_img = Image.new('RGBA', (10, 10), color=(255, 0, 0, 128))
        cv_img = pil_to_opencv(pil_img)

        assert cv_img.shape == (10, 10, 4)
        assert cv_img.dtype == np.uint8

    def test_pil_grayscale_to_opencv(self):
        """Test converting PIL L to OpenCV grayscale."""
        pil_img = Image.new('L', (10, 10), color=128)
        cv_img = pil_to_opencv(pil_img)

        assert cv_img.shape == (10, 10)
        assert cv_img.dtype == np.uint8

    def test_invalid_input_raises(self):
        """Test that non-PIL input raises TypeError."""
        with pytest.raises(TypeError):
            pil_to_opencv("not an image")

    def test_ensure_contiguous_true(self):
        """Test ensure_contiguous parameter."""
        pil_img = Image.new('RGB', (10, 10))
        cv_img = pil_to_opencv(pil_img, ensure_contiguous=True)
        assert cv_img.flags['C_CONTIGUOUS']


class TestOpencvToPil:
    """Tests for opencv_to_pil function."""

    def test_opencv_bgr_to_pil(self):
        """Test converting OpenCV BGR to PIL RGB."""
        cv_img = np.zeros((10, 10, 3), dtype=np.uint8)
        cv_img[:, :, 2] = 255  # Red in BGR
        pil_img = opencv_to_pil(cv_img)

        assert pil_img.mode == 'RGB'

    def test_opencv_grayscale_to_pil(self):
        """Test converting OpenCV grayscale to PIL."""
        cv_img = np.zeros((10, 10), dtype=np.uint8)
        cv_img[:] = 128
        pil_img = opencv_to_pil(cv_img)

        assert pil_img.mode == 'L'

    def test_opencv_rgba_to_pil(self):
        """Test converting OpenCV RGBA to PIL RGBA."""
        cv_img = np.zeros((10, 10, 4), dtype=np.uint8)
        cv_img[:, :, 0] = 255  # R
        cv_img[:, :, 3] = 255  # A
        pil_img = opencv_to_pil(cv_img, channel_order='RGB')

        assert pil_img.mode == 'RGBA'

    def test_invalid_input_raises(self):
        """Test that non-numpy input raises TypeError."""
        with pytest.raises(TypeError):
            opencv_to_pil("not an array")

    def test_invalid_channel_order_raises(self):
        """Test invalid channel_order raises ValueError."""
        cv_img = np.zeros((10, 10, 3), dtype=np.uint8)
        with pytest.raises(ValueError):
            opencv_to_pil(cv_img, channel_order='INVALID')

    def test_bool_array_to_pil(self):
        """Test boolean array converts to mode '1'."""
        cv_img = np.zeros((10, 10), dtype=np.bool_)
        cv_img[2:5, 2:5] = True
        pil_img = opencv_to_pil(cv_img)

        assert pil_img.mode == '1'


class TestToPilImage:
    """Tests for to_pil_image function."""

    def test_from_pil_image(self):
        """Test passing PIL image through."""
        pil_img = Image.new('RGB', (10, 10))
        result = to_pil_image(pil_img)
        assert result is pil_img

    def test_from_numpy_array(self):
        """Test converting numpy array to PIL."""
        arr = np.zeros((10, 10, 3), dtype=np.uint8)
        result = to_pil_image(arr)
        assert isinstance(result, Image.Image)

    def test_from_existing_path(self):
        """Test loading from file path."""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            img = Image.new('RGB', (10, 10), color=(100, 150, 200))
            img.save(f.name)
            result = to_pil_image(f.name)
            assert isinstance(result, Image.Image)
            Path(f.name).unlink()

    def test_from_nonexistent_path_raises(self):
        """Test nonexistent path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            to_pil_image("/nonexistent/path.png")

    def test_invalid_type_raises(self):
        """Test invalid type raises TypeError."""
        with pytest.raises(TypeError):
            to_pil_image(12345)


class TestToNumpyImage:
    """Tests for to_numpy_image function."""

    def test_from_numpy_array(self):
        """Test numpy array passes through."""
        arr = np.zeros((10, 10, 3), dtype=np.uint8)
        result = to_numpy_image(arr)
        assert result is arr

    def test_from_file_path(self, tmp_path):
        """Test loading from file path."""
        img_path = tmp_path / "test.png"
        img = Image.new('RGB', (10, 10), color=(50, 100, 150))
        img.save(img_path)

        result = to_numpy_image(str(img_path))
        assert isinstance(result, np.ndarray)
        assert result.shape == (10, 10, 3)

    def test_from_nonexistent_path_raises(self):
        """Test nonexistent path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            to_numpy_image("/nonexistent/path.png")

    def test_normalize_dims_2d(self):
        """Test normalize_dims expands 2D to 3D."""
        arr = np.zeros((10, 10), dtype=np.uint8)
        result = to_numpy_image(arr, normalize_dims=True)
        assert result.shape == (10, 10, 1)

    def test_force_3d_grayscale(self):
        """Test force_3d converts grayscale to RGB."""
        arr = np.zeros((10, 10), dtype=np.uint8)
        result = to_numpy_image(arr, force_3d=True)
        assert result.shape == (10, 10, 3)

    def test_drop_alpha(self):
        """Test drop_alpha removes alpha channel."""
        arr = np.zeros((10, 10, 4), dtype=np.uint8)
        result = to_numpy_image(arr, drop_alpha=True)
        assert result.shape == (10, 10, 3)

    def test_force_uint8_from_float(self):
        """Test force_uint8 converts float to uint8."""
        arr = np.zeros((10, 10), dtype=np.float32)
        arr[:] = 0.5
        result = to_numpy_image(arr, force_uint8=True)
        assert result.dtype == np.uint8
        assert result.max() == 127 or result.max() == 128

    def test_force_copy(self):
        """Test force_copy creates a copy."""
        arr = np.zeros((10, 10, 3), dtype=np.uint8)
        result = to_numpy_image(arr, force_copy=True)
        assert result is not arr


class TestImwrite:
    """Tests for imwrite function."""

    def test_write_rgb_image(self, tmp_path):
        """Test writing RGB image to PNG."""
        arr = np.zeros((10, 10, 3), dtype=np.uint8)
        arr[:, :, 0] = 255  # Red
        output_path = tmp_path / "output.png"

        imwrite(arr, str(output_path))

        assert output_path.exists()

    def test_write_grayscale_image(self, tmp_path):
        """Test writing grayscale image."""
        arr = np.zeros((10, 10), dtype=np.uint8)
        arr[:] = 128
        output_path = tmp_path / "gray.png"

        imwrite(arr, str(output_path))

        assert output_path.exists()

    def test_write_bool_array(self, tmp_path):
        """Test writing boolean array converts to uint8."""
        arr = np.zeros((10, 10), dtype=np.bool_)
        arr[2:5, 2:5] = True
        output_path = tmp_path / "bool.png"

        imwrite(arr, str(output_path))

        assert output_path.exists()

    def test_write_float_array(self, tmp_path):
        """Test writing float array (0.0-1.0)."""
        arr = np.zeros((10, 10), dtype=np.float32)
        arr[:] = 0.5
        output_path = tmp_path / "float.png"

        imwrite(arr, str(output_path))

        assert output_path.exists()

    def test_float_out_of_range_raises(self, tmp_path):
        """Test float array out of range raises ValueError."""
        arr = np.zeros((10, 10), dtype=np.float32)
        arr[:] = 1.5  # Out of valid range
        output_path = tmp_path / "invalid.png"

        with pytest.raises(ValueError):
            imwrite(arr, str(output_path))
