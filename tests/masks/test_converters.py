"""
Tests for imtools.masks.converters module.
"""
import numpy as np
import pytest
import torch

from imtools.masks.converters import (
    masks_to_label_image_torch_vectorized,
    masks_to_label_image_torch_loop,
    masks_to_label_image_numpy_loop,
    masks_to_label_image_numpy_vectorized,
    binary_mask_to_label_image,
    masks_to_label_image,
)


class TestMasksToLabelImageTorchVectorized:
    """Tests for torch vectorized converter."""

    def test_basic_conversion(self):
        """Test basic mask stack to label image."""
        masks = np.zeros((3, 20, 20), dtype=np.bool_)
        masks[0, 2:5, 2:5] = True
        masks[1, 10:15, 10:15] = True
        masks[2, 15:18, 15:18] = True

        result = masks_to_label_image_torch_vectorized(masks)

        assert result.shape == (20, 20)
        assert result.dtype == np.int32
        # Check that labels are assigned correctly
        assert result[3, 3] == 1
        assert result[12, 12] == 2

    def test_with_torch_tensor(self):
        """Test with torch tensor input."""
        masks = torch.zeros((3, 10, 10), dtype=torch.bool)
        masks[0, 1:3, 1:3] = True

        result = masks_to_label_image_torch_vectorized(masks)

        assert result.shape == (10, 10)

    def test_empty_masks(self):
        """Test with all-zero masks."""
        masks = np.zeros((3, 10, 10), dtype=np.bool_)

        result = masks_to_label_image_torch_vectorized(masks)

        assert result.shape == (10, 10)
        assert result.max() == 0


class TestMasksToLabelImageTorchLoop:
    """Tests for torch loop converter."""

    def test_basic_conversion(self):
        """Test basic mask stack to label image."""
        masks = np.zeros((2, 15, 15), dtype=np.bool_)
        masks[0, 2:5, 2:5] = True
        masks[1, 8:12, 8:12] = True

        result = masks_to_label_image_torch_loop(masks)

        assert result.shape == (15, 15)
        assert result[3, 3] == 1
        assert result[10, 10] == 2


class TestMasksToLabelImageNumpyLoop:
    """Tests for numpy loop converter."""

    def test_basic_conversion(self):
        """Test basic mask stack to label image."""
        masks = np.zeros((3, 20, 20), dtype=np.bool_)
        masks[0, 2:5, 2:5] = True
        masks[1, 10:15, 10:15] = True

        result = masks_to_label_image_numpy_loop(masks)

        assert result.shape == (20, 20)
        assert result.dtype == np.int32

    def test_with_torch_tensor(self):
        """Test torch tensor is converted to numpy."""
        masks = torch.zeros((2, 10, 10), dtype=torch.bool)
        masks[0, 1:3, 1:3] = True

        result = masks_to_label_image_numpy_loop(masks)

        assert isinstance(result, np.ndarray)


class TestMasksToLabelImageNumpyVectorized:
    """Tests for numpy vectorized converter."""

    def test_basic_conversion(self):
        """Test basic mask stack to label image."""
        masks = np.zeros((3, 20, 20), dtype=np.bool_)
        masks[0, 2:5, 2:5] = True
        masks[1, 10:15, 10:15] = True

        result = masks_to_label_image_numpy_vectorized(masks)

        assert result.shape == (20, 20)
        assert result.dtype == np.int32


class TestBinaryMaskToLabelImage:
    """Tests for binary mask to label image converter."""

    def test_single_blob(self):
        """Test single connected component."""
        mask = np.zeros((20, 20), dtype=np.bool_)
        mask[5:10, 5:10] = True

        result = binary_mask_to_label_image(mask)

        assert result.shape == (20, 20)
        assert result.dtype == np.int32
        # Background is 0, blob should be 1
        assert result.max() == 1

    def test_multiple_blobs(self):
        """Test multiple connected components."""
        mask = np.zeros((20, 20), dtype=np.bool_)
        mask[2:4, 2:4] = True   # Blob 1
        mask[10:12, 10:12] = True  # Blob 2

        result = binary_mask_to_label_image(mask)

        # Should have background + 2 blobs = max label 2
        assert result.max() == 2

    def test_connectivity_4(self):
        """Test 4-connectivity."""
        mask = np.zeros((10, 10), dtype=np.bool_)
        # Diagonal pixels - not connected with 4-connectivity
        mask[2, 2] = True
        mask[3, 3] = True

        result = binary_mask_to_label_image(mask, connectivity=4)

        # Should be separate blobs
        assert result.max() == 2

    def test_connectivity_8(self):
        """Test 8-connectivity."""
        mask = np.zeros((10, 10), dtype=np.bool_)
        # Diagonal pixels - connected with 8-connectivity
        mask[2, 2] = True
        mask[3, 3] = True

        result = binary_mask_to_label_image(mask, connectivity=8)

        # Should be single blob
        assert result.max() == 1


class TestMasksToLabelImageDispatcher:
    """Tests for the main dispatcher function."""

    @pytest.mark.parametrize("use_loop", [True, False])
    @pytest.mark.parametrize("use_numpy", [True, False])
    def test_dispatcher_options(self, use_loop, use_numpy):
        """Test dispatcher with various options."""
        masks = np.zeros((3, 15, 15), dtype=np.bool_)
        masks[0, 2:5, 2:5] = True
        masks[1, 8:12, 8:12] = True

        result = masks_to_label_image(masks, use_loop=use_loop, use_numpy=use_numpy)

        assert result.shape == (15, 15)
        assert result.dtype == np.int32

    def test_dispatcher_numpy_auto(self):
        """Test dispatcher with use_numpy=None (auto)."""
        masks = np.zeros((2, 10, 10), dtype=np.bool_)
        masks[0, 1:3, 1:3] = True

        result = masks_to_label_image(masks, use_numpy=None)

        assert result.shape == (10, 10)

    def test_empty_mask_stack(self):
        """Test with empty mask stack."""
        masks = np.zeros((0, 10, 10), dtype=np.bool_)

        result = masks_to_label_image(masks)

        assert result.shape == (10, 10)
