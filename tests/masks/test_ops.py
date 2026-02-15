"""
Tests for imtools.masks.ops module.
"""
import numpy as np
import pytest

from imtools.masks.ops import (
    get_biggest_blob,
    bbox_from_mask,
    fill_holes_mask,
    extract_region_metadata,
)


class TestGetBiggestBlob:
    """Tests for get_biggest_blob function."""

    def test_single_blob(self):
        """Test extracting single blob."""
        mask = np.zeros((20, 20), dtype=np.int32)
        mask[5:10, 5:10] = 1

        result = get_biggest_blob(mask.astype(np.bool_))

        assert result.shape == (20, 20)
        assert result.dtype == np.bool_
        assert result[6, 6] == True
        assert result[15, 15] == False

    def test_multiple_blobs(self):
        """Test extracting largest from multiple blobs."""
        mask = np.zeros((30, 30), dtype=np.int32)
        # Small blob
        mask[2:4, 2:4] = 1
        # Larger blob
        mask[10:20, 10:20] = 1

        result = get_biggest_blob(mask.astype(np.bool_))

        # Should return only the larger blob
        assert result[15, 15] == True
        assert result[3, 3] == False

    def test_empty_mask(self):
        """Test empty mask returns zeros."""
        mask = np.zeros((10, 10), dtype=np.bool_)

        result = get_biggest_blob(mask)

        assert result.shape == (10, 10)
        assert result.max() == False

    def test_return_props(self):
        """Test returning properties."""
        mask = np.zeros((20, 20), dtype=np.int32)
        mask[5:10, 5:10] = 1

        result, props = get_biggest_blob(mask.astype(np.bool_), return_props=True)

        assert isinstance(props, dict)
        assert 'area' in props

    def test_invalid_input_not_array(self):
        """Test non-numpy input raises error."""
        with pytest.raises(AssertionError):
            get_biggest_blob("not an array")

    def test_invalid_input_non_binary(self):
        """Test non-binary input raises error."""
        mask = np.ones((10, 10), dtype=np.int32) * 5  # Not 0/1

        with pytest.raises(AssertionError):
            get_biggest_blob(mask)


class TestBboxFromMask:
    """Tests for bbox_from_mask function."""

    def test_simple_bbox(self):
        """Test simple bounding box extraction."""
        mask = np.zeros((20, 30), dtype=np.bool_)
        mask[5:15, 10:25] = True

        bbox = bbox_from_mask(mask)

        assert bbox == [10, 5, 25, 15]

    def test_full_mask(self):
        """Test bounding box for full mask."""
        mask = np.ones((10, 10), dtype=np.bool_)

        bbox = bbox_from_mask(mask)

        assert bbox == [0, 0, 10, 10]

    def test_empty_mask(self):
        """Test empty mask returns full image bbox."""
        mask = np.zeros((20, 20), dtype=np.bool_)

        bbox = bbox_from_mask(mask)

        # Returns full image bbox when mask is empty
        assert bbox == [0, 0, 20, 20]

    def test_invalid_not_2d(self):
        """Test non-2D input raises error."""
        mask = np.ones((10, 10, 3), dtype=np.bool_)

        with pytest.raises(AssertionError):
            bbox_from_mask(mask)

    def test_invalid_not_bool(self):
        """Test non-boolean input raises error."""
        mask = np.ones((10, 10), dtype=np.int32)

        with pytest.raises(AssertionError):
            bbox_from_mask(mask)


class TestFillHolesMask:
    """Tests for fill_holes_mask function."""

    def test_fill_holes_basic(self):
        """Test basic hole filling."""
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[5:15, 5:15] = 1
        # Create a hole in the middle
        mask[9:11, 9:11] = 0

        result = fill_holes_mask(mask.astype(np.bool_))

        # The hole should be filled
        assert result[10, 10] == True

    def test_no_holes(self):
        """Test mask with no holes."""
        mask = np.ones((15, 15), dtype=np.bool_)

        result = fill_holes_mask(mask)

        # Should remain all True
        assert result.max() == True

    def test_all_zeros(self):
        """Test empty mask."""
        mask = np.zeros((10, 10), dtype=np.bool_)

        result = fill_holes_mask(mask)

        # Should remain all False
        assert result.max() == False

    def test_without_closing(self):
        """Test filling without morphological closing."""
        mask = np.zeros((20, 20), dtype=np.bool_)
        mask[5:15, 5:15] = True

        result = fill_holes_mask(mask, apply_closing=False)

        assert result.shape == (20, 20)


class TestExtractRegionMetadata:
    """Tests for extract_region_metadata function."""

    def test_single_region(self):
        """Test extracting metadata for single region."""
        mask = np.zeros((20, 20), dtype=np.int32)
        mask[5:10, 5:10] = 1

        metadata = extract_region_metadata(mask.astype(np.bool_))

        assert len(metadata) == 1
        assert 'area' in metadata[0]

    def test_multiple_regions(self):
        """Test extracting metadata for multiple regions."""
        mask = np.zeros((30, 30), dtype=np.int32)
        mask[2:5, 2:5] = 1
        mask[10:15, 10:15] = 2
        mask[20:25, 20:25] = 3

        metadata = extract_region_metadata(mask.astype(np.bool_))

        assert len(metadata) == 3

    def test_empty_mask(self):
        """Test empty mask."""
        mask = np.zeros((10, 10), dtype=np.bool_)

        metadata = extract_region_metadata(mask)

        assert metadata == []

    def test_with_threshold(self):
        """Test with size threshold."""
        mask = np.zeros((30, 30), dtype=np.int32)
        mask[2:4, 2:4] = 1  # Small region (4 pixels)
        mask[10:20, 10:20] = 2  # Large region (100 pixels)

        metadata = extract_region_metadata(mask, item_size_threshold=10)

        # Should have metadata for both regions
        assert len(metadata) == 2
