"""
Tests for imtools.annotations.parsers module.
"""
import numpy as np
import pytest
from unittest.mock import Mock, MagicMock

from imtools.annotations.parsers import (
    yolo_to_annotations,
    label_image_to_annotations,
)
from imtools.annotations.labels import LabelFormat


class TestYoloToAnnotations:
    """Tests for yolo_to_annotations function."""

    def test_yolo_to_annotations_empty_masks(self):
        """Test with None masks returns empty list."""
        result = Mock()
        result.masks = None

        annotations = yolo_to_annotations(result)

        assert annotations == []


class TestLabelImageToAnnotations:
    """Tests for label_image_to_annotations function."""

    def test_label_image_basic(self):
        """Test basic label image conversion."""
        # Create a simple label image with two regions
        label_img = np.zeros((50, 50), dtype=np.int32)
        label_img[5:15, 5:15] = 1
        label_img[25:35, 25:35] = 2

        annotations = label_image_to_annotations(label_img, class_name='object')

        assert len(annotations) == 2
        assert all(a['score'] == 1.0 for a in annotations)  # GT has conf=1.0

    def test_label_image_empty(self):
        """Test empty label image returns empty list."""
        label_img = np.zeros((50, 50), dtype=np.int32)

        annotations = label_image_to_annotations(label_img)

        assert annotations == []

    def test_label_image_single_region(self):
        """Test label image with single region."""
        label_img = np.zeros((50, 50), dtype=np.int32)
        label_img[10:20, 10:20] = 1

        annotations = label_image_to_annotations(label_img, class_name='ball')

        assert len(annotations) == 1
        assert annotations[0]['score'] == 1.0

    def test_label_image_with_format(self):
        """Test label image with custom format."""
        label_img = np.zeros((50, 50), dtype=np.int32)
        label_img[10:20, 10:20] = 1

        annotations = label_image_to_annotations(
            label_img,
            class_name='car',
            label_format=LabelFormat.PIPE_BASIC
        )

        assert len(annotations) == 1
        assert 'car' in annotations[0]['text']
        assert '|' in annotations[0]['text']
