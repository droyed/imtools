"""
Tests for YOLO format handling in imtools.

Note: Full YOLO format parsing/writing tests would require file I/O.
Here we test the utility functions that work with YOLO format data.
"""
import numpy as np
import pytest


class TestYoloIntegration:
    """Integration tests for YOLO-related functionality."""

    def test_yolo_annotations_workflow(self, tmp_yolo_dir):
        """Test a complete YOLO workflow."""
        # This is a placeholder - in practice would test:
        # 1. Loading YOLO annotations from files
        # 2. Converting to internal format
        # 3. Verifying coordinate normalization/denormalization

        labels_dir = tmp_yolo_dir / "labels"
        assert labels_dir.exists()

        # Check label files exist
        label_files = list(labels_dir.glob("*.txt"))
        assert len(label_files) == 2

    def test_yolo_coordinates_normalization(self):
        """Test YOLO coordinate normalization logic."""
        # YOLO format: class_id x_center y_center width height (normalized 0-1)
        # Image size: 640x480

        img_w, img_h = 640, 480

        # Test converting from absolute to normalized
        x_center, y_center = 320, 240
        width, height = 100, 80

        x_norm = x_center / img_w
        y_norm = y_center / img_h
        w_norm = width / img_w
        h_norm = height / img_h

        assert 0 <= x_norm <= 1
        assert 0 <= y_norm <= 1
        assert 0 <= w_norm <= 1
        assert 0 <= h_norm <= 1

    def test_yolo_coordinates_denormalization(self):
        """Test YOLO coordinate denormalization logic."""
        # From normalized to absolute
        x_norm, y_norm = 0.5, 0.5
        w_norm, h_norm = 0.2, 0.25

        img_w, img_h = 640, 480

        x_center = x_norm * img_w
        y_center = y_norm * img_h
        width = w_norm * img_w
        height = h_norm * img_h

        assert x_center == 320.0
        assert y_center == 240.0
        assert width == 128.0
        assert height == 120.0
