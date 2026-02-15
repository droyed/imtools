"""
Tests for imtools.viz.overlay_viewer module.

Tests the OverlayViewer class computational logic without GUI.
"""
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from imtools.viz.overlay_viewer import (
    OverlayViewer,
    calculate_fit_dimensions,
    METHOD_PARAMS,
)


class TestCalculateFitDimensions:
    """Tests for calculate_fit_dimensions function."""

    def test_exact_fit(self):
        """Test when image matches container."""
        # Use padding=0 to get exact dimensions
        w, h, x, y = calculate_fit_dimensions(100, 100, 100, 100, padding=0)

        assert w == 100
        assert h == 100
        assert x == 0
        assert y == 0

    def test_wider_container(self):
        """Test image narrower than container."""
        w, h, x, y = calculate_fit_dimensions(50, 100, 100, 100, padding=0)

        # Image fits with aspect ratio preserved, so width is 50, height is 100
        assert w == 50
        assert h == 100
        assert x > 0  # centered

    def test_taller_container(self):
        """Test image taller than container."""
        w, h, x, y = calculate_fit_dimensions(100, 50, 100, 100, padding=0)

        # Image fits with aspect ratio preserved
        assert w == 100
        assert h == 50
        assert y > 0  # centered


class TestOverlayViewerValidation:
    """Tests for OverlayViewer input validation."""

    def test_validate_inputs_valid(self):
        """Test valid inputs pass validation."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        label = np.zeros((100, 100), dtype=np.int32)

        # Should not raise
        OverlayViewer._validate_inputs(img, label, 'preset')

    def test_validate_not_numpy(self):
        """Test non-numpy image raises error."""
        with pytest.raises(ValueError):
            OverlayViewer._validate_inputs("not array", np.zeros((10, 10)), 'preset')

    def test_validate_empty_image(self):
        """Test empty image raises error."""
        img = np.array([])
        with pytest.raises(ValueError):
            OverlayViewer._validate_inputs(img, np.zeros((10, 10)), 'preset')

    def test_validate_wrong_dims(self):
        """Test wrong dimensions raise error."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        label = np.zeros((50, 50), dtype=np.int32)

        with pytest.raises(ValueError):
            OverlayViewer._validate_inputs(img, label, 'preset')

    def test_validate_label_not_2d(self):
        """Test non-2D label raises error."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        label = np.zeros((100, 100, 3), dtype=np.int32)

        with pytest.raises(ValueError):
            OverlayViewer._validate_inputs(img, label, 'preset')

    def test_validate_invalid_method(self):
        """Test invalid method raises error."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        label = np.zeros((100, 100), dtype=np.int32)

        with pytest.raises(ValueError):
            OverlayViewer._validate_inputs(img, label, 'invalid_method')


class TestOverlayViewerRemap:
    """Tests for label remapping functionality."""

    def test_remap_sparse_labels(self):
        """Test remapping sparse labels."""
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        label = np.zeros((50, 50), dtype=np.int32)
        # Sparse labels: 0, 10, 20 instead of 0, 1, 2
        label[5:10, 5:10] = 10
        label[20:25, 20:25] = 20

        viewer = OverlayViewer(img, label, 'preset')

        # Should have remapped to contiguous
        assert viewer.num_labels == 3

    def test_remap_dense_labels(self):
        """Test dense labels stay dense."""
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        label = np.zeros((50, 50), dtype=np.int32)
        label[5:10, 5:10] = 1
        label[20:25, 20:25] = 2

        viewer = OverlayViewer(img, label, 'preset')

        assert viewer.num_labels == 3


class TestOverlayViewerBlend:
    """Tests for blend computation."""

    def test_blend_basic(self):
        """Test basic blending."""
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        img[:] = [128, 128, 128]
        label = np.zeros((50, 50), dtype=np.int32)
        label[10:20, 10:20] = 1

        viewer = OverlayViewer(img, label, 'preset')
        result = viewer.blend(0.5)

        assert result is not None
        assert len(result) == 50 * 50 * 4

    def test_blend_no_labels(self):
        """Test blending with no labels."""
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        label = np.zeros((50, 50), dtype=np.int32)  # All zeros = background only

        viewer = OverlayViewer(img, label, 'preset')
        result = viewer.blend(0.5)

        assert result is not None


class TestMethodParams:
    """Tests for METHOD_PARAMS configuration."""

    def test_all_methods_defined(self):
        """Test all expected methods are defined."""
        expected = ['preset', 'hsv', 'golden_ratio', 'kmeans', 'colormap', 'colormap_extended']

        for method in expected:
            assert method in METHOD_PARAMS

    def test_preset_no_params(self):
        """Test preset method has no extra params."""
        assert METHOD_PARAMS['preset'] is None

    def test_hsv_has_saturation(self):
        """Test HSV method has saturation parameter."""
        assert METHOD_PARAMS['hsv']['label'] == 'saturation'

    def test_colormap_has_third_dropdown(self):
        """Test colormap has third dropdown."""
        assert METHOD_PARAMS['colormap']['has_third_dropdown'] is True
        assert METHOD_PARAMS['colormap']['third_label'] == 'colormap'


class TestOverlayViewerGetResult:
    """Tests for result retrieval."""

    def test_get_result(self):
        """Test getting blended result."""
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        label = np.zeros((50, 50), dtype=np.int32)

        viewer = OverlayViewer(img, label, 'preset')

        # The result requires DPG context, so test what we can
        settings = viewer.get_result()[1]

        assert 'alpha' in settings
        assert 'method' in settings
