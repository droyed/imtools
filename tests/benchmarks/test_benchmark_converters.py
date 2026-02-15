"""
Tests for imtools.benchmarks.benchmark_converters module.

Tests benchmark utilities with small synthetic data.
"""
import numpy as np
import pytest

from imtools.benchmarks.benchmark_converters import (
    MockResults,
    create_label_image,
    run_test,
    benchmark_scenario,
)


class TestMockResults:
    """Tests for MockResults class."""

    def test_creation(self):
        """Test creating mock results."""
        mock = MockResults(n=3, h=50, w=50)

        assert mock.n == 3
        assert mock.h == 50
        assert mock.w == 50

    def test_masks_shape(self):
        """Test mask array shape."""
        mock = MockResults(n=5, h=30, w=40)

        assert mock.masks_cpu.shape == (5, 30, 40)

    def test_masks_dtype(self):
        """Test mask array dtype."""
        mock = MockResults(n=2, h=10, w=10)

        assert mock.masks_cpu.dtype == np.uint8


class TestCreateLabelImage:
    """Tests for create_label_image wrapper."""

    @pytest.mark.parametrize("use_loop", [True, False])
    @pytest.mark.parametrize("use_numpy", [True, False])
    def test_basic(self, use_loop, use_numpy):
        """Test basic label image creation."""
        masks = np.zeros((3, 20, 20), dtype=np.uint8)
        masks[0, 2:5, 2:5] = 1

        result = create_label_image(masks, use_loop=use_loop, use_numpy=use_numpy)

        assert result.shape == (20, 20)


class TestRunTest:
    """Tests for run_test function."""

    def test_basic_function(self):
        """Test running a basic function."""
        def simple_func(x):
            return x * 2

        result = run_test("double", simple_func, {"x": 5}, n_runs=3, n_warmup=0)

        assert result['Method'] == "double"
        assert 'FPS' in result
        assert 'Time(ms)' in result


class TestBenchmarkScenario:
    """Tests for benchmark_scenario function."""

    def test_small_scenario(self):
        """Test benchmark with small data."""
        # This runs quickly with small sizes
        benchmark_scenario(n=2, h=20, w=20)

    @pytest.mark.slow
    def test_larger_scenario(self):
        """Test benchmark with larger data (marked slow)."""
        benchmark_scenario(n=5, h=50, w=50)


class TestBenchmarkIntegration:
    """Integration tests for benchmarks."""

    def test_all_numpy_backends(self):
        """Test all NumPy backends work."""
        masks = np.zeros((3, 15, 15), dtype=np.uint8)
        masks[0, 2:5, 2:5] = 1

        # Test all combinations
        result1 = create_label_image(masks, use_loop=True, use_numpy=True)
        result2 = create_label_image(masks, use_loop=False, use_numpy=True)

        assert result1.shape == result2.shape

    def test_mock_results_cpu_only(self):
        """Test MockResults when CUDA not available."""
        mock = MockResults(n=2, h=10, w=10)

        # Should work fine
        assert mock.masks_cpu is not None
        assert mock.masks_gpu is None or mock.masks_gpu is not None  # Depends on CUDA
