"""Performance benchmarks for mask-to-label-image converters.

Measures throughput (FPS) and latency of all NumPy and PyTorch backend
variants across configurable scenarios.

Usage::

    # Run from the project root (package must be installed or on PYTHONPATH)
    python -m imtools.benchmarks
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch
import pandas as pd

# Update imports to match the new package structure
from imtools.masks.converters import (
    masks_to_label_image_torch_vectorized,
    masks_to_label_image_torch_loop,
    masks_to_label_image_numpy_loop,
    masks_to_label_image_numpy_vectorized
)


def create_label_image(
    masks_array: np.ndarray | torch.Tensor,
    use_loop: bool = True,
    use_numpy: bool = True,
    device: str | None = None,
) -> np.ndarray:
    """Route a benchmark call to the appropriate converter backend.

    Args:
        masks_array: Boolean mask stack of shape ``(N, H, W)``.
        use_loop: If ``True`` (default), use the loop implementation.
        use_numpy: If ``True`` (default), use NumPy; otherwise PyTorch.
        device: Target device for PyTorch backend (``None`` | ``'cpu'`` |
            ``'cuda'``).

    Returns:
        2-D ``int32`` NumPy array label image of shape ``(H, W)``.
    """
    if use_numpy:
        if use_loop:
            return masks_to_label_image_numpy_loop(masks_array)
        else:
            return masks_to_label_image_numpy_vectorized(masks_array)
    else:
        if use_loop:
            return masks_to_label_image_torch_loop(masks_array, device=device)
        else:
            return masks_to_label_image_torch_vectorized(masks_array, device=device)

# ==========================================
# 2. Mock Data Generator
# ==========================================

class MockResults:
    """Generate synthetic mask data for benchmarking.

    Attributes:
        n: Number of masks.
        h: Image height in pixels.
        w: Image width in pixels.
        masks_cpu: ``uint8`` NumPy mask array of shape ``(n, h, w)`` with
            ~20 % foreground probability per pixel.
        masks_gpu: Same data as a CUDA :class:`torch.Tensor`, or ``None``
            if CUDA is unavailable.
    """

    def __init__(self, n: int, h: int, w: int, device: str = 'cpu') -> None:
        self.n = n
        self.h = h
        self.w = w
        # Generate random boolean masks (simulated as 0/1 uint8)
        # 20% probability of a pixel being part of a mask
        self.masks_cpu = (np.random.rand(n, h, w) > 0.8).astype(np.uint8)
        
        self.masks_gpu: Optional[torch.Tensor] = None
        if torch.cuda.is_available():
            self.masks_gpu = torch.from_numpy(self.masks_cpu).cuda()

# ==========================================
# 3. Benchmarking Logic
# ==========================================

def run_test(
    name: str,
    func: Callable[..., Any],
    kwargs: Dict[str, Any],
    n_runs: int = 20,
    n_warmup: int = 5,
) -> Dict[str, Any]:
    """Run a single timed benchmark case.

    Args:
        name: Human-readable test name used as the ``'Method'`` key in the
            result dictionary.
        func: Callable to benchmark.
        kwargs: Keyword arguments passed to ``func`` on each call.
        n_runs: Number of timed iterations.
        n_warmup: Number of warm-up iterations (excluded from timing).

    Returns:
        Dictionary with keys ``'Method'`` (:class:`str`), ``'FPS'``
        (:class:`float`), and ``'Time(ms)'`` (:class:`float`).
        ``'FPS'`` is ``0.0`` and ``'Time(ms)'`` is ``-1`` on error.
    """
    # Check if GPU is involved to handle synchronization
    masks_arg = kwargs.get('masks_array')
    is_gpu = torch.cuda.is_available() and (
        (isinstance(masks_arg, torch.Tensor) and masks_arg.is_cuda) or
        kwargs.get('device') == 'cuda'
    )
    
    # Warmup
    try:
        for _ in range(n_warmup):
            func(**kwargs)
    except Exception as e:
        # Useful for debugging if a specific backend fails
        print(f"Error in {name}: {e}")
        return {"Method": name, "FPS": 0.0, "Time(ms)": -1} 

    # Timing
    if is_gpu: torch.cuda.synchronize()
    start_t = time.perf_counter()
    
    for _ in range(n_runs):
        func(**kwargs)
        
    if is_gpu: torch.cuda.synchronize()
    end_t = time.perf_counter()
    
    avg_time_ms = ((end_t - start_t) / n_runs) * 1000
    fps = 1000.0 / avg_time_ms if avg_time_ms > 0 else 0
    
    return {"Method": name, "FPS": fps, "Time(ms)": avg_time_ms}

def benchmark_scenario(n: int, h: int, w: int) -> None:
    """Run and print all converter benchmarks for a single ``(N, H, W)`` scenario.

    Args:
        n: Number of masks.
        h: Image height in pixels.
        w: Image width in pixels.
    """
    print(f"\n>>> Benchmarking Scenario: N={n}, H={h}, W={w}")
    
    mock = MockResults(n, h, w)
    results = []
    
    # --- Define the Variants ---
    
    # 1. NumPy Baseline
    results.append(run_test("NumPy | Vectorized", create_label_image, 
        {"masks_array": mock.masks_cpu, "use_loop": False, "use_numpy": True}))
    results.append(run_test("NumPy | Loop", create_label_image, 
        {"masks_array": mock.masks_cpu, "use_loop": True, "use_numpy": True}))

    if mock.masks_gpu is not None:
        # 2. GPU Resident (Best Case)
        results.append(run_test("Torch GPU | Vectorized (No Transfer)", create_label_image, 
            {"masks_array": mock.masks_gpu, "use_loop": False, "use_numpy": False, "device": None}))
        results.append(run_test("Torch GPU | Loop (No Transfer)", create_label_image, 
            {"masks_array": mock.masks_gpu, "use_loop": True, "use_numpy": False, "device": None}))

        # 3. CPU Input -> Device Arg
        results.append(run_test("Torch CPU In | Vectorized (Force CPU)", create_label_image, 
            {"masks_array": mock.masks_cpu, "use_loop": False, "use_numpy": False, "device": 'cpu'}))
        results.append(run_test("Torch CPU In | Loop (Force CUDA)", create_label_image, 
            {"masks_array": mock.masks_cpu, "use_loop": True, "use_numpy": False, "device": 'cuda'}))
        
        # 4. GPU Input -> Device Arg (Cross Device)
        results.append(run_test("Torch GPU In | Vectorized (Force CPU)", create_label_image, 
            {"masks_array": mock.masks_gpu, "use_loop": False, "use_numpy": False, "device": 'cpu'}))
        results.append(run_test("Torch GPU In | Loop (Force CUDA)", create_label_image, 
            {"masks_array": mock.masks_gpu, "use_loop": True, "use_numpy": False, "device": 'cuda'}))

    # Display Results for this Scenario
    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values(by="FPS", ascending=False)
        print(df.to_string(index=False, formatters={"FPS": "{:.1f}".format, "Time(ms)": "{:.2f}".format}))
    else:
        print("No results generated (GPU might be missing).")

# ==========================================
# 4. Main Runner
# ==========================================

def run_benchmark_scenarios() -> None:
    """Run all predefined benchmark scenarios and print results to stdout."""
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available. GPU tests will be skipped.")

    version = "0.1.0"
    print(f"[INFO]: Running benchmarks for imtools version: {version}")

    # Define Scenarios (N, Height, Width)
    scenarios = [
        (10, 640, 640),    # Standard
        #(100, 640, 640),   # Crowded
        #(50, 1920, 1080),  # Full HD
        #(250, 1280, 1280)  # High Res + Many Objects (Stress Test)
    ]

    for n, h, w in scenarios:
        benchmark_scenario(n, h, w)

if __name__ == "__main__":
    run_benchmark_scenarios()