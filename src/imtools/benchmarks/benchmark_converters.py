'''
Important Note on Running It

Because you are using relative imports (e.g., from ..converters) inside the benchmark script, 
you must run it as a module from the root folder:

```bash
# Run from the folder containing 'imtools'
python -m imtools.benchmarks.benchmark_masks
'''

import time
import numpy as np
import torch
import pandas as pd

# Update imports to match the new converters.py function names
from ..converters import (
    masks_to_label_image_torch_vectorized, 
    masks_to_label_image_torch_loop, 
    masks_to_label_image_numpy_loop, 
    masks_to_label_image_numpy_vectorized
)


def create_label_image(masks_array, use_loop=True, use_numpy=True, device=None):
    """
    Wrapper function to route benchmark calls to the correct 
    new function in converters.py
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
    def __init__(self, n, h, w, device='cpu'):
        self.n = n
        self.h = h
        self.w = w
        # Generate random boolean masks (simulated as 0/1 uint8)
        # 20% probability of a pixel being part of a mask
        self.masks_cpu = (np.random.rand(n, h, w) > 0.8).astype(np.uint8)
        
        if torch.cuda.is_available():
            self.masks_gpu = torch.from_numpy(self.masks_cpu).cuda()
        else:
            self.masks_gpu = None

# ==========================================
# 3. Benchmarking Logic
# ==========================================

def run_test(name, func, kwargs, n_runs=20, n_warmup=5):
    """Runs a single test case with timing."""
    # Check if GPU is involved to handle synchronization
    is_gpu = torch.cuda.is_available() and (
        (isinstance(kwargs.get('masks_array'), torch.Tensor) and kwargs.get('masks_array').is_cuda) or 
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

def benchmark_scenario(n, h, w):
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

def run_benchmark_scenarios():
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