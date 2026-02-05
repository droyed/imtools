# Code Review: Overlay Viewer Refactoring Summary

## Overview

This document summarizes all changes made to `overlay_viewer.py` (Dear PyGui Mask Overlay Viewer) during the code review and refactoring session for the commits from start to finish before pushing.

---

## Changes by Category

### 1. Import Organization
- Moved `os` and `platform` from inside `get_system_font_path()` to module-level imports
- Added `from typing import Any, Callable` for type hints

### 2. Type Annotations
- Fixed incorrect return types: `-> dict` changed to `-> tuple[Image.Image, dict]` for both `run()` and `run_overlay_viewer()`
- Added type hints to all callback and helper methods:
  - `on_alpha_change`, `on_method_change`, `on_param_change`, `on_third_change`
  - `on_key_press`, `on_window_resize`
  - `_compute_lut`, `_refresh_display`, `_redraw_image`, `_cycle_combo`

### 3. Constants & Magic Strings
- Added `FONT_SIZE = 15`
- Added `PRIMARY_WINDOW_TAG = "primary_window"`
- Added `WINDOW_HANDLER_TAG = "window_handler"`
- Replaced all hardcoded string occurrences with constants

### 4. Input Validation
- Added `_validate_inputs()` static method with checks for:
  - `img`: must be 3D numpy array with 3 channels (RGB)
  - `label_image`: must be 2D numpy array
  - Matching spatial dimensions between `img` and `label_image`
  - Valid `initial_method` value

### 5. Exception Handling
- Replaced bare `except Exception: pass` with specific `except (SystemError, RuntimeError):` plus explanatory comment

### 6. Sparse Label Handling (Memory Safety)
- Added `_remap_sparse_labels()` static method
- Prevents memory crash when label IDs are sparse (e.g., COCO dataset with ID 20000)
- Uses `np.searchsorted` for efficient O(N log U) remapping
- New attributes: `_unique_labels`, `_contiguous_labels`

### 7. Performance Optimization (Pre-allocated Buffers)
- Added pre-allocated buffers in `__init__`:
  - `_img_float`: Pre-normalized image (float32, 0.0-1.0)
  - `_rgb_buffer`: Contiguous RGB buffer for `cv2.addWeighted`
  - `_display_buffer`: Flat RGBA buffer for DPG texture
  - `_buffer_view`: Reshaped view into display buffer
- Changed LUT storage from `uint8` to `float32` (`_lut_float`)
- Refactored `blend()` for zero-allocation operation per frame

### 8. Method Refactoring (`run()` decomposition)
Broke monolithic `run()` (~145 lines) into focused helper methods:

| Method | Purpose |
|--------|---------|
| `_apply_theme()` | Create and bind dark theme |
| `_setup_fonts()` | Load system font |
| `_create_texture()` | Initialize dynamic texture |
| `_build_alpha_slider()` | Alpha slider widget |
| `_build_method_combo()` | Method selection combo |
| `_build_param_combo()` | Parameter combo widget |
| `_build_third_combo()` | Colormap combo widget |
| `_build_control_panel()` | Control panel orchestrator |
| `_build_ui()` | Main UI orchestrator |
| `_register_handlers()` | Keyboard & resize handlers |
| `_configure_viewport()` | Viewport setup |

- Added `try/finally` block to guarantee `dpg.destroy_context()` cleanup

---

## Summary Table

| Issue | Severity | Resolution |
|-------|----------|------------|
| Wrong return type annotations | High | Fixed to `tuple[Image.Image, dict]` |
| Bare `except: pass` | High | Specific exceptions + comment |
| Sparse labels → memory crash | High | Contiguous remapping via `searchsorted` |
| Imports inside function | Low | Moved to module level |
| Unused variable | Low | Changed to `_` |
| Magic strings | Medium | Centralized as constants |
| Missing type hints | Medium | Added to all methods |
| No input validation | Medium | Added `_validate_inputs()` |
| Slow per-frame allocation | Low-Medium | Pre-allocated buffers |
| Monolithic `run()` method | Medium | Decomposed into 11 helper methods |
| No cleanup guarantee | Medium | Added `try/finally` |

---

## File Statistics

| Metric | Before | After |
|--------|--------|-------|
| Total lines | ~766 | ~958 |
| `run()` method lines | ~145 | ~15 |
| Helper methods | 0 | 11 |
| Type-annotated methods | Partial | Complete |

---

## Not Implemented

- **Step 9 (Null Object Pattern):** Optional polish to eliminate `if config is None` checks. Skipped as low priority.
