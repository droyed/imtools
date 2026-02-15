# Changelog

## [0.2.0] - 2026-02-15

### Added
- New module `formats.py` with `pil_to_opencv()`, `opencv_to_pil()`, `to_pil_image()`, `to_numpy_image()`, and `imwrite()` for comprehensive PIL ↔ OpenCV/NumPy format conversions (11+ color mode support including 16-bit, scientific, palette, and transparency modes)
- New module `converters.py` with multi-backend mask-to-label-image converters: PyTorch GPU vectorized/loop and NumPy vectorized/loop implementations, plus high-level dispatchers `masks_to_label_image()`, `yolo_to_label_image()`, and `binary_mask_to_label_image()`
- New module `viz_pipeline.py` with `overlay_visualize()` unified visualization pipeline and `draw_labels()` for text annotation rendering
- New module `annotations.py` with `yolo_to_annotations()` and `label_image_to_annotations()` converters
- New module `label_formats.py` with `LabelFormat` enum (39 variants), `LabelContext` dataclass, and `resolve_label()` dispatcher for flexible annotation string formatting
- New module `common.py` with `Annotation`, `BlendConfig`, `LabelStyle`, and `TitleConfig` dataclasses for configuration and data representation
- New module `display_utils.py` with `show_cv2()`, `show_mpl()`, `show_cv2_fullscreen()`, and display/scaling utilities
- New module `compose.py` with `text_on_canvas()` and `stack_images()` (migrated and enhanced from removed modules)
- New `benchmarks/` sub-package with `run_benchmark_scenarios()` for measuring converter performance across backends (NumPy, PyTorch GPU/CPU); GPU vectorized achieves ~1722 FPS for 640×640 inputs
- New `demos/` package with six demo scripts covering color generation, format conversions, binary mask overlays, YOLO segmentation overlays, and SAM3 segmentation overlays
- `Makefile` with `install`, `lint`, `test`, `demo`, `clean`, and `help` targets
- `pyyaml` and `scikit-image` added as runtime dependencies in `pyproject.toml`
- Optional `test` dependency group in `pyproject.toml` with `scipy` and `ultralytics`
- New helper `_get_regions()` and `extract_region_metadata()` in `mask_utils.py`

### Changed
- Public API restructured in `__init__.py`: exports now come from `formats`, `converters`, `viz_pipeline`, `display_utils`, and `color_gen`
- `color_gen.py` cleaned up: removed inline demo code and `__main__` block
- `mask_utils.py`: removed `generate_label_image()` (moved to `converters.py`), removed direct OpenCV dependency
- `README.md` overhauled to reflect new segmentation visualization focus with badges, module overview table, YOLO workflow example, and demo instructions
- Version bumped to `0.2.0`

### Removed
- `draw.py` — `draw_mask_overlays()` and `text_on_canvas()` replaced by `viz_pipeline.py` and `compose.py`
- `image_layout.py` — `stack_images()` replaced by `compose.py`
- `read_write.py` — `load_image()` replaced by `formats.py`
- `tests/` directory — test suite to be rewritten against new API
- `requirements.txt`, `requirements-dev.txt`, `setup.py` — replaced by `pyproject.toml`
- `demos/demo_run_overlay_viewer.py` and related docs — superseded by new demo suite

## [0.1.1] - 2026-01-24

### Added
- New helper function `_extract_properties()` in `mask_utils.py` for extracting available properties from regionprops objects
- New helper function `_extract_info()` in `mask_utils.py` for extracting property values from regionprops objects
- Optional `return_props` parameter in `get_biggest_blob()` function to return region properties alongside the mask
- Support for `pathlib.Path` objects in `load_image()` function

### Changed
- `draw_mask_overlays()` in `draw.py` now explicitly converts images to RGB format via `convert_to_rgb=True` parameter
- `get_biggest_blob()` in `mask_utils.py` can now return a tuple of `(output_mask, props)` when `return_props=True`
- `load_image()` in `read_write.py` now accepts both string paths and `pathlib.Path` objects
- Updated type hints and docstrings to reflect `pathlib.Path` support
- Improved error messages in `load_image()` to use consistent path variable naming

### Technical Details
- `load_image()` now converts `Path` objects to strings internally for `cv2.imread()` compatibility
- Region properties dictionary includes all available attributes from scikit-image's regionprops
