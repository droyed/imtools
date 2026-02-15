# Changelog

## [1.1.0] - 2026-02-15

### Added

- New hierarchical subpackage structure: `core/`, `annotations/`, `masks/`, `viz/`, `benchmarks/`
- `core/types.py` — `Annotation`, `BlendConfig`, `TitleConfig`, `LabelStyle` dataclasses with preset factory methods
- `core/formats.py` — PIL ↔ OpenCV conversion (`pil_to_opencv`, `opencv_to_pil`, `to_pil_image`, `to_numpy_image`, `imwrite`) handling 11+ color modes
- `annotations/labels.py` — `LabelFormat` enum with 40+ format variants, `LabelContext` dataclass, and `resolve_label()` dispatcher
- `annotations/parsers.py` — `yolo_to_annotations()` and `label_image_to_annotations()` converters
- `annotations/yolo.py` — `yolo_to_label_image()` converting YOLO `Results` to a 2D integer label image
- `masks/converters.py` — four backend implementations for mask-to-label-image conversion: PyTorch vectorized (~1722 FPS on 640×640), PyTorch loop, NumPy vectorized, NumPy loop; unified `masks_to_label_image()` dispatcher and `binary_mask_to_label_image()`
- `viz/pipeline.py` — `overlay_visualize()`, `draw_labels()`, and `create_label_overlay_from_labelimg()`
- `viz/colors.py` — consolidated color generation module (`generate_colors()`, `create_color_palette_image()`)
- `viz/compose.py` — `text_on_canvas()`, `stack_images()`, `add_title()` with enhanced options (gap, background color, aspect ratio)
- `viz/display.py` — `show_cv2()`, `show_mpl()`, `show_cv2_fullscreen()` display helpers
- `benchmarks/` subpackage with `run_benchmark_scenarios()` and a `__main__` CLI entry point
- `Makefile` with `install`, `lint`, `test`, `demo`, `docs-serve`, `docs-build`, `clean` targets
- `mkdocs.yml` — MkDocs Material site configuration with light/dark mode toggle and `mkdocstrings` API reference
- Extended `docs/` with API reference pages for all five subpackages and a `USAGE_overlay_visualize.md` guide
- `dev-docs/plan-docs/` — internal planning documents for reorganization, docstrings, demos, tests, and README
- 6 demo scripts under `demos/`: `demo_config.py`, `demo_conversions.py`, `demo_overlay_mask.py`, `demo_overlay_yolo.py`, `demo_overlay_sam3.py` (plus the existing overlay-viewer demo)
- Full pytest suite with `tests/conftest.py` (shared fixtures) and 18 test modules organized by subpackage
- `pyyaml` and `scikit-image` added as runtime dependencies
- Optional dependency groups: `[test]` (scipy, ultralytics) and `[docs]` (mkdocs, mkdocs-material, mkdocstrings)
- Pytest markers `slow` and `integration` declared in `pyproject.toml`

### Changed

- Package restructured from a flat layout (`read_write.py`, `draw.py`, `mask_utils.py`, `image_layout.py`, `color_gen.py`) into proper subpackages
- `__init__.py` public API updated to re-export symbols from all new subpackages
- `load_image()` (from `read_write.py`) replaced by `to_numpy_image()` and `to_pil_image()` in `core/formats.py`
- `draw_mask_overlays()` (from `draw.py`) replaced by `viz.pipeline.overlay_visualize()` and `viz.pipeline.draw_labels()`
- `stack_images()` (from `image_layout.py`) moved to `viz/compose.py` with new `gap`, `bg_color`, and `preserve_aspect` parameters
- `get_biggest_blob()`, `bbox_from_mask()`, `fill_holes_mask()` moved from `mask_utils.py` to `masks/ops.py`
- `generate_label_image()` (from `mask_utils.py`) moved to `masks/converters.py` and renamed `binary_mask_to_label_image()`
- `color_gen.py` functions moved to `viz/colors.py`; demo code removed from the module
- `OverlayViewer` / `run_overlay_viewer()` moved from `viz/overlay_viewer.py` to the new `viz/` subpackage (same filename, new import path `imtools.viz.overlay_viewer`)
- `pyproject.toml` updated with new dependency groups, pytest configuration, and corrected package metadata
- README overhauled: new module overview table, YOLO segmentation workflow, benchmark and demo instructions
- Tests rewritten from scratch and reorganized into subpackage-mirroring directory tree

### Removed

- `read_write.py` — replaced by `core/formats.py`
- `draw.py` — replaced by `viz/pipeline.py`
- `mask_utils.py` — replaced by `masks/ops.py` and `masks/converters.py`
- `image_layout.py` — replaced by `viz/compose.py`
- `color_gen.py` as a top-level module — functionality moved to `viz/colors.py`
- `setup.py` — build configuration consolidated into `pyproject.toml`
- `requirements.txt` and `requirements-dev.txt` — replaced by `pyproject.toml` dependency groups
- `docs/code_changes/` directory

## [1.0.0] - 2026-02-15

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
