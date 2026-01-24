# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
