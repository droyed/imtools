# Unit Test Implementation Plan for `imtools`

## Overview

Set up a comprehensive `pytest`-based test suite under `tests/` mirroring the `src/imtools/` package structure. All tests runnable via `pytest tests/ -v`.

---

## Step 0 — Project Prep

1. **Confirm dev dependencies exist.** Ensure `pytest` (and optionally `pytest-cov`) are listed in `pyproject.toml` or `requirements-dev.txt`. If not, add them.
2. **Verify the package is importable.** Run `python -c "import imtools"` from the repo root. If it fails, ensure `src/` layout is configured correctly in `pyproject.toml` (e.g. `[tool.setuptools.packages.find] where = ["src"]`) and install in editable mode: `pip install -e .`.

---

## Step 1 — Create the `tests/` Directory Structure

Mirror the source layout so every module gets a corresponding test file:

```
tests/
├── conftest.py                        # Shared fixtures & test data helpers
├── core/
│   ├── __init__.py
│   ├── test_formats.py
│   └── test_types.py
├── annotations/
│   ├── __init__.py
│   ├── test_labels.py
│   ├── test_parsers.py
│   └── test_yolo.py
├── masks/
│   ├── __init__.py
│   ├── test_converters.py
│   └── test_ops.py
├── viz/
│   ├── __init__.py
│   ├── test_colors.py
│   ├── test_compose.py
│   ├── test_display.py
│   └── test_overlay_viewer.py
└── benchmarks/
    ├── __init__.py
    └── test_benchmark_converters.py
```

**Claude Code instructions:**

- Create every directory and `__init__.py` file listed above.
- Each `test_*.py` file should start with a module docstring and the relevant import (e.g. `from imtools.core.formats import ...`).

---

## Step 2 — Configure `pytest`

Add a `[tool.pytest.ini_options]` section to `pyproject.toml` (or create `pytest.ini` / `setup.cfg`):

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
addopts = "-v --tb=short"
markers = [
    "slow: marks tests that are slow (deselect with '-m \"not slow\"')",
    "integration: marks integration tests requiring external resources",
]
```

This ensures `pytest tests/ -v` resolves imports from `src/` correctly.

---

## Step 3 — Build Shared Fixtures (`tests/conftest.py`)

Create reusable fixtures that many test modules will need:

| Fixture | Purpose |
|---|---|
| `sample_image` | Returns a small numpy array (e.g. 64×64×3 uint8) for any test needing an image. |
| `sample_mask` | Returns a binary or multi-class mask array. |
| `sample_bbox_list` | Returns a list of bounding-box dicts/tuples in the format `imtools` expects. |
| `sample_yolo_annotation` | Returns a string or file path representing a YOLO-format annotation. |
| `tmp_image_file` | Writes `sample_image` to a temp PNG via `tmp_path` and yields the path. |
| `tmp_yolo_dir` | Creates a temp directory populated with a few YOLO `.txt` label files and corresponding dummy images. |

**Claude Code instructions:**

- Read each source module to understand the data types it expects (numpy arrays, file paths, dicts, etc.).
- Build fixtures that produce minimal but valid instances of those types.
- Use `@pytest.fixture` with appropriate `scope` (default `function`; use `session` for expensive I/O).

---

## Step 4 — Write Tests Module-by-Module

For each module, follow this pattern:

1. **Read the source file** to catalog every public function/class.
2. **Write at least one test per public function/method**, covering:
   - **Happy path** — typical valid input → expected output.
   - **Edge cases** — empty inputs, single-element inputs, boundary values.
   - **Error handling** — invalid inputs raise the expected exceptions (`pytest.raises`).
3. **Use parametrize** (`@pytest.mark.parametrize`) for functions with multiple input formats or value ranges.

### 4a — `tests/core/test_types.py`

- Test any custom dataclasses, enums, or type-validation helpers defined in `core/types.py`.
- Verify construction, equality, serialization, and invalid-field rejection.

### 4b — `tests/core/test_formats.py`

- Test image format detection, conversion, and any I/O wrappers.
- Parametrize over supported formats (PNG, JPEG, BMP, etc.) if applicable.
- Use `tmp_path` for any file-writing tests.

### 4c — `tests/annotations/test_labels.py`

- Test label creation, validation, and any label-mapping utilities.
- Edge case: duplicate labels, empty label list.

### 4d — `tests/annotations/test_parsers.py`

- Test parsing from various annotation file formats.
- Provide small inline or fixture-based sample files.
- Verify round-trip: parse → serialize → parse yields identical data.

### 4e — `tests/annotations/test_yolo.py`

- Test YOLO-specific read/write using `tmp_yolo_dir` fixture.
- Verify coordinate normalization/denormalization.
- Parametrize over single-object and multi-object annotations.

### 4f — `tests/masks/test_converters.py`

- Test mask format conversions (e.g. binary ↔ polygon ↔ RLE).
- Verify round-trip fidelity (original ≈ converted-back within tolerance).
- Edge cases: all-zero mask, all-one mask, single-pixel mask.

### 4g — `tests/masks/test_ops.py`

- Test mask operations (union, intersection, erosion, dilation, etc.).
- Use `np.testing.assert_array_equal` or `assert_array_almost_equal`.
- Edge cases: non-overlapping masks, fully overlapping masks.

### 4h — `tests/viz/test_colors.py`

- Test color palette generation, hex↔RGB conversion, etc.
- Verify output types and value ranges (0–255 or 0.0–1.0).

### 4i — `tests/viz/test_compose.py`

- Test image composition / tiling / grid layout functions.
- Verify output shape matches expectations given input shapes + layout params.

### 4j — `tests/viz/test_display.py`

- For interactive display functions (e.g. `plt.show()` wrappers), test that they **return** or **create** the expected figure/axis objects without actually rendering.
- Use `matplotlib.use("Agg")` backend in a fixture or `conftest.py` to avoid GUI pop-ups.

### 4k — `tests/viz/test_overlay_viewer.py`

- Test overlay blending logic (alpha, colormap application).
- Mock or stub any GUI/interactive components; test the pure computation parts.

### 4l — `tests/benchmarks/test_benchmark_converters.py`

- Test that benchmark utilities run without error on small synthetic data.
- Mark with `@pytest.mark.slow` if they take >1 s.

---

## Step 5 — Add Test Utilities (`tests/helpers.py`, optional)

If multiple test files share comparison logic, create `tests/helpers.py` with:

- `assert_images_equal(a, b)` — wraps `np.testing.assert_array_equal` with a descriptive message.
- `assert_bboxes_close(a, b, tol)` — compares bounding box lists within a tolerance.
- `generate_random_mask(h, w, n_classes)` — helper for mask tests.

---

## Step 6 — Validate the Full Suite

Run the complete suite and fix any issues:

```bash
# Basic run
pytest tests/ -v

# With coverage report
pytest tests/ -v --cov=imtools --cov-report=term-missing

# Exclude slow tests during development
pytest tests/ -v -m "not slow"
```

**Acceptance criteria:**

- `pytest tests/ -v` exits with **0 failures**.
- Every public function in the package has at least one corresponding test.
- Coverage is reported (target ≥ 80% line coverage as a starting goal).

---

## Step 7 — CI Integration (Optional Follow-Up)

Add a GitHub Actions workflow (`.github/workflows/tests.yml`):

```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -e ".[dev]"
      - run: pytest tests/ -v --cov=imtools
```

---

## Summary Checklist

| # | Task | Status |
|---|---|---|
| 0 | Confirm `pytest` installable and package importable | ☐ |
| 1 | Create `tests/` directory tree with all test files | ☐ |
| 2 | Configure `pytest` in `pyproject.toml` | ☐ |
| 3 | Write shared fixtures in `conftest.py` | ☐ |
| 4a–4l | Write tests for each module (12 test files) | ☐ |
| 5 | Extract shared test helpers if needed | ☐ |
| 6 | Run full suite, achieve 0 failures, report coverage | ☐ |
| 7 | (Optional) Add CI workflow | ☐ |
