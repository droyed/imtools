# Implementation Plan: Write README.md for `imtools`

## Objective

Generate a concise, well-structured `README.md` for the `imtools` package by inspecting the source code directly.

---

## Step 1: Understand the Package Structure

- Read `src/imtools/__init__.py` to identify the top-level public API, version, and package description.
- List all subpackages: `annotations`, `benchmarks`, `core`, `masks`, `viz`.

## Step 2: Inspect Each Subpackage

For **each** subpackage, read every `.py` file and extract:

- Public classes, functions, and constants (skip `_`-prefixed internals).
- One-line purpose of each module (from docstrings or code inspection).
- Key dependencies (e.g., numpy, PIL, opencv, matplotlib).

### 2a — `core/`

- `types.py` — data types / type aliases used across the package.
- `formats.py` — image format utilities.

### 2b — `annotations/`

- `labels.py` — label data structures or mappings.
- `parsers.py` — generic annotation parsing.
- `yolo.py` — YOLO-format annotation handling.

### 2c — `masks/`

- `converters.py` — mask format conversions (e.g., polygon ↔ bitmap).
- `ops.py` — mask operations (union, intersection, etc.).

### 2d — `viz/`

- `colors.py` — color palettes / utilities.
- `compose.py` — image composition / tiling.
- `display.py` — display helpers (e.g., matplotlib wrappers).
- `overlay_viewer.py` — interactive overlay viewing.
- `pipeline.py` — visualization pipeline abstraction.

### 2e — `benchmarks/`

- `benchmark_converters.py` — performance benchmarks for converters.
- `__main__.py` — CLI entry point for running benchmarks.

## Step 3: Identify Installation Requirements

- Check for `pyproject.toml`, `setup.py`, `setup.cfg`, or `requirements.txt` at the project root.
- Extract the dependency list and any optional/extra dependency groups.
- Note the minimum Python version if specified.

## Step 4: Check for CLI Entry Points or Scripts

- Look at `benchmarks/__main__.py` for `python -m imtools.benchmarks` usage.
- Check `pyproject.toml` `[project.scripts]` for any registered CLI commands.

## Step 5: Draft the README

Write the following sections — keep each section **brief** (aim for the whole README to be under ~150 lines):

### Sections to Include

1. **Title & one-liner** — package name + single sentence describing what it does.
2. **Installation** — `pip install` command (or editable install instructions).
3. **Package Overview** — short table or list mapping each subpackage to its purpose.
4. **Quick Start** — 2–3 minimal code snippets showing the most common workflows (e.g., load annotations, convert masks, visualize results).
5. **Benchmarks** — how to run them (`python -m imtools.benchmarks`).
6. **License** — reference the license file if present, otherwise leave a placeholder.

## Step 6: Review & Finalize

- Verify all referenced function/class names actually exist in the source.
- Ensure code snippets use correct import paths (`from imtools.masks import ...`).
- Remove any section that has no real content (e.g., skip License if no LICENSE file exists).
- Write the final `README.md` to the project root.
