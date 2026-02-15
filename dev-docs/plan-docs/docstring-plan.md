# Docstring Implementation Plan for `imtools`

## Objective

Audit every source file in `src/imtools/` and ensure:

1. **Docstrings** — Every public function, class, and method has a comprehensive Google-style docstring (summary, parameters, returns, raises, example).
2. **Type hints** — Every function signature and class attribute has complete, accurate type annotations.

---

## Conventions

- **Style:** Google-style docstrings.
- **Scope:** All public functions, classes, and methods. Private/internal helpers (`_name`) get a one-line summary at minimum.
- **Content per docstring:**
  1. One-line summary (imperative mood, e.g. "Convert mask to polygon.")
  2. Extended description (if non-obvious behavior, edge cases, or design rationale exist)
  3. `Args:` block with name, type, and description for each parameter
  4. `Returns:` / `Yields:` block with type and description
  5. `Raises:` block listing expected exceptions
  6. `Example:` block with a minimal, runnable snippet (for all public API surfaces)
- **Validation:** After each module, run `python -c "import imtools.<module>"` to confirm no syntax errors were introduced.

### Type Hint Conventions

- **Imports:** Use `from __future__ import annotations` at the top of every file to enable PEP 604 (`X | Y`) syntax and forward references consistently.
- **Scope:** All function parameters, return types, class/instance attributes, and module-level variables.
- **Specificity rules:**
  - Prefer concrete types over `Any`. Use `Any` only when truly unconstrained.
  - Use `numpy.ndarray` (or `np.ndarray`) for array parameters; add shape/dtype info in the docstring (not the annotation).
  - Use `pathlib.Path | str` for filesystem paths (or `os.PathLike[str]` where appropriate).
  - Use `collections.abc` abstract types for inputs (`Sequence`, `Mapping`, `Iterable`) and concrete types for outputs (`list`, `dict`).
  - Use `TypeAlias` for complex or reused type expressions (define them in `core/types.py` where possible).
  - Use `Optional[X]` or `X | None` consistently (prefer `X | None` with future annotations).
  - Class attributes: annotate in the class body or `__init__`; for dataclasses use field annotations.
  - Callbacks / callables: use `Callable[[ArgTypes], ReturnType]` with specific signatures.
- **Validation per module:** Run `mypy --ignore-missing-imports <file>` after annotating each file (no errors/warnings on the edited file).

---

## Step-by-step Plan

### Phase 1 — Core Layer

Start here because other modules depend on these types and utilities.

#### Step 1: `core/types.py` — Docstrings + Type Hints

- Read the file and inventory all classes, dataclasses, TypedDict definitions, NamedTuples, type aliases, and standalone functions.
- Add `from __future__ import annotations` if missing.
- Add module-level docstring describing the purpose of the types module.
- Add class-level docstrings with attribute descriptions for every type/dataclass.
- Add docstrings to any `__init__`, `__post_init__`, and public methods.
- Add type annotations to all function parameters and return types.
- Annotate all class/instance attributes and dataclass fields with precise types.
- Define reusable `TypeAlias` entries for types shared across the package (e.g., `BBox`, `Polygon`, `Mask`).
- Verify: `python -c "from imtools.core.types import *"`
- Verify: `mypy --ignore-missing-imports src/imtools/core/types.py`

#### Step 2: `core/formats.py` — Docstrings + Type Hints

- Inventory all functions/classes related to image format handling.
- Add `from __future__ import annotations` if missing.
- Add module-level docstring.
- Document each function: what formats are supported, expected input/output, edge cases.
- Include usage examples for key conversion or validation functions.
- Add type annotations to all function signatures and return types.
- Use `TypeAlias` types from `core/types.py` where applicable.
- Verify: `python -c "from imtools.core.formats import *"`
- Verify: `mypy --ignore-missing-imports src/imtools/core/formats.py`

#### Step 3: `core/__init__.py`

- Add module-level docstring summarizing the core subpackage.
- Document any re-exported symbols.

---

### Phase 2 — Annotations Layer

#### Step 4: `annotations/labels.py` — Docstrings + Type Hints

- Inventory label-related classes and functions.
- Add `from __future__ import annotations` if missing.
- Add module-level docstring.
- Document each class (attributes, methods) and standalone function.
- Include examples showing label creation and manipulation.
- Add type annotations to all function parameters, return types, and class attributes.
- Import and use shared types from `core/types.py`.
- Verify: `python -c "from imtools.annotations.labels import *"`
- Verify: `mypy --ignore-missing-imports src/imtools/annotations/labels.py`

#### Step 5: `annotations/parsers.py` — Docstrings + Type Hints

- Inventory parser functions.
- Add `from __future__ import annotations` if missing.
- Add module-level docstring describing supported annotation formats.
- Document each parser: input format, expected file structure, return types.
- Add examples with sample input/output.
- Add type annotations to all function signatures and return types.
- Use `Path | str` for file path parameters; type dict/list returns precisely (e.g., `list[LabelType]`).
- Verify: `python -c "from imtools.annotations.parsers import *"`
- Verify: `mypy --ignore-missing-imports src/imtools/annotations/parsers.py`

#### Step 6: `annotations/yolo.py` — Docstrings + Type Hints

- Inventory YOLO-specific utilities.
- Add `from __future__ import annotations` if missing.
- Add module-level docstring.
- Document each function: YOLO format specifics (normalized xywh, class indices, etc.), conversion directions, coordinate conventions.
- Add examples showing YOLO ↔ other format conversion.
- Add type annotations to all function signatures; use precise array/tuple types for coordinate data.
- Verify: `python -c "from imtools.annotations.yolo import *"`
- Verify: `mypy --ignore-missing-imports src/imtools/annotations/yolo.py`

#### Step 7: `annotations/__init__.py`

- Add module-level docstring summarizing the annotations subpackage.
- Document re-exported symbols.

---

### Phase 3 — Masks Layer

#### Step 8: `masks/converters.py` — Docstrings + Type Hints

- Inventory mask conversion functions.
- Add `from __future__ import annotations` if missing.
- Add module-level docstring.
- Document each function: input/output types (e.g., binary mask → polygon, RLE → mask), array shapes and dtypes expected.
- Include examples.
- Add type annotations to all function signatures and return types; use `np.ndarray` for masks, precise types for RLE/polygon data.
- Verify: `python -c "from imtools.masks.converters import *"`
- Verify: `mypy --ignore-missing-imports src/imtools/masks/converters.py`

#### Step 9: `masks/ops.py` — Docstrings + Type Hints

- Inventory mask operations (union, intersection, erosion, dilation, etc.).
- Add `from __future__ import annotations` if missing.
- Add module-level docstring.
- Document each operation: expected mask format, parameters, return shape/dtype.
- Include examples.
- Add type annotations to all function signatures and return types.
- Verify: `python -c "from imtools.masks.ops import *"`
- Verify: `mypy --ignore-missing-imports src/imtools/masks/ops.py`

#### Step 10: `masks/__init__.py`

- Add module-level docstring.
- Document re-exported symbols.

---

### Phase 4 — Visualization Layer

#### Step 11: `viz/colors.py` — Docstrings + Type Hints

- Inventory color utilities (palettes, color mapping, conversions).
- Add `from __future__ import annotations` if missing.
- Add module-level docstring.
- Document each function/class with color format details (RGB, BGR, hex, etc.).
- Add type annotations; define color type aliases if needed (e.g., `Color = tuple[int, int, int]`).
- Verify: `python -c "from imtools.viz.colors import *"`
- Verify: `mypy --ignore-missing-imports src/imtools/viz/colors.py`

#### Step 12: `viz/compose.py` — Docstrings + Type Hints

- Inventory compositing/layout functions.
- Add `from __future__ import annotations` if missing.
- Add module-level docstring.
- Document each function: what it composes, input image expectations, output shape.
- Include examples.
- Add type annotations to all function signatures and return types.
- Verify: `python -c "from imtools.viz.compose import *"`
- Verify: `mypy --ignore-missing-imports src/imtools/viz/compose.py`

#### Step 13: `viz/display.py` — Docstrings + Type Hints

- Inventory display/show functions.
- Add `from __future__ import annotations` if missing.
- Add module-level docstring.
- Document each function: backend used (matplotlib, cv2, etc.), blocking behavior, parameters.
- Add type annotations to all function signatures; use `None` return for display-only functions.
- Verify: `python -c "from imtools.viz.display import *"`
- Verify: `mypy --ignore-missing-imports src/imtools/viz/display.py`

#### Step 14: `viz/overlay_viewer.py` — Docstrings + Type Hints

- Inventory viewer class(es) and functions.
- Add `from __future__ import annotations` if missing.
- Add module-level docstring.
- Document the viewer class: initialization, methods, keyboard/mouse interaction if any.
- Include a usage example showing typical viewer workflow.
- Add type annotations to all methods (including `self` return types for fluent APIs), constructor parameters, and instance attributes.
- Verify: `python -c "from imtools.viz.overlay_viewer import *"`
- Verify: `mypy --ignore-missing-imports src/imtools/viz/overlay_viewer.py`

#### Step 15: `viz/pipeline.py` — Docstrings + Type Hints

- Inventory pipeline functions/classes.
- Add `from __future__ import annotations` if missing.
- Add module-level docstring.
- Document the pipeline API: how steps are chained, configuration, input/output.
- Include an end-to-end example.
- Add type annotations to all functions and methods; type pipeline step callables with `Callable` signatures.
- Verify: `python -c "from imtools.viz.pipeline import *"`
- Verify: `mypy --ignore-missing-imports src/imtools/viz/pipeline.py`

#### Step 16: `viz/__init__.py`

- Add module-level docstring.
- Document re-exported symbols.

---

### Phase 5 — Benchmarks Layer

#### Step 17: `benchmarks/benchmark_converters.py` — Docstrings + Type Hints

- Inventory benchmark functions.
- Add `from __future__ import annotations` if missing.
- Add module-level docstring.
- Document each benchmark: what it measures, how to interpret results.
- Add type annotations to all function signatures and return types.
- Verify: `python -c "from imtools.benchmarks.benchmark_converters import *"`
- Verify: `mypy --ignore-missing-imports src/imtools/benchmarks/benchmark_converters.py`

#### Step 18: `benchmarks/__main__.py` — Docstrings + Type Hints

- Add `from __future__ import annotations` if missing.
- Add module-level docstring explaining CLI usage (`python -m imtools.benchmarks`).
- Document any argument parsing and entry-point functions.
- Add type annotations to all functions (including `main() -> None`).

#### Step 19: `benchmarks/__init__.py`

- Add module-level docstring.

---

### Phase 6 — Top-Level Init

#### Step 20: `__init__.py`

- Add package-level docstring summarizing `imtools`: purpose, subpackages, quick-start example.
- Document the public API surface (re-exported names, version, etc.).

---

### Phase 7 — Validation & Cleanup

#### Step 21: Full import check

```bash
python -c "import imtools; print('OK')"
```

#### Step 22: Docstring coverage audit

Run a quick script to verify every public symbol has a docstring:

```bash
python -c "
import importlib, pkgutil, inspect, imtools

missing = []
for importer, modname, ispkg in pkgutil.walk_packages(imtools.__path__, 'imtools.'):
    mod = importlib.import_module(modname)
    for name, obj in inspect.getmembers(mod):
        if name.startswith('_'):
            continue
        if inspect.isfunction(obj) or inspect.isclass(obj):
            if not obj.__doc__:
                missing.append(f'{modname}.{name}')

if missing:
    print('Missing docstrings:')
    for m in missing:
        print(f'  - {m}')
else:
    print('All public symbols documented.')
"
```

#### Step 23: Type hint coverage audit

Run mypy across the full package to catch missing or incorrect annotations:

```bash
mypy --ignore-missing-imports --disallow-untyped-defs --no-error-summary src/imtools/
```

Additionally, verify no untyped public functions remain:

```bash
python -c "
import importlib, pkgutil, inspect, typing, imtools

untyped = []
for importer, modname, ispkg in pkgutil.walk_packages(imtools.__path__, 'imtools.'):
    mod = importlib.import_module(modname)
    for name, obj in inspect.getmembers(mod, inspect.isfunction):
        if name.startswith('_'):
            continue
        hints = typing.get_type_hints(obj, include_extras=True)
        sig = inspect.signature(obj)
        for param_name, param in sig.parameters.items():
            if param_name == 'self' or param_name == 'cls':
                continue
            if param_name not in hints:
                untyped.append(f'{modname}.{name}() -> param \"{param_name}\"')
        if 'return' not in hints:
            untyped.append(f'{modname}.{name}() -> return type')

if untyped:
    print('Missing type hints:')
    for u in untyped:
        print(f'  - {u}')
else:
    print('All public functions fully typed.')
"
```

#### Step 24: Fix any gaps found in Steps 22–23

- Address every item in the missing docstrings and missing type hints lists.
- Re-run both audits until clean.

#### Step 25: Final mypy strict pass

Run a final strict check to confirm no regressions:

```bash
mypy --ignore-missing-imports --disallow-untyped-defs --warn-return-any --warn-unused-ignores src/imtools/
```

---

## Notes for Claude Code Usage

- Feed this plan one step at a time. For each step, provide the command: *"Read `<file>` and add comprehensive Google-style docstrings and complete type annotations to all public functions, classes, and methods. Follow the conventions in the plan."*
- After each step, run both verification commands (import check + mypy) before moving on.
- If a function's behavior is ambiguous from the code alone, add a `# TODO: verify behavior` comment in the docstring for manual review later.
- If an accurate type cannot be determined from context, use a precise `# TODO: narrow type` comment alongside a temporary `Any` annotation — do not leave it untyped.
- When adding `from __future__ import annotations`, check for any runtime use of annotations (e.g., `isinstance` checks on string-ified types, Pydantic models) that could break — adjust as needed.
