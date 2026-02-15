# Requirements.txt Generator - Implementation Plan

## Overview
Generate a comprehensive `requirements.txt` file by parsing all relevant files in a Python codebase.

---

## Step 1: Discover All Relevant Files

Scan the project directory recursively for:
- `**/*.py` - Python source files
- `**/*.ipynb` - Jupyter notebooks
- `setup.py` - Legacy package configuration
- `setup.cfg` - Declarative setup configuration
- `pyproject.toml` - Modern Python packaging (PEP 517/518)
- `requirements*.txt` - Existing requirements files
- `Pipfile` - Pipenv dependency file
- `poetry.lock` / `pyproject.toml` - Poetry projects
- `conda.yaml` / `environment.yml` - Conda environments

---

## Step 2: Parse Python Source Files (*.py)

### 2.1 Extract Import Statements
Use AST (Abstract Syntax Tree) parsing for accuracy:

```python
import ast

def extract_imports(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        tree = ast.parse(f.read())
    
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split('.')[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module.split('.')[0])
    return imports
```

### 2.2 Handle Edge Cases
- **Try/except imports**: Parse both branches
- **Conditional imports**: `if sys.platform == 'win32': import winreg`
- **Dynamic imports**: `importlib.import_module('package')` - extract string literal if possible
- **Type checking imports**: `if TYPE_CHECKING: import pandas`

---

## Step 3: Parse Jupyter Notebooks (*.ipynb)

### 3.1 Structure
Notebooks are JSON files with this structure:
```json
{
  "cells": [
    {
      "cell_type": "code",
      "source": ["import pandas as pd\n", "import numpy as np"]
    }
  ]
}
```

### 3.2 Extraction Steps
1. Load JSON content
2. Filter cells where `cell_type == "code"`
3. Join `source` array into single string
4. Apply same AST parsing as .py files
5. Also check for `!pip install` or `%pip install` magic commands

---

## Step 4: Parse setup.py

### 4.1 Static Analysis (Preferred)
Look for common patterns:
```python
setup(
    install_requires=[
        'requests>=2.25.0',
        'numpy',
    ],
    extras_require={
        'dev': ['pytest', 'black'],
    }
)
```

### 4.2 Extraction Strategy
1. Use AST to find `setup()` call
2. Extract `install_requires` keyword argument
3. Extract `extras_require` for optional dependencies
4. Handle variables: if `install_requires=REQUIREMENTS`, trace variable definition

### 4.3 Handle Dynamic setup.py
If setup.py reads from files:
```python
with open('requirements.txt') as f:
    requirements = f.read().splitlines()
```
Follow the file reference and parse that file instead.

---

## Step 5: Parse setup.cfg

### 5.1 Structure
INI-style configuration:
```ini
[options]
install_requires =
    requests>=2.25.0
    numpy>=1.20.0

[options.extras_require]
dev =
    pytest
    black
```

### 5.2 Extraction Steps
1. Use `configparser` to read the file
2. Get `[options]` section, key `install_requires`
3. Split by newlines, strip whitespace
4. Optionally parse `[options.extras_require]`

---

## Step 6: Parse pyproject.toml

### 6.1 PEP 621 Format (Standard)
```toml
[project]
dependencies = [
    "requests>=2.25.0",
    "numpy",
]

[project.optional-dependencies]
dev = ["pytest", "black"]
```

### 6.2 Poetry Format
```toml
[tool.poetry.dependencies]
python = "^3.9"
requests = "^2.25.0"
numpy = {version = "^1.20", optional = true}

[tool.poetry.group.dev.dependencies]
pytest = "^7.0"
```

### 6.3 Extraction Steps
1. Use `tomllib` (Python 3.11+) or `tomli` package
2. Check for `project.dependencies` (PEP 621)
3. Check for `tool.poetry.dependencies` (Poetry)
4. Normalize version specifiers to pip format

---

## Step 7: Parse Pipfile

### 7.1 Structure
```toml
[packages]
requests = "*"
numpy = ">=1.20"

[dev-packages]
pytest = "*"
```

### 7.2 Extraction Steps
1. Parse as TOML
2. Extract `[packages]` section
3. Convert version specifiers: `"*"` → no version, `">=1.20"` → `>=1.20`

---

## Step 8: Parse environment.yml (Conda)

### 8.1 Structure
```yaml
dependencies:
  - python=3.9
  - numpy=1.20
  - pip:
      - requests>=2.25.0
```

### 8.2 Extraction Steps
1. Parse YAML
2. Extract `dependencies` list
3. Separate conda packages from pip packages (under `pip:` key)
4. For requirements.txt, prioritize pip packages

---

## Step 9: Filter Standard Library Modules

### 9.1 Get Standard Library List
```python
import sys
import stdlib_list  # or use importlib.util

STDLIB = set(stdlib_list.stdlib_list(sys.version_info[:2]))
```

### 9.2 Common Standard Library Modules to Exclude
```
os, sys, re, json, collections, itertools, functools,
typing, pathlib, datetime, time, math, random, copy,
subprocess, threading, multiprocessing, logging, unittest,
argparse, configparser, csv, io, pickle, sqlite3, http,
urllib, email, html, xml, asyncio, contextlib, dataclasses,
enum, abc, warnings, traceback, inspect, importlib, pkgutil
```

### 9.3 Filter Strategy
1. Start with extracted imports
2. Remove all standard library modules
3. Remove local project modules (same package name as project)

---

## Step 10: Map Import Names to PyPI Package Names

### 10.1 Common Mismatches
| Import Name | PyPI Package |
|-------------|--------------|
| `cv2` | `opencv-python` |
| `PIL` | `Pillow` |
| `sklearn` | `scikit-learn` |
| `yaml` | `PyYAML` |
| `bs4` | `beautifulsoup4` |
| `dateutil` | `python-dateutil` |
| `dotenv` | `python-dotenv` |
| `jwt` | `PyJWT` |
| `serial` | `pyserial` |
| `usb` | `pyusb` |

### 10.2 Resolution Strategy
1. Maintain a mapping dictionary for known mismatches
2. Query PyPI API if uncertain: `https://pypi.org/pypi/{package}/json`
3. Flag unresolved packages for manual review

---

## Step 11: Determine Package Versions

### 11.1 Priority Order
1. **Existing version specs** from setup.py/pyproject.toml/requirements.txt
2. **Installed versions**: `pip show {package}` or `importlib.metadata`
3. **Latest stable version**: Query PyPI API
4. **No version**: Leave unpinned (least preferred)

### 11.2 Version Pinning Strategy
- **Exact pin** (`==1.2.3`): For reproducibility
- **Compatible release** (`~=1.2`): Allows patch updates
- **Minimum version** (`>=1.2.0`): More flexible

---

## Step 12: Generate requirements.txt

### 12.1 Output Format
```
# Auto-generated requirements.txt
# Generated on: YYYY-MM-DD

# Core dependencies
numpy==1.24.0
pandas==2.0.0
requests==2.31.0

# Optional: Development dependencies (uncomment if needed)
# pytest==7.4.0
# black==23.0.0
```

### 12.2 Best Practices
1. Sort alphabetically
2. Include generation timestamp as comment
3. Group by category if helpful
4. Add comments for non-obvious packages
5. Warn about packages that couldn't be resolved

---

## Step 13: Validation

1. **Syntax check**: Ensure valid requirements.txt format
2. **Duplicate check**: No repeated packages
3. **Conflict check**: No conflicting version specifiers
4. **Dry run**: `pip install --dry-run -r requirements.txt`

---

## Output Summary

After completion, report:
- Total packages found: X
- From source imports: X
- From config files: X
- Standard library filtered: X
- Unresolved packages: [list]
- Warnings: [any issues encountered]
