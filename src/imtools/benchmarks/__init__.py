"""Benchmarks subpackage for imtools.

Run all benchmarks from the command line::

    python -m imtools.benchmarks

Public API:
    - :func:`~imtools.benchmarks.benchmark_converters.run_benchmark_scenarios`
"""

from .benchmark_converters import run_benchmark_scenarios

__all__ = [
    "run_benchmark_scenarios",
]