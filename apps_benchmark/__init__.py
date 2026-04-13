"""
IonQ Quantum Application Benchmarking Framework.

This package provides a framework for running quantum benchmarks on various
backends and managing custom backends and benchmarks.

Copyright (c) 2025 IonQ, Inc. All rights reserved.
Licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0
International (CC BY-NC-ND 4.0) License.
For details, visit: https://creativecommons.org/licenses/by-nc-nd/4.0/
"""

__version__ = "1.0.0"

# Export commonly used classes
from apps_benchmark.core.backend import AbstractAsyncBackend, AbstractBackend, JobStatus
from apps_benchmark.core.benchmark import AbstractAlgoRunner, BenchmarkSubmissionRecord
from apps_benchmark.errors import (
    AppsBenchmarkError,
    BackendError,
    BenchmarkError,
    ConfigError,
    RegistryError,
)
from apps_benchmark.primitives.benchmark_case import BenchmarkCase

__all__ = [
    "__version__",
    "AbstractBackend",
    "AbstractAsyncBackend",
    "JobStatus",
    "AbstractAlgoRunner",
    "BenchmarkSubmissionRecord",
    "BenchmarkCase",
    "AppsBenchmarkError",
    "BackendError",
    "BenchmarkError",
    "ConfigError",
    "RegistryError",
]
