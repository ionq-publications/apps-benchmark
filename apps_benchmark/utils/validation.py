"""
Validation utilities for apps-benchmark.

This module provides validation helpers for backends, benchmarks, and registries.

Copyright (c) 2025 IonQ, Inc. All rights reserved.
Licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0
International (CC BY-NC-ND 4.0) License.
For details, visit: https://creativecommons.org/licenses/by-nc-nd/4.0/
"""

import inspect
from typing import Any, Mapping, Sequence

from apps_benchmark.errors import (
    BackendValidationError,
    BenchmarkValidationError,
)


def validate_backend_interface(backend_class: type[Any]) -> None:
    """
    Validate that a backend class implements AbstractBackend interface.

    Args:
        backend_class: Backend class to validate

    Raises:
        BackendValidationError: If interface is not properly implemented
    """
    # Import here to avoid circular imports
    from apps_benchmark.core.backend import AbstractBackend

    # Check if it's a class
    if not inspect.isclass(backend_class):
        raise BackendValidationError(f"Expected a class, got {type(backend_class).__name__}")

    # Check if it inherits from AbstractBackend
    if not issubclass(backend_class, AbstractBackend):
        raise BackendValidationError(
            f"Backend class must inherit from AbstractBackend. "
            f"Found bases: {[b.__name__ for b in backend_class.__bases__]}"
        )

    # Check if it's not abstract (can be instantiated)
    if inspect.isabstract(backend_class):
        abstract_methods = [
            name
            for name, method in inspect.getmembers(backend_class)
            if getattr(method, "__isabstractmethod__", False)
        ]
        raise BackendValidationError(
            f"Backend class has unimplemented abstract methods: {abstract_methods}"
        )

    # Check required methods exist
    required_methods = ["name", "run"]
    for method_name in required_methods:
        if not hasattr(backend_class, method_name):
            raise BackendValidationError(f"Backend class missing required method: {method_name}()")


def validate_benchmark_interface(runner_class: type[Any]) -> None:
    """
    Validate that a benchmark runner implements AbstractAlgoRunner interface.

    Args:
        runner_class: Runner class to validate

    Raises:
        BenchmarkValidationError: If interface is not properly implemented
    """
    # Import here to avoid circular imports
    from apps_benchmark.core.benchmark import AbstractAlgoRunner

    # Check if it's a class
    if not inspect.isclass(runner_class):
        raise BenchmarkValidationError(f"Expected a class, got {type(runner_class).__name__}")

    # Check if it inherits from AbstractAlgoRunner
    if not issubclass(runner_class, AbstractAlgoRunner):
        raise BenchmarkValidationError(
            f"Runner class must inherit from AbstractAlgoRunner. "
            f"Found bases: {[b.__name__ for b in runner_class.__bases__]}"
        )

    # Check if it's not abstract
    if inspect.isabstract(runner_class):
        abstract_methods = [
            name
            for name, method in inspect.getmembers(runner_class)
            if getattr(method, "__isabstractmethod__", False)
        ]
        raise BenchmarkValidationError(
            f"Runner class has unimplemented abstract methods: {abstract_methods}"
        )

    # Check required methods exist
    required_methods = [
        "name",
        "setup_algo_inputs",
        "execute_benchmark_algo",
        "compute_merit_figures",
    ]
    for method_name in required_methods:
        if not hasattr(runner_class, method_name):
            raise BenchmarkValidationError(f"Runner class missing required method: {method_name}()")


def check_duplicate_backend(backend_name: str, registry: dict) -> None:
    """
    Check if backend name already exists (either builtin or DIY).

    Args:
        backend_name: Name to check
        registry: Current registry data (contains diy_backends)

    Raises:
        BackendValidationError: If name exists
    """
    # Check against builtin backends (discovered on-the-fly)
    from apps_benchmark.core.registry import _discover_builtin_backends

    builtin_backends = _discover_builtin_backends()
    if backend_name in builtin_backends:
        raise BackendValidationError(
            f"Backend '{backend_name}' already exists as a built-in backend.\n"
            f"Please choose a different name for your DIY backend."
        )

    # Check against DIY backends in registry
    if backend_name in registry.get("diy_backends", {}):
        raise BackendValidationError(
            f"Backend '{backend_name}' already registered as a DIY backend.\n"
            f"Please choose a different name or remove the existing backend first."
        )


def check_duplicate_benchmark(benchmark_name: str, category: str, registry: dict) -> None:
    """
    Check if benchmark name already exists (either builtin or DIY) for a category.

    Args:
        benchmark_name: Name to check
        category: Benchmark category
        registry: Current registry data (contains diy_benchmarks)

    Raises:
        BenchmarkValidationError: If name exists
    """
    # Check against builtin benchmarks (discovered on-the-fly)
    from apps_benchmark.core.registry import _discover_builtin_benchmarks

    builtin_benchmarks = _discover_builtin_benchmarks()
    if category in builtin_benchmarks:
        if benchmark_name in builtin_benchmarks[category].get("runners", []):
            raise BenchmarkValidationError(
                f"Benchmark '{benchmark_name}' already exists as a built-in benchmark in category '{category}'.\n"
                f"Please choose a different name for your DIY benchmark."
            )

    # Check against DIY benchmarks in registry
    diy_benchmarks = registry.get("diy_benchmarks", {})
    if category in diy_benchmarks:
        if benchmark_name in diy_benchmarks[category]:
            raise BenchmarkValidationError(
                f"Benchmark '{benchmark_name}' already registered as a DIY benchmark in category '{category}'.\n"
                f"Please choose a different name or remove the existing benchmark first."
            )


def check_duplicate_benchmark_case_ids(
    benchmark_cases: Sequence[Mapping[str, Any]],
    registry: dict[str, Any],
) -> None:
    """
    Ensure benchmark case IDs are unique across the registry.

    Args:
        benchmark_cases: Benchmark case metadata entries
        registry: Current registry data

    Raises:
        BenchmarkValidationError: If any benchmark case UUID is duplicated
    """
    incoming_ids: set[str] = set()
    for benchmark_case in benchmark_cases:
        case_id = benchmark_case.get("uuid")
        if not isinstance(case_id, str) or not case_id.strip():
            raise BenchmarkValidationError(
                f"Benchmark case entry is missing a valid 'uuid': {benchmark_case}"
            )
        if case_id in incoming_ids:
            raise BenchmarkValidationError(
                f"Duplicate benchmark case ID '{case_id}' found in benchmark registration payload."
            )
        incoming_ids.add(case_id)

    existing_ids: set[str] = set()

    for builtin_info in registry.get("builtin_benchmarks", {}).values():
        for benchmark_case in builtin_info.get("benchmark_cases", []):
            case_id = benchmark_case.get("uuid")
            if isinstance(case_id, str) and case_id.strip():
                existing_ids.add(case_id)

    for diy_category in registry.get("diy_benchmarks", {}).values():
        for benchmark_info in diy_category.values():
            for benchmark_case in benchmark_info.get("benchmark_cases", []):
                case_id = benchmark_case.get("uuid")
                if isinstance(case_id, str) and case_id.strip():
                    existing_ids.add(case_id)

    duplicate_ids = sorted(incoming_ids & existing_ids)
    if duplicate_ids:
        duplicates = ", ".join(duplicate_ids)
        raise BenchmarkValidationError(f"Benchmark case ID(s) already registered: {duplicates}")
