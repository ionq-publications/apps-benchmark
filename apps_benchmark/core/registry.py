"""
Registry system for managing backends and benchmarks.

This module handles auto-discovery of built-in components and registration
of user-defined (DIY) components.

Copyright (c) 2025 IonQ, Inc. All rights reserved.
Licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0
International (CC BY-NC-ND 4.0) License.
For details, visit: https://creativecommons.org/licenses/by-nc-nd/4.0/
"""

import importlib
import importlib.util
import inspect
import json
from pathlib import Path
from typing import Any, cast

import pandas as pd

from apps_benchmark.core.backend import AbstractBackend
from apps_benchmark.errors import (
    BackendValidationError,
    BenchmarkValidationError,
    RegistryError,
)
from apps_benchmark.utils.config import get_local_dev_dir_from_config
from apps_benchmark.utils.file_ops import (
    load_registry,
    snake_to_camel,
    write_registry_atomic,
)
from apps_benchmark.utils.validation import (
    check_duplicate_backend,
    check_duplicate_benchmark,
    check_duplicate_benchmark_case_ids,
    validate_backend_interface,
    validate_benchmark_interface,
)


def get_local_dev_dir() -> Path:
    """
    Get the local_dev directory path from configuration.

    The location is configured in ~/config_local_dev.json.
    If the config file doesn't exist, it's created with default location ~/local_dev.

    Returns:
        Path to local_dev directory (from config)
    """
    return get_local_dev_dir_from_config()


def ensure_local_dev_dir() -> Path:
    """
    Create ~/local_dev/ directory structure if it doesn't exist.

    Returns:
        Path to local_dev directory
    """
    local_dev = get_local_dev_dir()
    local_dev.mkdir(exist_ok=True)

    backends_dir = local_dev / "backends"
    backends_dir.mkdir(exist_ok=True)

    benchmarks_dir = local_dev / "benchmarks"
    benchmarks_dir.mkdir(exist_ok=True)

    return local_dev


def initialize_empty_registry(registry_path: Path) -> None:
    """
    Create empty registry file for DIY components only.

    Args:
        registry_path: Path to registry file
    """
    if registry_path.name == "backends.json":
        empty_registry = {"version": "1.0", "diy_backends": {}}
    else:  # benchmarks.json
        empty_registry = {
            "version": "1.0",
            "diy_benchmarks": {},
        }

    write_registry_atomic(registry_path, empty_registry)


def _load_problem_instance_registry_entry(json_file: Path, builtin: bool) -> dict:
    """
    Load problem instance metadata for the benchmark registry.

    Raises:
        RegistryError: If the problem instance is missing a valid instance_id
    """
    with open(json_file) as f:
        data = json.load(f)

    instance_id = data.get("instance_id")
    if not isinstance(instance_id, str) or not instance_id.strip():
        raise RegistryError(f"Problem instance file {json_file} is missing required 'instance_id'.")

    solution_algorithms = data.get("solution_algorithms")
    if not isinstance(solution_algorithms, list):
        raise RegistryError(
            f"Problem instance file {json_file} is missing valid 'solution_algorithms'."
        )

    open_solution_algorithms = data.get("open_solution_algorithms") or []
    if not isinstance(open_solution_algorithms, list) or not all(
        isinstance(algorithm, str) for algorithm in open_solution_algorithms
    ):
        raise RegistryError(
            f"Problem instance file {json_file} has invalid 'open_solution_algorithms'."
        )

    invalid_open_algorithms = sorted(set(open_solution_algorithms) - set(solution_algorithms))
    if invalid_open_algorithms:
        raise RegistryError(
            f"Problem instance file {json_file} marks unknown open_solution_algorithms: "
            f"{invalid_open_algorithms}."
        )

    return {
        "uuid": instance_id,
        "name": data.get("instance_name", json_file.stem),
        "problem_type": data.get("problem_type"),
        "file": str(json_file),
        "builtin": builtin,
        "open_solution_algorithms": open_solution_algorithms,
        "all_solutions_open": bool(solution_algorithms)
        and set(solution_algorithms) == set(open_solution_algorithms),
    }


def _discover_builtin_backends() -> dict[str, dict]:
    """
    Scan apps_benchmark/backends/ and auto-discover backend classes.

    Returns:
        Dict mapping backend_name -> backend_info
    """
    backends = {}

    # Find backends directory in installed package
    import apps_benchmark.backends

    backends_dir = Path(apps_benchmark.backends.__file__).parent

    # Scan for .py files
    for py_file in backends_dir.glob("*.py"):
        if py_file.name.startswith("_"):
            continue  # Skip __init__.py, __pycache__, etc.

        module_name = py_file.stem
        full_module = f"apps_benchmark.backends.{module_name}"

        try:
            # Import module
            module = importlib.import_module(full_module)

            # Find backend classes
            for name, obj in inspect.getmembers(module, inspect.isclass):
                # Check if it's a backend (subclass of AbstractBackend)
                if (
                    issubclass(obj, AbstractBackend)
                    and obj is not AbstractBackend
                    and obj.__module__ == full_module
                ):
                    backend_name = module_name  # Use module name as backend name

                    backends[backend_name] = {
                        "module": full_module,
                        "class": name,
                        "builtin": True,
                        "location": "/backends",
                        "registered_at": pd.Timestamp.now(tz="UTC").isoformat(),
                    }
                    break  # One backend per module

        except Exception as e:
            # Log warning but don't fail
            print(f"Warning: Failed to discover backend in {module_name}: {e}")

    return backends


def _discover_builtin_benchmarks() -> dict[str, dict]:
    """
    Scan apps_benchmark/benchmarks/ and auto-discover benchmarks.

    Returns:
        Dict mapping category -> benchmark_info
    """
    benchmarks = {}

    import apps_benchmark.benchmarks

    benchmarks_dir = Path(apps_benchmark.benchmarks.__file__).parent

    # Scan for category directories
    for category_dir in benchmarks_dir.iterdir():
        if not category_dir.is_dir() or category_dir.name.startswith("_"):
            continue

        category_name = category_dir.name

        # Find algorithm runners
        algorithms_dir = category_dir / "algorithms"
        runners = []
        if algorithms_dir.exists():
            for py_file in sorted(algorithms_dir.glob("*_runner.py")):
                runner_name = py_file.stem.replace("_runner", "")
                runners.append(runner_name)

        # Find problem instances
        benchmark_cases = []
        for json_file in sorted(category_dir.glob("**/benchmark_cases/*.json")):
            try:
                # Load and validate problem instance
                entry = _load_problem_instance_registry_entry(json_file, builtin=True)
                benchmark_cases.append(entry)
            except RegistryError:
                raise
            except Exception as e:
                print(f"Warning: Failed to load {json_file}: {e}")

        benchmarks[category_name] = {
            "location": "/benchmarks",
            "runners": runners,
            "benchmark_cases": benchmark_cases,
        }

    return benchmarks


def initialize_registries() -> None:
    """
    Initialize registries on first run.

    Creates ~/local_dev/ directory and empty registry files for DIY components only.
    Built-in components are discovered on-the-fly and not stored in registries.
    """
    local_dev = ensure_local_dev_dir()

    # Initialize backend registry (DIY only)
    backends_registry_path = local_dev / "backends.json"
    if not backends_registry_path.exists():
        initialize_empty_registry(backends_registry_path)
        print(f"✓ Created DIY backend registry at {backends_registry_path}")

    # Initialize benchmark registry (DIY only)
    benchmarks_registry_path = local_dev / "benchmarks.json"
    if not benchmarks_registry_path.exists():
        initialize_empty_registry(benchmarks_registry_path)
        print(f"✓ Created DIY benchmark registry at {benchmarks_registry_path}")


def register_diy_backend(backend_name: str) -> None:
    """
    Register a DIY backend from ~/local_dev/backends/.

    Args:
        backend_name: Name of backend (e.g., "my_custom_backend")

    Raises:
        BackendValidationError: If backend fails validation
        FileNotFoundError: If backend file not found
    """
    # 1. Construct paths
    local_dev = get_local_dev_dir()
    backend_file = local_dev / "backends" / f"{backend_name}.py"

    if not backend_file.exists():
        raise FileNotFoundError(
            f"Backend file not found: {backend_file}\nPlease create the file before registering."
        )

    # 2. Expected class name (CamelCase conversion)
    expected_class_name = snake_to_camel(backend_name)

    # 3. Import module dynamically
    spec = importlib.util.spec_from_file_location(backend_name, backend_file)
    if spec is None or spec.loader is None:
        raise BackendValidationError(f"Failed to load module from {backend_file}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # 4. Look for class
    backend_class = getattr(module, expected_class_name, None)
    if backend_class is None:
        available_classes = [
            name
            for name, obj in inspect.getmembers(module, inspect.isclass)
            if not name.startswith("_")
        ]
        raise BackendValidationError(
            f"Based on '{backend_name}' expected class '{expected_class_name}' "
            f"but did not find it. Backend not registered.\n\n"
            f"Available classes in {backend_name}.py: {available_classes}\n"
            f"Did you mean to name your class '{expected_class_name}'?"
        )

    # 5. Validate interface
    validate_backend_interface(backend_class)

    # 6. Update registry
    registry_path = local_dev / "backends.json"
    registry = load_registry(registry_path)

    # Check for duplicates
    check_duplicate_backend(backend_name, registry)

    registry["diy_backends"][backend_name] = {
        "module": str(backend_file),
        "class": expected_class_name,
        "builtin": False,
        "location": "/backends",  # DIY backends also go in this location
        "registered_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "validated": True,
    }

    write_registry_atomic(registry_path, registry)

    print(f"✓ Backend '{backend_name}' registered successfully.")


def register_diy_benchmark(benchmark_name: str, category: str) -> None:
    """
    Register a DIY benchmark from ~/local_dev/benchmarks/.

    Args:
        benchmark_name: Name of benchmark runner (e.g., "my_vqe")
        category: Benchmark category (e.g., "chemistry")

    Raises:
        BenchmarkValidationError: If benchmark fails validation
        FileNotFoundError: If runner file not found
    """
    # 1. Construct paths
    local_dev = get_local_dev_dir()
    category_dir = local_dev / "benchmarks" / category
    runner_file = category_dir / "algorithms" / f"{benchmark_name}_runner.py"

    if not runner_file.exists():
        raise FileNotFoundError(
            f"Runner file not found: {runner_file}\nPlease create the file before registering."
        )

    # 2. Expected class name
    expected_class_name = snake_to_camel(benchmark_name) + "Runner"

    # 3. Import runner module
    spec = importlib.util.spec_from_file_location(benchmark_name, runner_file)
    if spec is None or spec.loader is None:
        raise BenchmarkValidationError(f"Failed to load module from {runner_file}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # 4. Look for runner class
    runner_class = getattr(module, expected_class_name, None)
    if runner_class is None:
        available_classes = [
            name
            for name, obj in inspect.getmembers(module, inspect.isclass)
            if not name.startswith("_")
        ]
        raise BenchmarkValidationError(
            f"Based on '{benchmark_name}' expected class '{expected_class_name}' "
            f"but did not find it. Benchmark not registered.\n\n"
            f"Available classes in {benchmark_name}_runner.py: {available_classes}\n"
            f"Did you mean to name your class '{expected_class_name}'?"
        )

    # 5. Validate interface
    validate_benchmark_interface(runner_class)

    # 6. Load problem instances
    instances_dir = category_dir / "benchmark_cases"
    benchmark_cases = []
    if instances_dir.exists():
        for json_file in instances_dir.glob("*.json"):
            try:
                # Load and validate problem instance
                entry = _load_problem_instance_registry_entry(json_file, builtin=False)
                benchmark_cases.append(entry)
            except RegistryError as e:
                raise BenchmarkValidationError(str(e)) from e
            except Exception as e:
                print(f"Warning: Failed to load {json_file}: {e}")

    # 7. Update registry
    registry_path = local_dev / "benchmarks.json"
    registry = load_registry(registry_path)

    # Check for duplicates
    check_duplicate_benchmark(benchmark_name, category, registry)
    check_duplicate_benchmark_case_ids(benchmark_cases, registry)

    # Ensure category exists
    if category not in registry["diy_benchmarks"]:
        registry["diy_benchmarks"][category] = {}

    registry["diy_benchmarks"][category][benchmark_name] = {
        "runner_module": str(runner_file),
        "runner_class": expected_class_name,
        "benchmark_cases": benchmark_cases,
        "registered_at": pd.Timestamp.now(tz="UTC").isoformat(),
    }

    write_registry_atomic(registry_path, registry)

    print(f"✓ Benchmark '{benchmark_name}' registered successfully in category '{category}'.")
    print(f"  Found {len(benchmark_cases)} problem instance(s).")


def list_builtin_backends() -> dict[str, dict]:
    """
    List all built-in backends discovered from installed package.

    Returns:
        Dict mapping backend_name -> backend_info
    """
    return _discover_builtin_backends()


def list_diy_backends() -> dict[str, dict]:
    """
    List all DIY backends from local_dev registry.

    Returns:
        Dict mapping backend_name -> backend_info
    """
    local_dev = get_local_dev_dir()
    registry_path = local_dev / "backends.json"

    if not registry_path.exists():
        return {}

    registry = load_registry(registry_path)
    return cast(dict[str, dict[str, Any]], registry.get("diy_backends", {}))


def list_builtin_benchmarks() -> dict[str, dict]:
    """
    List all built-in benchmarks discovered from installed package.

    Returns:
        Dict mapping category_name -> category_info
    """
    return _discover_builtin_benchmarks()


def list_diy_benchmarks() -> dict[str, dict]:
    """
    List all DIY benchmarks from local_dev registry.

    Returns:
        Dict mapping category_name -> category_info
    """
    local_dev = get_local_dev_dir()
    registry_path = local_dev / "benchmarks.json"

    if not registry_path.exists():
        return {}

    registry = load_registry(registry_path)
    return cast(dict[str, dict[str, Any]], registry.get("diy_benchmarks", {}))
