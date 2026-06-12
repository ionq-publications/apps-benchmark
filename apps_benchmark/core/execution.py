"""
Shared benchmark execution helpers.

This module contains registry-backed loading and selection logic used by both
the CLI and interactive orchestration APIs.
"""

from __future__ import annotations

import importlib
import importlib.util
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, cast

from apps_benchmark.core.backend import AbstractBackend
from apps_benchmark.core.benchmark import AbstractAlgoRunner
from apps_benchmark.core.registry import (
    list_builtin_backends,
    list_builtin_benchmarks,
    list_diy_backends,
    list_diy_benchmarks,
)
from apps_benchmark.errors import (
    BackendError,
    BackendNotFoundError,
    BenchmarkCaseError,
    BenchmarkError,
    BenchmarkValidationError,
    OpenBenchmarkError,
)
from apps_benchmark.primitives.benchmark_case import BenchmarkCase
from apps_benchmark.utils.file_ops import snake_to_camel

DEFAULT_SHOTS_PER_QC: int = 1000


def resolve_shots(
    user_shots: int | None = None,
    config_shots: int | None = None,
    benchmark_shots: int | None = None,
    *,
    reporter: Callable[[str], None] | None = None,
) -> int:
    """
    Resolve shots per circuit by user, config, benchmark, then package default.
    """
    shot_sources = [
        ("user", user_shots),
        ("config", config_shots),
        ("benchmark", benchmark_shots),
        ("default", DEFAULT_SHOTS_PER_QC),
    ]
    for source, shots in shot_sources:
        if shots is None:
            continue
        resolved_shots = int(shots)
        if resolved_shots <= 0:
            raise BenchmarkValidationError(f"Shots must be positive. Got {resolved_shots}.")
        if reporter is not None:
            if source == "default":
                reporter(f"Shots per circuit falls to default: {resolved_shots}")
            elif source == "benchmark":
                reporter(f"Shots per circuit set per benchmark: {resolved_shots}")
            else:
                reporter(f"Shots per circuit set by {source}: {resolved_shots}")
        return resolved_shots

    raise BenchmarkValidationError("Could not resolve a shot count.")


def get_shots_for_case(benchmark_case: BenchmarkCase) -> int | None:
    """Return benchmark-specific recommended shots, if configured."""
    configured_shots = benchmark_case.data.get("recommended_minimum_shots_per_qc")
    if configured_shots is None:
        return None

    shots = int(configured_shots)
    if shots <= 0:
        raise BenchmarkCaseError(
            f"Shots in BenchmarkCase '{benchmark_case.instance_name}' must be positive. "
            f"Got {shots}."
        )
    return shots


def get_available_runner_names(
    category: str,
    builtin_benchmarks: dict[str, dict[str, Any]] | None = None,
    diy_benchmarks: dict[str, dict[str, Any]] | None = None,
) -> set[str]:
    """Return all registered runner names for a category."""
    if builtin_benchmarks is None:
        builtin_benchmarks = list_builtin_benchmarks()
    if diy_benchmarks is None:
        diy_benchmarks = list_diy_benchmarks()

    runner_names: set[str] = set()
    if category in builtin_benchmarks:
        runner_names.update(builtin_benchmarks[category].get("runners", []))
    if category in diy_benchmarks:
        runner_names.update(diy_benchmarks[category].keys())
    return runner_names


def classify_solution_algorithms(
    benchmark_case: BenchmarkCase,
    available_runner_names: set[str],
) -> tuple[list[str], list[str], list[str]]:
    """Partition case algorithms into open, shipped, and unregistered entries."""
    open_algorithms = benchmark_case.open_solution_algorithms or []
    open_algorithm_set = set(open_algorithms)

    shipped_algorithms = [
        algorithm
        for algorithm in benchmark_case.solution_algorithms
        if algorithm not in open_algorithm_set and algorithm in available_runner_names
    ]
    unregistered_algorithms = [
        algorithm
        for algorithm in benchmark_case.solution_algorithms
        if algorithm not in open_algorithm_set and algorithm not in available_runner_names
    ]
    return open_algorithms, shipped_algorithms, unregistered_algorithms


def _raise_for_unregistered_algorithms(
    benchmark_case: BenchmarkCase,
    unregistered_algorithms: list[str],
) -> None:
    if unregistered_algorithms:
        raise BenchmarkError(
            f"Case '{benchmark_case.instance_name}' lists algorithms with no registered runners: "
            f"{', '.join(unregistered_algorithms)}. Mark them in open_solution_algorithms if "
            "they are intentionally bring-your-own-solver benchmarks."
        )


def select_solution_algorithm(
    benchmark_case: BenchmarkCase,
    available_runner_names: set[str],
    requested_algorithm: str | None = None,
) -> str:
    """Choose a runnable algorithm, or raise a registry/open-benchmark error."""
    open_algorithms, shipped_algorithms, unregistered_algorithms = classify_solution_algorithms(
        benchmark_case, available_runner_names
    )

    if requested_algorithm is not None:
        if requested_algorithm not in benchmark_case.solution_algorithms:
            raise BenchmarkError(
                f"algorithm '{requested_algorithm}' not available for this problem. "
                f"Choose from: {benchmark_case.solution_algorithms}."
            )
        _raise_for_unregistered_algorithms(benchmark_case, unregistered_algorithms)
        if requested_algorithm in open_algorithms:
            raise OpenBenchmarkError(
                f"algorithm '{requested_algorithm}' for case '{benchmark_case.instance_name}' "
                "is tagged as an open benchmark solver. apps-benchmark does not ship "
                "this solver; bring your own solver."
            )
        return requested_algorithm

    _raise_for_unregistered_algorithms(benchmark_case, unregistered_algorithms)

    if shipped_algorithms:
        return shipped_algorithms[0]

    if open_algorithms:
        raise OpenBenchmarkError(
            f"Case '{benchmark_case.instance_name}' is tagged as an open benchmark. "
            "apps-benchmark does not ship solvers for: "
            f"{', '.join(open_algorithms)}. Bring your own solver."
        )

    raise BenchmarkError(
        f"Case '{benchmark_case.instance_name}' does not define any runnable solution algorithms."
    )


def _get_ionq_backend_api_key() -> str:
    import os

    api_key = os.getenv("IONQ_API_KEY")
    if not api_key:
        raise BackendError("IONQ_API_KEY environment variable not set")
    return api_key


def load_backend(backend_name: str) -> AbstractBackend:
    """Load a backend instance by registry name."""
    builtin_backends = list_builtin_backends()
    diy_backends = list_diy_backends()

    backend_info = None
    if backend_name in builtin_backends:
        backend_info = builtin_backends[backend_name]
    elif backend_name in diy_backends:
        backend_info = diy_backends[backend_name]
    else:
        all_backends = [*builtin_backends.keys()] + [*diy_backends.keys()]
        available = ", ".join(all_backends)
        raise BackendNotFoundError(
            f"Backend '{backend_name}' not found in registry.\n"
            f"Available backends: {available if available else 'none'}\n"
            f"Use 'apps-benchmark add --backend <name>' to register a DIY backend."
        )

    try:
        if backend_info["builtin"]:
            module = importlib.import_module(str(backend_info["module"]))
        else:
            module_path = Path(str(backend_info["module"]))
            spec = importlib.util.spec_from_file_location(backend_name, module_path)
            if spec is None or spec.loader is None:
                raise BackendError(f"Failed to load module from {module_path}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

        backend_class = cast(type[AbstractBackend], getattr(module, str(backend_info["class"])))
        if backend_name == "ionq_cloud_backend":
            from qiskit_ionq import IonQProvider

            provider = IonQProvider(token=_get_ionq_backend_api_key())
            ionq_sim = provider.get_backend("simulator")
            ionq_sim.set_options(noise_model="forte-1")
            return cast(AbstractBackend, backend_class(ionq_sim, optimization_level=1))

        return cast(AbstractBackend, backend_class())

    except Exception as exc:
        raise BackendError(f"Failed to load backend '{backend_name}': {exc}") from exc


def find_benchmark_case_by_uuid(case_uuid: str) -> tuple[Path, str] | None:
    """Find a benchmark case file and category by instance UUID."""
    for category, entry in iter_benchmark_case_entries():
        if entry.get("uuid") == case_uuid:
            return Path(str(entry["file"])), category
    return None


def _load_diy_runner(
    category: str,
    runner_name: str,
    runner_info: Mapping[str, Any],
) -> AbstractAlgoRunner:
    runner_module_path = Path(str(runner_info["runner_module"]))
    runner_class_name = str(runner_info["runner_class"])

    try:
        module_name = f"local_dev.benchmarks.{category}.algorithms.{runner_name}_runner"
        spec = importlib.util.spec_from_file_location(module_name, runner_module_path)
        if spec is None or spec.loader is None:
            raise BenchmarkError(
                f"Failed to load spec for DIY runner '{runner_name}' from {runner_module_path}"
            )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        runner_class = cast(type[AbstractAlgoRunner], getattr(module, runner_class_name))
        return runner_class()
    except Exception as exc:
        raise BenchmarkError(
            f"Failed to load DIY runner '{runner_name}' from category '{category}': {exc}"
        ) from exc


def _load_builtin_runner(category: str, runner_name: str) -> AbstractAlgoRunner:
    class_name = snake_to_camel(runner_name) + "Runner"

    try:
        module_name = f"apps_benchmark.benchmarks.{category}.algorithms.{runner_name}_runner"
        module = importlib.import_module(module_name)
        runner_class = cast(type[AbstractAlgoRunner], getattr(module, class_name))
        return runner_class()
    except Exception as exc:
        raise BenchmarkError(
            f"Failed to load runner '{runner_name}' from category '{category}': {exc}"
        ) from exc


def load_runner(
    category: str,
    runner_name: str,
) -> AbstractAlgoRunner:
    """Load an algorithm runner from built-in or DIY benchmark registries."""
    builtin = list_builtin_benchmarks()
    if category in builtin and runner_name in builtin[category].get("runners", []):
        return _load_builtin_runner(category, runner_name)

    diy = list_diy_benchmarks()
    if category in diy and runner_name in diy[category]:
        return _load_diy_runner(category, runner_name, diy[category][runner_name])

    raise BenchmarkError(
        f"Runner '{runner_name}' from category '{category}' was not found in the registry."
    )


def iter_benchmark_case_entries(
    *,
    include_builtin: bool = True,
    include_diy: bool = True,
) -> list[tuple[str, dict[str, Any]]]:
    """Return benchmark case registry entries as ``(category, entry)`` pairs."""
    entries: list[tuple[str, dict[str, Any]]] = []

    if include_builtin:
        for category, info in list_builtin_benchmarks().items():
            for case_info in info.get("benchmark_cases", []):
                entries.append((category, case_info))

    if include_diy:
        for category, runners in list_diy_benchmarks().items():
            for runner_info in runners.values():
                for case_info in runner_info.get("benchmark_cases", []):
                    entries.append((category, case_info))

    return entries


def load_benchmark_case_entry(
    entry: Mapping[str, Any],
) -> BenchmarkCase:
    """Load a ``BenchmarkCase`` from a registry entry."""
    case_path = Path(str(entry["file"]))
    try:
        return BenchmarkCase.load_from_database(case_path)
    except Exception as exc:
        raise BenchmarkCaseError(f"Failed to load benchmark case '{case_path}': {exc}") from exc
