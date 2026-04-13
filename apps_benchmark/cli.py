"""
Command-line interface for apps-benchmark.

This module provides the Click-based CLI for the apps-benchmark tool.

Copyright (c) 2025 IonQ, Inc. All rights reserved.
Licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0
International (CC BY-NC-ND 4.0) License.
For details, visit: https://creativecommons.org/licenses/by-nc-nd/4.0/
"""

import importlib
import importlib.util
import json
from pathlib import Path
from typing import cast

import click

from apps_benchmark.core.backend import AbstractBackend
from apps_benchmark.core.benchmark import AbstractAlgoRunner, BenchmarkSubmissionRecord
from apps_benchmark.core.registry import (
    initialize_registries,
    list_builtin_backends,
    list_builtin_benchmarks,
    list_diy_backends,
    list_diy_benchmarks,
    register_diy_backend,
    register_diy_benchmark,
)
from apps_benchmark.errors import (
    BackendError,
    BackendNotFoundError,
    BenchmarkError,
    ConfigNotFoundError,
    ConfigValidationError,
)
from apps_benchmark.primitives.benchmark_case import BenchmarkCase
from apps_benchmark.utils.cli_config import load_cli_config, save_cli_config
from apps_benchmark.utils.config import get_config_file_path, get_local_dev_dir_from_config

_DEFAULT_SHOTS_PER_QC = 1000

def _resolve_shots(
        cli_set_shots: int | None,
        config_set_shots: int | None,
        benchmark_set_shots: int | None,
    ) -> int:
    """"
    Shot count gets a little weird. Priority order for shot count is:
        1. User-specified CLI shots (if provided)
        2. Config file shots (if provided)
        3. Benchmark-specific recommended shots (if provided in benchmark metadata)
        4. Default shots (1000)
    """
    if cli_set_shots is not None:
        click.echo(f"Shots per circuit set by user: {cli_set_shots}")
        return cli_set_shots #CLI user set value overrides all else
    elif config_set_shots is not None:
        click.echo(f"Shots per circuit set by config: {config_set_shots}")
        return config_set_shots  # config create from a past CLI run , 2nd priority
    elif benchmark_set_shots is not None:
        click.echo(f"Shots per circuit set per benchmark: {benchmark_set_shots}")
        return benchmark_set_shots #benchmark-specific recommended shots, 3rd priority
    click.echo(f"Shots per circuit falls to default: {_DEFAULT_SHOTS_PER_QC}")
    return _DEFAULT_SHOTS_PER_QC #default shots, lowest priority

def _get_shots_for_case(bm_case: BenchmarkCase) -> int | None:
    """ get shots for a benchmark. None, or an integer >= 1. """
    configured_shots = bm_case.data.get("recommended_minimum_shots_per_qc")
    if configured_shots is not None:
        config_shots = int(configured_shots)
        if config_shots <= 0:
            raise ValueError(f"Error: shots in BenchmarkCase '{bm_case.instance_name}' must be a positive integer. "
                             f"Got {config_shots}")
        return config_shots
    return None


# hacky but unblocks timeline
def _get_ionq_backend_api_key():
    """
    Get IonQ API key from environment variable.

    Returns:
        API key string
    """
    import os

    API_KEY = os.getenv("IONQ_API_KEY")
    if not API_KEY:
        raise BackendError("IONQ_API_KEY environment variable not set")
    return API_KEY


def _load_backend(backend_name: str) -> AbstractBackend:
    """
    Load a backend by name from the registry.

    Args:
        backend_name: Name of backend to load

    Returns:
        Backend instance

    Raises:
        BackendNotFoundError: If backend not found in registry
        BackendError: If backend fails to load
    """
    # Get backend registries
    builtin_backends = list_builtin_backends()
    diy_backends = list_diy_backends()

    # Search in both builtin and DIY backends
    backend_info = None
    if backend_name in builtin_backends:
        backend_info = builtin_backends[backend_name]
    elif backend_name in diy_backends:
        backend_info = diy_backends[backend_name]
    else:
        # Backend not found
        all_backends = [*builtin_backends.keys()] + [*diy_backends.keys()]
        # ^ avoiding a weird python dict_keys unpacking bug -fm
        available = ", ".join(all_backends)
        raise BackendNotFoundError(
            f"Backend '{backend_name}' not found in registry.\n"
            f"Available backends: {available if available else 'none'}\n"
            f"Use 'apps-benchmark add --backend <name>' to register a DIY backend."
        )

    # Load backend module
    try:
        if backend_info["builtin"]:
            # Built-in backend - import from apps_benchmark.backends
            module_name = str(backend_info["module"])
            module = importlib.import_module(module_name)
        else:
            # DIY backend - load from file path
            module_path = Path(str(backend_info["module"]))
            spec = importlib.util.spec_from_file_location(backend_name, module_path)
            if spec is None or spec.loader is None:
                raise BackendError(f"Failed to load module from {module_path}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

        # Get backend class
        backend_class = getattr(module, backend_info["class"])
        if backend_name == "ionq_cloud_backend":
            ## hacky way to do this, fix later -fm
            ##   Pattern in the old repo (legacy):
            # Step 1: Create provider and get backend
            from qiskit_ionq import IonQProvider

            API_KEY = _get_ionq_backend_api_key()
            ionq_provider = IonQProvider(token=API_KEY)
            ionq_sim = ionq_provider.get_backend("simulator")

            # Optional: set noise model, for this release forte-1 only
            ionq_sim.set_options(noise_model="forte-1")

            # Step 2: Pass the backend to IonQCloudBackend
            # backend_instance = IonQCloudBackend(target=ionq_sim, optimization_level=1)

            backend_instance = backend_class(ionq_sim, optimization_level=1)
        else:
            # Instantiate backend
            backend_class = cast(type[AbstractBackend], getattr(module, str(backend_info["class"])))
            backend_instance = backend_class()

        return backend_instance

    except Exception as exc:
        raise BackendError(f"Failed to load backend '{backend_name}': {exc}") from exc


def _find_benchmark_case_by_uuid(uuid: str) -> tuple[Path, str, str] | None:
    """
    Find benchmark case file by UUID in built-in and DIY benchmarks.

    Args:
        uuid: Benchmark case UUID to find

    Returns:
        Tuple of (problem_path, category, runner_name) or None if not found
    """
    import json
    from pathlib import Path

    # Search built-in benchmarks
    benchmarks_dir = Path(__file__).parent / "benchmarks"

    for category_dir in benchmarks_dir.iterdir():
        if not category_dir.is_dir() or category_dir.name.startswith("_"):
            continue

        instances_dir = category_dir / "benchmark_cases"
        if not instances_dir.exists():
            continue

        for json_file in instances_dir.glob("*.json"):
            try:
                with open(json_file) as f:
                    data = json.load(f)

                if data.get("instance_id") == uuid:
                    # Found it! Determine runner name from solution_algorithms
                    solution_algos = data.get("solution_algorithms")
                    if isinstance(solution_algos, list) and solution_algos:
                        runner_name = solution_algos[0]
                        if isinstance(runner_name, str):
                            return (json_file, category_dir.name, runner_name)
            except Exception:
                continue

    # Search DIY benchmarks
    diy = list_diy_benchmarks()
    for category, runners in diy.items():
        for runner_name, runner_info in runners.items():
            for case_info in runner_info.get("benchmark_cases", []):
                if case_info.get("uuid") == uuid:
                    case_file = Path(case_info["file"])
                    return (case_file, category, runner_name)

    return None


def _load_diy_runner(category: str, runner_name: str, runner_info: dict) -> AbstractAlgoRunner:
    """
    Load a DIY runner from local_dev.

    Args:
        category: Benchmark category (e.g., "chemistry-x")
        runner_name: Runner name (e.g., "fqe_puccd")
        runner_info: Runner metadata from DIY registry

    Returns:
        Instantiated runner

    Raises:
        BenchmarkError: If runner cannot be loaded
    """
    import importlib

    runner_module_path = Path(runner_info["runner_module"])
    runner_class_name = runner_info["runner_class"]

    try:
        # Use a module path that matches the expected pattern for benchmark_category extraction
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
    """
    Load a built-in runner from apps_benchmark.benchmarks.

    Args:
        category: Benchmark category (e.g., "chemistry")
        runner_name: Runner name (e.g., "vqe_puccd")

    Returns:
        Instantiated runner

    Raises:
        BenchmarkError: If runner cannot be loaded
    """
    import importlib

    from apps_benchmark.utils.file_ops import snake_to_camel

    # Convert runner name to class name
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


def _load_runner(category: str, runner_name: str) -> AbstractAlgoRunner:
    """
    Load algorithm runner by category and name from built-in or DIY benchmarks.

    Tries to load from built-in benchmarks first, then falls back to DIY benchmarks.

    Args:
        category: Benchmark category (e.g., "chemistry")
        runner_name: Runner name (e.g., "vqe_puccd")

    Returns:
        Instantiated runner

    Raises:
        BenchmarkError: If runner not found in either built-in or DIY
    """
    # First, try to load from built-in benchmarks
    try:
        return _load_builtin_runner(category, runner_name)
    except BenchmarkError:
        # If built-in fails, check if this is a DIY benchmark
        diy = list_diy_benchmarks()
        if category in diy and runner_name in diy[category]:
            runner_info = diy[category][runner_name]
            return _load_diy_runner(category, runner_name, runner_info)

        # If neither built-in nor DIY found, raise the original error
        raise


def _run_single_benchmark(backend: AbstractBackend, uuid: str, cli_shots: int | None, config_shots: int | None, algorithm: str | None = None) -> None:
    """
    Run a single benchmark by UUID.

    Args:
        backend: Backend instance
        uuid: Problem instance UUID
        cli_shots: CLI-specified shots
        config_shots: Config-specified shots
        algorithm: Solution algorithm to use. If specified, must be in the benchmark's
                   solution_algorithms list. If None, uses the first algorithm in the list.
                   Use this to select alternative solution methods for benchmarks that
                   support multiple approaches (e.g., 'qft_lcu' instead of 'qft').
    """
    click.echo(f"\nSearching for problem instance '{uuid}'...")

    # Find benchmark case
    result = _find_benchmark_case_by_uuid(uuid)
    if result is None:
        click.echo(f"Error: Benchmark case '{uuid}' not found", err=True)
        raise SystemExit(1)

    problem_path, category, default_runner_name = result
    runner_name = algorithm if algorithm else default_runner_name
    click.echo(f"✓ Found: {problem_path.name} (category: {category})")

    # Load problem instance
    try:
        problem = BenchmarkCase.load_from_database(problem_path)
        click.echo(f"✓ Loaded problem: {problem.instance_name}")
        click.echo(f"  Type: {problem.problem_type}")
        click.echo(f"  Qubits: {problem.num_qubits}")

        # Show available algorithms and which one will be used
        if len(problem.solution_algorithms) > 1:
            click.echo(f"  Available algorithms: {', '.join(problem.solution_algorithms)}")
            click.echo(f"  Using algorithm: {runner_name}")
            if not algorithm:
                click.echo("  ℹ  Use --algorithm to select a different solution algorithm")
        else:
            click.echo(f"  Algorithm: {problem.solution_algorithms[0]}")
    except Exception as exc:
        click.echo(f"Error loading problem instance: {exc}", err=True)
        raise SystemExit(1) from exc

    if algorithm and algorithm not in problem.solution_algorithms:
        click.echo(f"Error: algorithm '{algorithm}' not available for this problem. Choose from: {problem.solution_algorithms}", err=True)
        raise SystemExit(1)

    # Load runner
    try:
        runner = _load_runner(category, runner_name)
        click.echo(f"✓ Loaded runner: {runner.name()}")
    except Exception as exc:
        click.echo(f"Error loading runner: {exc}", err=True)
        raise SystemExit(1) from exc

    # Run benchmark
    try:
        # do weird shot fallback to resolution logic
        benchmark_shots = _get_shots_for_case(problem)
        resolved_shots = _resolve_shots(cli_shots, config_shots, benchmark_shots)

        click.echo(f"\nRunning benchmark on backend '{backend.name()}'...")
        click.echo(f"  Shots per circuit: {resolved_shots}")
        click.echo("")

        record = runner.run_benchmark(problem, backend, shots=resolved_shots)

        # Display results
        click.echo("\n" + "=" * 60)
        click.echo("BENCHMARK RESULTS")
        click.echo("=" * 60)
        click.echo(f"Status: {record.status}")
        click.echo(f"Score: {record.score:.6f}")
        click.echo(f"Total shots: {record.total_shots:,}")
        click.echo(f"Execution time: {record.time_to_soln.total_seconds():.2f}s")
        click.echo("\nProblem-specific data:")
        for key, value in record.problem_specific_data.items():
            if isinstance(value, float):
                click.echo(f"  {key}: {value:.8f}")
            elif type(value).__name__ in ["list", "dict"]:
                click.echo(f"  {key}: [complex data structure]")
            else:
                click.echo(f"  {key}: {value}")
        click.echo("=" * 60)

    except Exception as exc:
        click.echo(f"\nError running benchmark: {exc}", err=True)
        import traceback

        traceback.print_exc()
        raise SystemExit(1) from exc


def _display_category_results(results: list, category: str, backend_name: str, shots: int) -> None:
    """
    Display aggregated results from category execution.

    Args:
        results: List of BenchmarkSubmissionRecord objects
        category: Benchmark category name
        backend_name: Backend name
        shots: Shots per circuit
    """
    if not results:
        return

    # Calculate summary statistics
    total_runs = len(results)
    completed = sum(1 for r in results if r.status == "done")
    errored = total_runs - completed
    avg_score = sum(r.score for r in results) / total_runs if total_runs > 0 else 0.0
    total_time = sum(r.time_to_soln.total_seconds() for r in results)
    total_shots = sum(r.total_shots for r in results)

    # Display summary
    click.echo("\n" + "=" * 60)
    click.echo("RESULTS SUMMARY")
    click.echo("=" * 60)
    click.echo(f"Category:        {category}")
    click.echo(f"Backend:         {backend_name}")
    click.echo(f"Total runs:      {total_runs}")
    click.echo(f"Completed:       {completed}")
    click.echo(f"Errored:         {errored}")
    click.echo(f"Average score:   {avg_score:.6f}")
    click.echo(f"Total time:      {total_time:.2f}s")
    click.echo(f"Total shots:     {total_shots:,}")

    # Display detailed results table
    click.echo("\n" + "-" * 60)
    click.echo("DETAILED RESULTS")
    click.echo("-" * 60)
    click.echo(f"{'Problem':<25} {'Algorithm':<15} {'Score':<10} {'Time (s)':<10}")
    click.echo("-" * 60)

    #list them alphabetically by problem name( which should lead by count, by convention)
    for result in sorted(results, key=lambda r: r.instance_name):
        problem_name = result.instance_name[:24]  # Truncate if too long
        algo_name = result.solution_algorithm[:14]
        score_str = f"{result.score:.6f}"
        time_str = f"{result.time_to_soln.total_seconds():.2f}"

        click.echo(f"{problem_name:<25} {algo_name:<15} {score_str:<10} {time_str:<10}")

    click.echo("=" * 60)


def _run_category_benchmarks(
    backend: AbstractBackend,
    category: str,
    qbit_max: int | None,
    cli_shots: int | None,
    config_shots: int | None,
    algorithm: str | None = None,
) -> None:
    """
    Run all benchmarks in a category.

    Args:
        backend: Backend instance
        category: Benchmark category
        qbit_max: Optional maximum qubits filter
        cli_shots: CLI-specified shots
        config_shots: Config-specified shots
        algorithm: Solution algorithm to use. If specified, only runs benchmarks that
                   support this algorithm (skips others). If None, uses the first
                   algorithm in each benchmark's solution_algorithms list. This allows
                   running a specific solution method across multiple benchmark cases.
    """
    click.echo(f"\nRunning benchmarks in category '{category}'...")
    click.echo(f"  Backend: {backend.name()}")
    click.echo(f"  Max qubits: {qbit_max if qbit_max is not None else 'all'}")

    # Get all benchmarks in category (both built-in and DIY)
    builtin = list_builtin_benchmarks()
    diy = list_diy_benchmarks()

    # Check if category exists in either built-in or DIY
    if category not in builtin and category not in diy:
        click.echo(f"\nError: Category '{category}' not found", err=True)
        all_categories = set(builtin.keys()) | set(diy.keys())
        available = ", ".join(sorted(all_categories))
        click.echo(f"Available categories: {available if available else 'none'}", err=True)
        raise SystemExit(1)

    # Collect benchmark cases from the category
    benchmark_cases = []

    # Add built-in benchmark cases if category exists in built-in
    if category in builtin:
        category_info = builtin[category]
        benchmark_cases.extend(category_info.get("benchmark_cases", []))

    # Add DIY benchmark cases if category exists in DIY
    if category in diy:
        for _runner_name, runner_info in diy[category].items():
            benchmark_cases.extend(runner_info.get("benchmark_cases", []))

    if not benchmark_cases:
        click.echo(f"\nNo benchmark cases found in category '{category}'", err=True)
        raise SystemExit(1)

    # Filter by qbit_max when explicitly requested
    filtered_cases = []
    for case_info in benchmark_cases:
        case_path = Path(case_info["file"])
        try:
            with open(case_path) as f:
                data = json.load(f)
            num_qubits = data.get("num_qubits", 0)
            if qbit_max is None or num_qubits <= qbit_max:
                filtered_cases.append(case_info)
        except Exception as e:
            click.echo(f"Warning: Failed to load case {case_path.name}")
            click.echo(f"Case is expected in case_config {category}: {e}", err=True)
            click.echo(f"Skipping case {case_path.name}", err=True)
            continue

    if not filtered_cases:
        if qbit_max is None:
            click.echo(f"\nNo benchmark cases found in category '{category}'", err=True)
        else:
            click.echo(f"\nNo benchmark cases found with num_qubits <= {qbit_max}", err=True)
        raise SystemExit(1)

    filtered_out_count = len(benchmark_cases) - len(filtered_cases)
    if qbit_max is not None:
        click.echo(
            f"Found {filtered_out_count} benchmark(s) not to run (filtered by --qbit-max={qbit_max})"
        )
    click.echo(f"\nFound {len(filtered_cases)} benchmark(s) to run:")
    for case_info in filtered_cases:
        click.echo(f"  - {case_info['name']} (UUID: {case_info['uuid']})")
    click.echo("")

    # Run all benchmarks
    results: list[BenchmarkSubmissionRecord] = []
    total = len(filtered_cases)

    for idx, case_info in enumerate(filtered_cases, 1):
        case_path = Path(case_info["file"])
        case_name = case_info["name"]

        click.echo(f"[{idx}/{total}] Running {case_name}...", nl=False)

        try:
            # Load problem instance
            problem = BenchmarkCase.load_from_database(case_path)

            runner_name = algorithm if algorithm else problem.solution_algorithms[0]
            if algorithm and algorithm not in problem.solution_algorithms:
                click.echo(f" skipped (algorithm '{algorithm}' not in {problem.solution_algorithms})")
                continue

            # Load runner
            runner = _load_runner(category, runner_name)

            # Run benchmark
            import time

            start = time.time()
            # do weird shot fallback to resolution logic
            benchmark_shots = _get_shots_for_case(problem)
            resolved_shots = _resolve_shots(cli_shots, config_shots, benchmark_shots)

            record = runner.run_benchmark(problem, backend, shots=resolved_shots)
            elapsed = time.time() - start

            results.append(record)

            click.echo(f" done ({elapsed:.1f}s, score={record.score:.6f})")

        except Exception as exc:
            click.echo(" FAILED SINGLE BENCHMARK, CONTINUING...", err=True)
            click.echo(f"\nError running single benchmark case '{case_name}': {exc}", err=True)
            import traceback

            traceback.print_exc()
            raise SystemExit(1) from exc #should we continue ? TBD by SME

    # Display results summary
    _display_category_results(results, category, backend.name(), resolved_shots)


@click.group()
@click.version_option(version="1.0.0", prog_name="apps-benchmark")
def main() -> None:
    """
    IonQ Quantum Application Benchmarking Framework.

    Run benchmarks on various quantum backends, manage backends and benchmarks,
    and save/load configurations for repeated use.
    """
    # Initialize registries on first run
    try:
        initialize_registries()
    except Exception:
        # Silently ignore errors during initialization
        # (registries might already exist)
        pass


@main.command()
@click.option(
    "--backend",
    type=str,
    help="Backend to run benchmarks on (e.g., 'qiskit', 'ionq.forte')",
)
@click.option(
    "--qbit-max",
    type=click.IntRange(min=1),
    default=None,
    help=(
        "Maximum number of qubits to use. Category runs include all discovered cases unless this "
        "filter is provided. Note case-uuid takes precedence over qbit-max"
    ),
)
@click.option(
    "--category",
    type=str,
    default=None,
    help="Run all benchmarks in a category (e.g., 'chemistry', 'qft')",
)
@click.option(
    "--case-uuid",
    type=str,
    default=None,
    help="Run a specific benchmark by UUID. Note case-uuid takes precedence over qbit-max",
)
@click.option(
    "--self-test",
    is_flag=True,
    help="Test backend connectivity (with --qbit-max, runs minimal circuit test)",
)
@click.option(
    "--shots",
    "cli_shots",  # use a different name to avoid confusion with other ways shots is set
    type=click.IntRange(min=1),
    default=None,
    help="User-specified number of shots per circuit (default: config-specific, benchmark-specific, else 1000)",
)
@click.option(
    "--save-config",
    type=str,
    default=None,
    help="Save current flags as a named config to apps-benchmark-config-<NAME>.json",
)
@click.option(
    "--load-config",
    type=str,
    default=None,
    help="Load flags from saved config file apps-benchmark-config-<NAME>.json",
)
@click.option(
    "--algorithm",
    type=str,
    default=None,
    help="Solution algorithm to use when multiple are available (e.g., 'qft_lcu', 'lr_qaoa'). "
         "Use this to select alternative algorithms for benchmark cases that support multiple solution methods. "
         "Defaults to the first algorithm listed in the benchmark case.",
)
@click.pass_context
def run(
    ctx: click.Context,
    backend: str,
    qbit_max: int | None,
    category: str | None,
    case_uuid: str | None,
    self_test: bool,
    cli_shots: int | None,
    save_config: str | None,
    load_config: str | None,
    algorithm: str | None,
) -> None:
    """
    Run benchmarks or test backend connectivity.

    Examples:

    \b
    # Test backend connectivity
    apps-benchmark run --self-test --backend=qiskit_aer_sim_backend

    \b
    # Run all benchmarks in a category
    apps-benchmark run --backend=ionq.forte --category=chemistry

    \b
    # Restrict a category run to smaller cases
    apps-benchmark run --backend=ionq.forte --category=chemistry --qbit-max=20

    \b
    # Run a specific benchmark
    apps-benchmark run --backend=qiskit --case-uuid=610cfb55

    \b
    # Run a specific benchmark with a specific solution algorithm
    apps-benchmark run --backend=qiskit --case-uuid=610cfb55 --algorithm=qft_lcu

    \b
    # Save configuration
    apps-benchmark run --backend=ionq.forte --qbit-max=11 --shots=1000 --save-config=production

    \b
    # Load and use configuration
    apps-benchmark run --load-config=production --category=chemistry
    """
    config_shots = None  # Placeholder for shots loaded from config (if any)

    if cli_shots is not None and cli_shots <= 0:
        click.echo("Error: --shots must be a positive integer", err=True)
        raise SystemExit(1)

    # Load configuration if requested
    if load_config:
        try:
            click.echo(f"Loading configuration '{load_config}'")
            config_data = load_cli_config(load_config)

            # Apply config values for parameters not explicitly provided (still None)
            if backend is None and "backend" in config_data:
                backend = config_data["backend"]
            if qbit_max is None and "qbit_max" in config_data:
                qbit_max = config_data["qbit_max"]
            if category is None and "category" in config_data:
                category = config_data["category"]
            if case_uuid is None and "case_uuid" in config_data:
                case_uuid = config_data["case_uuid"]
            if "shots" in config_data:
                config_shots = config_data["shots"]
            if algorithm is None and "algorithm" in config_data:
                algorithm = config_data["algorithm"]
        except ConfigNotFoundError as exc:
            click.echo(f"Error: {exc}", err=True)
            raise SystemExit(1) from exc
        except ConfigValidationError as exc:
            click.echo(f"Error: {exc}", err=True)
            raise SystemExit(1) from exc

    # Track whether qbit_max was explicitly provided by CLI or loaded config
    qbit_max_explicitly_set = qbit_max is not None

    # Handle save-config if requested
    if save_config:
        try:
            config_data = {
                "version": "1.0",
                "backend": backend,
                "qbit_max": qbit_max,
                "shots": cli_shots, #Only store explicit CLI shot count, not a derived or fallback count
                "category": category,
                "case_uuid": case_uuid,
                "algorithm": algorithm,
            }
            save_cli_config(save_config, config_data)
            click.echo(f"Configuration saved as '{save_config}'")
        except ConfigValidationError as exc:
            click.echo(f"Error saving configuration: {exc}", err=True)
            raise SystemExit(1) from exc

    # Handle self-test mode
    if self_test:
        if not backend:
            click.echo("Error: --backend is required for --self-test", err=True)
            raise SystemExit(1)

        try:
            backend_instance = _load_backend(backend)
            click.echo(f"Testing backend '{backend}'...")

            # Validate connection
            is_valid = backend_instance.validate_connection()

            if is_valid:
                click.echo(f"✓ Backend '{backend}' is available and ready")

                # If qbit_max provided, run a minimal circuit test
                if qbit_max_explicitly_set:
                    from qiskit import QuantumCircuit

                    # Create simple test circuit
                    num_qubits = min(qbit_max, 2)  # Use at most 2 qubits for test
                    qc = QuantumCircuit(num_qubits)
                    qc.h(0)
                    if num_qubits > 1:
                        qc.cx(0, 1)

                    click.echo(
                        f"  Running test circuit ({num_qubits} qubit{'s' if num_qubits > 1 else ''})..."
                    )
                    # For self-test, use CLI shots or config shots or default
                    test_shots = _resolve_shots(cli_shots, config_shots, None)
                    results, job_id, job_data = backend_instance.run([qc], shots=test_shots)
                    click.echo("  ✓ Test circuit executed successfully")
                    click.echo(f"  Job ID: {job_id}")
                    click.echo(f"  Result: {results[0]}")

                return
            else:
                click.echo(f"✗ Backend '{backend}' validation failed", err=True)
                raise SystemExit(1)

        except BackendNotFoundError as exc:
            click.echo(f"Error: {exc}", err=True)
            raise SystemExit(1) from exc
        except BackendError as exc:
            click.echo(f"Backend error: {exc}", err=True)
            raise SystemExit(1) from exc
        except Exception as exc:
            click.echo(f"Unexpected error: {exc}", err=True)
            raise SystemExit(1) from exc

    # Validate required parameters for benchmark execution
    if not backend:
        click.echo("Error: --backend is required", err=True)
        raise SystemExit(1)

    if not case_uuid and not category:
        click.echo("Error: Either --case-uuid or --category is required", err=True)
        raise SystemExit(1)

    # Load backend
    try:
        backend_instance = _load_backend(backend)
    except BackendNotFoundError as exc:
        click.echo(f"Error: {exc}", err=True)
        raise SystemExit(1) from exc
    except Exception as exc:
        click.echo(f"Error loading backend: {exc}", err=True)
        raise SystemExit(1) from exc

    # Execute benchmark by UUID
    if case_uuid:
        _run_single_benchmark(backend_instance, case_uuid, cli_shots, config_shots, algorithm)
    elif category:
        _run_category_benchmarks(backend_instance, category, qbit_max, cli_shots, config_shots, algorithm)
    # # TODO: Implement full benchmark run logic
    # if category:
    #     click.echo(f"Running benchmarks in category '{category}'")
    # elif case_uuid:
    #     click.echo(f"Running benchmark with UUID '{case_uuid}'")

    # click.echo("Run command - implementation pending")
    # click.echo(f"Backend: {backend}")
    # click.echo(f"Qbit max: {qbit_max}")
    # click.echo(f"Category: {category}")
    # click.echo(f"Case UUID: {case_uuid}")
    # click.echo(f"Shots: {shots}")
    # click.echo(f"Save config: {save_config}")
    # click.echo(f"Load config: {load_config}")


@main.command(name="list")
@click.option(
    "--backends",
    is_flag=True,
    help="List all available backends",
)
@click.option(
    "--category",
    type=str,
    help="List benchmarks in a specific category only",
)
def list_resources(backends: bool, category: str | None) -> None:
    """
    List available backends and benchmarks.

    Examples:

    \b
    # List all benchmarks by category (default)
    apps-benchmark list

    \b
    # List all available backends
    apps-benchmark list --backends

    \b
    # List benchmarks in a single category
    apps-benchmark list --category=chemistry
    """
    if backends:
        # List backends
        builtin_backends = list_builtin_backends()
        diy_backends = list_diy_backends()

        if not builtin_backends and not diy_backends:
            click.echo("No backends registered.")
            return

        click.echo("Available backends:")

        # List builtin backends
        for backend_name in sorted(builtin_backends.keys()):
            click.echo(f"  - {backend_name} (built-in)")

        # List DIY backends
        for backend_name in sorted(diy_backends.keys()):
            click.echo(f"  - {backend_name} (DIY)")
    else:
        # List benchmarks
        builtin = list_builtin_benchmarks()
        diy = list_diy_benchmarks()

        if not builtin and not diy:
            click.echo("No benchmarks registered.")
            return

        if category:
            # List specific category
            click.echo(f"Benchmarks in category '{category}':")
            if category in builtin:
                click.echo("  Built-in:")
                info = builtin[category]
                for runner in info.get("runners", []):
                    click.echo(f"    - {runner}")
            if category in diy:
                click.echo("  DIY:")
                for benchmark_name in diy[category].keys():
                    click.echo(f"    - {benchmark_name}")
        else:
            # List all categories
            click.echo("Available benchmark categories:")
            all_categories = set(builtin.keys()) | set(diy.keys())
            for cat in sorted(all_categories):
                click.echo(f"  - {cat}")
                if cat in builtin:
                    info = builtin[cat]
                    click.echo(f"      Built-in runners: {len(info.get('runners', []))}")
                    click.echo(
                        f"      Built-in problem instances: {len(info.get('benchmark_cases', []))}"
                    )
                if cat in diy:
                    click.echo(f"      DIY benchmarks: {len(diy[cat])}")


@main.command(name="add")
@click.option(
    "--backend",
    type=str,
    help="Register a DIY backend from ~/local_dev/backends/",
)
@click.option(
    "--benchmark",
    type=str,
    help="Register a DIY benchmark (requires --category)",
)
@click.option(
    "--category",
    type=str,
    help="Category for DIY benchmark (e.g., 'chemistry', 'optimization')",
)
def add_resource(backend: str | None, benchmark: str | None, category: str | None) -> None:
    """
    Add a DIY backend or benchmark.

    Examples:

    \b
    # Add a backend
    apps-benchmark add --backend my_custom_backend

    \b
    # Add a benchmark
    apps-benchmark add --benchmark my_vqe --category=chemistry
    """
    if backend:
        # Register backend
        try:
            register_diy_backend(backend)
        except Exception as exc:
            click.echo(f"Error registering backend: {exc}", err=True)
            raise SystemExit(1) from exc

    elif benchmark:
        # Register benchmark
        if not category:
            click.echo("Error: --category is required when adding a benchmark", err=True)
            raise SystemExit(1)

        try:
            register_diy_benchmark(benchmark, category)
        except Exception as exc:
            click.echo(f"Error registering benchmark: {exc}", err=True)
            raise SystemExit(1) from exc

    else:
        click.echo("Error: Must specify either --backend or --benchmark", err=True)
        raise SystemExit(1)


@main.command(name="local-dev")
def local_dev() -> None:
    """
    Show local development configuration.

    Displays the configuration file location, local_dev directory path,
    and registry file locations.

    Example:

    \b
    # Show local dev configuration
    apps-benchmark local-dev
    """
    try:
        # Get configuration paths
        config_file = get_config_file_path()
        local_dev_dir = get_local_dev_dir_from_config()

        # Check if config file exists
        config_exists = config_file.exists()
        local_dev_exists = local_dev_dir.exists()

        # Registry paths
        backends_registry = local_dev_dir / "backends.json"
        benchmarks_registry = local_dev_dir / "benchmarks.json"
        backends_dir = local_dev_dir / "backends"
        benchmarks_dir = local_dev_dir / "benchmarks"

        # Display configuration
        click.echo("Local Development Configuration")
        click.echo("=" * 60)
        click.echo(f"\nConfig file:          {config_file}")
        click.echo(f"  Status:             {'✓ exists' if config_exists else '✗ not found'}")

        click.echo(f"\nLocal dev directory:  {local_dev_dir}")
        click.echo(f"  Status:             {'✓ exists' if local_dev_exists else '✗ not found'}")

        if local_dev_exists:
            click.echo(f"\n  Backends dir:       {backends_dir}")
            click.echo(
                f"    Status:           {'✓ exists' if backends_dir.exists() else '✗ not found'}"
            )

            click.echo(f"\n  Benchmarks dir:     {benchmarks_dir}")
            click.echo(
                f"    Status:           {'✓ exists' if benchmarks_dir.exists() else '✗ not found'}"
            )

            click.echo(f"\n  Backend registry:   {backends_registry}")
            click.echo(
                f"    Status:           {'✓ exists' if backends_registry.exists() else '✗ not found'}"
            )

            click.echo(f"\n  Benchmark registry: {benchmarks_registry}")
            click.echo(
                f"    Status:           {'✓ exists' if benchmarks_registry.exists() else '✗ not found'}"
            )

            # Count DIY components
            diy_backends = list_diy_backends()
            diy_benchmarks = list_diy_benchmarks()

            click.echo("\nRegistered DIY components:")
            click.echo(f"  Backends:           {len(diy_backends)}")
            click.echo(f"  Benchmark categories: {len(diy_benchmarks)}")
        else:
            click.echo("\nℹ Run any apps-benchmark command to initialize local_dev directory")

        click.echo("=" * 60)

    except Exception as exc:
        click.echo(f"Error reading configuration: {exc}", err=True)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
