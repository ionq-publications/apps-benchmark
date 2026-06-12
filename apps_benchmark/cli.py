"""
Command-line interface for apps-benchmark.

This module provides the Click-based CLI for the apps-benchmark tool.

Copyright (c) 2025 IonQ, Inc. All rights reserved.
Licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0
International (CC BY-NC-ND 4.0) License.
For details, visit: https://creativecommons.org/licenses/by-nc-nd/4.0/
"""

import json
from pathlib import Path

import click

from apps_benchmark.core import execution as execution_helpers
from apps_benchmark.core.backend import AbstractBackend
from apps_benchmark.core.benchmark import AbstractAlgoRunner, BenchmarkSubmissionRecord
from apps_benchmark.core.execution import (
    DEFAULT_SHOTS_PER_QC,
    classify_solution_algorithms,
    find_benchmark_case_by_uuid,
    get_available_runner_names,
    get_shots_for_case,
    load_backend,
    load_runner,
    resolve_shots,
    select_solution_algorithm,
)
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
    OpenBenchmarkError,
)
from apps_benchmark.primitives.benchmark_case import BenchmarkCase
from apps_benchmark.utils.cli_config import load_cli_config, save_cli_config
from apps_benchmark.utils.config import get_config_file_path, get_local_dev_dir_from_config

_DEFAULT_SHOTS_PER_QC = DEFAULT_SHOTS_PER_QC


def _resolve_shots(
    cli_set_shots: int | None,
    config_set_shots: int | None,
    benchmark_set_shots: int | None,
) -> int:
    return resolve_shots(
        user_shots=cli_set_shots,
        config_shots=config_set_shots,
        benchmark_shots=benchmark_set_shots,
        reporter=click.echo,
    )


def _get_shots_for_case(bm_case: BenchmarkCase) -> int | None:
    return get_shots_for_case(bm_case)


def _get_available_runner_names(
    category: str,
    builtin_benchmarks: dict[str, dict] | None = None,
    diy_benchmarks: dict[str, dict] | None = None,
) -> set[str]:
    return get_available_runner_names(category, builtin_benchmarks, diy_benchmarks)


def _classify_solution_algorithms(
    benchmark_case: BenchmarkCase,
    available_runner_names: set[str],
) -> tuple[list[str], list[str], list[str]]:
    return classify_solution_algorithms(benchmark_case, available_runner_names)


def _select_solution_algorithm(
    benchmark_case: BenchmarkCase,
    available_runner_names: set[str],
    requested_algorithm: str | None = None,
) -> str:
    return select_solution_algorithm(
        benchmark_case,
        available_runner_names,
        requested_algorithm,
    )


def _load_backend(backend_name: str) -> AbstractBackend:
    return load_backend(backend_name)


def _find_benchmark_case_by_uuid(uuid: str) -> tuple[Path, str, str] | None:
    result = find_benchmark_case_by_uuid(uuid)
    if result is None:
        return None
    case_path, category = result
    benchmark_case = BenchmarkCase.load_from_database(case_path)
    runner_name = benchmark_case.solution_algorithms[0]
    return case_path, category, runner_name


def _load_diy_runner(category: str, runner_name: str, runner_info: dict) -> AbstractAlgoRunner:
    return execution_helpers._load_diy_runner(category, runner_name, runner_info)


def _load_builtin_runner(category: str, runner_name: str) -> AbstractAlgoRunner:
    return execution_helpers._load_builtin_runner(category, runner_name)


def _load_runner(category: str, runner_name: str) -> AbstractAlgoRunner:
    return load_runner(category, runner_name)


def _run_single_benchmark(
    backend: AbstractBackend,
    uuid: str,
    cli_shots: int | None,
    config_shots: int | None,
    algorithm: str | None = None,
) -> None:
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
                   support multiple approaches (e.g., 'hidden_phase_qft' instead of
                   'cosine_qft').
    """
    click.echo(f"\nSearching for problem instance '{uuid}'...")

    # Find benchmark case
    result = _find_benchmark_case_by_uuid(uuid)
    if result is None:
        click.echo(f"Error: Benchmark case '{uuid}' not found", err=True)
        raise SystemExit(1)

    problem_path, category, _default_runner_name = result
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
            if not algorithm:
                click.echo("  ℹ  Use --algorithm to select a different solution algorithm")
        else:
            click.echo(f"  Algorithm: {problem.solution_algorithms[0]}")
    except Exception as exc:
        click.echo(f"Error loading problem instance: {exc}", err=True)
        raise SystemExit(1) from exc

    available_runner_names = _get_available_runner_names(category)
    open_algorithms, _, unregistered_algorithms = _classify_solution_algorithms(
        problem, available_runner_names
    )
    if len(problem.solution_algorithms) > 1:
        click.echo(f"  Available algorithms: {', '.join(problem.solution_algorithms)}")
        if not algorithm:
            click.echo("  ℹ  Use --algorithm to select a different solution algorithm")
    if open_algorithms:
        click.echo(
            f"  Open benchmark algorithms: {', '.join(open_algorithms)} (bring your own solver)"
        )
    if unregistered_algorithms:
        click.echo(
            "  Warning: algorithms missing registered runners: "
            f"{', '.join(unregistered_algorithms)}"
        )

    try:
        runner_name = _select_solution_algorithm(problem, available_runner_names, algorithm)
        if len(problem.solution_algorithms) > 1:
            click.echo(f"  Using algorithm: {runner_name}")
        else:
            click.echo(f"  Algorithm: {runner_name}")
    except BenchmarkError as exc:
        click.echo(f"Error selecting algorithm: {exc}", err=True)
        raise SystemExit(1) from exc

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

    # list them alphabetically by problem name( which should lead by count, by convention)
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
    qbit_max: int,
    cli_shots: int | None,
    config_shots: int | None,
    algorithm: str | None = None,
) -> None:
    """
    Run all benchmarks in a category.

    Args:
        backend: Backend instance
        category: Benchmark category
        qbit_max: Maximum qubits
        cli_shots: CLI-specified shots
        config_shots: Config-specified shots
        algorithm: Solution algorithm to use. If specified, only runs benchmarks that
                   support this algorithm (skips others). If None, uses the first
                   shipped closed algorithm in each benchmark's solution_algorithms
                   list. This allows running a specific solution method across
                   multiple benchmark cases.
    """
    click.echo(f"\nRunning benchmarks in category '{category}'...")
    click.echo(f"  Backend: {backend.name()}")
    click.echo(f"  Max qubits: {qbit_max}")

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

    # Filter by qbit_max
    filtered_cases = []
    for case_info in benchmark_cases:
        case_path = Path(case_info["file"])
        try:
            with open(case_path) as f:
                data = json.load(f)
            num_qubits = data.get("num_qubits", 0)
            if num_qubits <= qbit_max:
                filtered_cases.append(case_info)
        except Exception as e:
            click.echo(f"Warning: Failed to load case {case_path.name}")
            click.echo(f"Case is expected in case_config {category}: {e}", err=True)
            click.echo(f"Skipping case {case_path.name}", err=True)
            continue

    if not filtered_cases:
        click.echo(f"\nNo benchmark cases found with num_qubits <= {qbit_max}", err=True)
        raise SystemExit(1)

    filtered_out_count = len(benchmark_cases) - len(filtered_cases)
    click.echo(f"Found {filtered_out_count} benchmark(s) not to run (filtered by --qbit-max={qbit_max})")
    click.echo(f"\nFound {len(filtered_cases)} benchmark(s) to run:")
    available_runner_names = _get_available_runner_names(category, builtin, diy)
    runnable_cases: list[tuple[dict, BenchmarkCase, str]] = []
    open_skipped = 0
    algorithm_filtered = 0

    for case_info in filtered_cases:
        case_path = Path(case_info["file"])
        try:
            problem = BenchmarkCase.load_from_database(case_path)
        except Exception as exc:
            click.echo(f"Warning: Failed to load case {case_path.name}", err=True)
            click.echo(f"Skipping case {case_path.name}: {exc}", err=True)
            continue

        if algorithm and algorithm not in problem.solution_algorithms:
            algorithm_filtered += 1
            continue

        try:
            runner_name = _select_solution_algorithm(problem, available_runner_names, algorithm)
        except OpenBenchmarkError:
            open_skipped += 1
            continue
        except BenchmarkError as exc:
            click.echo(
                f"\nError selecting algorithm for benchmark case '{problem.instance_name}': {exc}",
                err=True,
            )
            raise SystemExit(1) from exc

        runnable_cases.append((case_info, problem, runner_name))

    filtered_out_count = len(benchmark_cases) - len(filtered_cases)
    click.echo(
        f"Found {filtered_out_count} benchmark(s) not to run (filtered by --qbit-max={qbit_max})"
    )
    if algorithm_filtered:
        click.echo(
            f"Found {algorithm_filtered} benchmark(s) not to run "
            f"(filtered by --algorithm={algorithm})"
        )
    if open_skipped:
        click.echo(
            "Found "
            f"{open_skipped} open benchmark case(s) not to run (require user-provided solvers)"
        )
    click.echo(f"\nFound {len(runnable_cases)} benchmark(s) to run:")
    for case_info, _, _ in runnable_cases:
        click.echo(f"  - {case_info['name']} (UUID: {case_info['uuid']})")
    click.echo("")

    # Run all benchmarks
    results: list[BenchmarkSubmissionRecord] = []
    total = len(runnable_cases)

    for idx, (case_info, problem, runner_name) in enumerate(runnable_cases, 1):
        case_name = case_info["name"]

        click.echo(f"[{idx}/{total}] Running {case_name}...", nl=False)

        try:
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
            raise SystemExit(1) from exc  # should we continue ? TBD by SME

    if open_skipped:
        click.echo(
            f"\nSkipped {open_skipped} open benchmark case(s) that require user-provided solvers."
        )
    if not results:
        click.echo(
            "\nError: no runnable closed benchmark cases matched. "
            "The selected cases are tagged as open benchmarks or otherwise lack "
            "shipped runners.",
            err=True,
        )
        raise SystemExit(1)

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
        "Maximum number of qubits to use (default: 10). Category runs filter out cases above this "
        "limit unless overridden. Note case-uuid takes precedence over qbit-max"
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
    help="Solution algorithm to use when multiple are available (e.g., 'hidden_phase_qft', 'lr_qaoa'). "
    "Use this to select alternative algorithms for benchmark cases that support multiple solution methods. "
    "Defaults to the first shipped closed algorithm listed in the benchmark case.",
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
    # Run all benchmarks in a category (default: --qbit-max=10 filters out larger cases)
    apps-benchmark run --backend=ionq.forte --qbit-max=4 --category=chemistry

    \b
    # Override the default category filter
    apps-benchmark run --backend=ionq.forte --category=chemistry --qbit-max=20

    \b
    # Run a specific benchmark
    apps-benchmark run --backend=qiskit --case-uuid=610cfb55

    \b
    # Run a specific benchmark with a specific solution algorithm
    apps-benchmark run --backend=qiskit --case-uuid=f75ae75f --algorithm=hidden_phase_qft

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

    # Track whether --qbit-max was explicitly provided before applying defaults
    qbit_max_explicitly_set = qbit_max is not None

    # Apply defaults for parameters still not set
    if qbit_max is None:
        qbit_max = 10

    # Handle save-config if requested
    if save_config:
        try:
            config_data = {
                "version": "1.0",
                "backend": backend,
                "qbit_max": qbit_max,
                "shots": cli_shots,  # Only store explicit CLI shot count, not a derived or fallback count
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
        _run_category_benchmarks(
            backend_instance, category, qbit_max, cli_shots, config_shots, algorithm
        )
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
                runners = info.get("runners", [])
                if runners:
                    for runner in runners:
                        click.echo(f"    - {runner}")
                else:
                    click.echo("    - none")
                open_algorithms = sorted(
                    {
                        algorithm
                        for case in info.get("benchmark_cases", [])
                        for algorithm in case.get("open_solution_algorithms", [])
                    }
                )
                if open_algorithms:
                    click.echo("  Open benchmark algorithms:")
                    for algorithm in open_algorithms:
                        click.echo(f"    - {algorithm}")
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
                    open_algorithms = sorted(
                        {
                            algorithm
                            for case in info.get("benchmark_cases", [])
                            for algorithm in case.get("open_solution_algorithms", [])
                        }
                    )
                    open_case_count = sum(
                        1
                        for case in info.get("benchmark_cases", [])
                        if case.get("open_solution_algorithms")
                    )
                    click.echo(f"      Built-in runners: {len(info.get('runners', []))}")
                    click.echo(
                        f"      Built-in problem instances: {len(info.get('benchmark_cases', []))}"
                    )
                    click.echo(f"      Open benchmark algorithms: {len(open_algorithms)}")
                    click.echo(f"      Open benchmark instances: {open_case_count}")
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
