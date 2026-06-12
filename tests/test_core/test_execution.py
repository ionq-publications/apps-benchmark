"""
Tests for shared benchmark execution helpers.
"""

import pytest
from apps_benchmark.core.execution import (
    classify_solution_algorithms,
    find_benchmark_case_by_uuid,
    get_available_runner_names,
    load_backend,
    load_runner,
    resolve_shots,
    select_solution_algorithm,
)
from apps_benchmark.errors import BenchmarkError, BenchmarkValidationError, OpenBenchmarkError
from apps_benchmark.primitives.benchmark_case import BenchmarkCase


def test_resolve_shots_uses_expected_precedence() -> None:
    messages: list[str] = []

    shots = resolve_shots(
        user_shots=25,
        config_shots=50,
        benchmark_shots=75,
        reporter=messages.append,
    )

    assert shots == 25
    assert messages == ["Shots per circuit set by user: 25"]


def test_resolve_shots_rejects_non_positive_values() -> None:
    with pytest.raises(BenchmarkValidationError, match="Shots must be positive"):
        resolve_shots(config_shots=-1)


def test_find_benchmark_case_by_uuid_uses_registry() -> None:
    result = find_benchmark_case_by_uuid("f75ae75f")

    assert result is not None
    case_path, category = result
    assert category == "qft"
    assert case_path.name == "qft_10_high_freq_challenge.json"


def test_classify_solution_algorithms_partitions_case_metadata() -> None:
    benchmark_case = BenchmarkCase(
        benchmark_category="example",
        problem_type="demo",
        instance_name="demo_case",
        instance_id="demo_001",
        num_qubits=2,
        solution_algorithms=["shipped", "open", "unregistered"],
        open_solution_algorithms=["open"],
        data={},
    )

    open_algorithms, shipped_algorithms, unregistered_algorithms = classify_solution_algorithms(
        benchmark_case,
        {"shipped"},
    )

    assert open_algorithms == ["open"]
    assert shipped_algorithms == ["shipped"]
    assert unregistered_algorithms == ["unregistered"]


def test_selecting_open_or_unregistered_algorithms_raises_clear_errors() -> None:
    benchmark_case = BenchmarkCase(
        benchmark_category="example",
        problem_type="demo",
        instance_name="demo_case",
        instance_id="demo_001",
        num_qubits=2,
        solution_algorithms=["open", "unregistered"],
        open_solution_algorithms=["open"],
        data={},
    )

    with pytest.raises(BenchmarkError, match="no registered runners"):
        select_solution_algorithm(benchmark_case, set(), "unregistered")


def test_unregistered_algorithms_fail_closed_even_with_shipped_runner() -> None:
    benchmark_case = BenchmarkCase(
        benchmark_category="example",
        problem_type="demo",
        instance_name="demo_case",
        instance_id="demo_001",
        num_qubits=2,
        solution_algorithms=["shipped", "unregistered"],
        data={},
    )

    with pytest.raises(BenchmarkError, match="no registered runners"):
        select_solution_algorithm(benchmark_case, {"shipped"})


def test_open_algorithms_still_raise_open_benchmark_errors() -> None:
    benchmark_case = BenchmarkCase(
        benchmark_category="example",
        problem_type="demo",
        instance_name="demo_case",
        instance_id="demo_001",
        num_qubits=2,
        solution_algorithms=["open"],
        open_solution_algorithms=["open"],
        data={},
    )

    with pytest.raises(OpenBenchmarkError, match="open benchmark"):
        select_solution_algorithm(benchmark_case, set(), "open")


def test_load_backend_and_runner_from_builtin_registries() -> None:
    backend = load_backend("mock_backend")
    runner = load_runner("qft", "cosine_qft")

    assert backend.name() == "mock"
    assert runner.name() == "cosine_qft"
    assert "cosine_qft" in get_available_runner_names("qft")
