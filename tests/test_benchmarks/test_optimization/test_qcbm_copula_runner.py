"""
Tests for the built-in QCBM copula optimization benchmark.
"""

from pathlib import Path

import numpy as np
from apps_benchmark.backends.mock_backend import MockBackend
from apps_benchmark.benchmarks.optimization.algorithms.qcbm_copula_runner import (
    QcbmCopulaRunner,
    convert_to_real_space,
    get_copula_samples,
)
from apps_benchmark.cli import _load_builtin_runner
from apps_benchmark.core.registry import list_builtin_benchmarks
from apps_benchmark.primitives.benchmark_case import BenchmarkCase

REPO_ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_CASES = REPO_ROOT / "apps_benchmark" / "benchmarks" / "optimization" / "benchmark_cases"


def test_ansatz_1_copula_case_builds_bound_all_to_all_circuit() -> None:
    benchmark_case = BenchmarkCase.load_from_database(
        BENCHMARK_CASES / "quantum_copula_ansatz_1_05_variables.instance.json"
    )
    runner = QcbmCopulaRunner()

    circuits = runner.get_benchmark_circuits(benchmark_case)

    assert len(circuits) == 1
    assert circuits[0].num_qubits == benchmark_case.num_qubits
    assert circuits[0].num_parameters == 0
    ops = circuits[0].count_ops()
    assert ops.get("rxx", 0) > 0
    assert ops.get("rzz", 0) == 0


def test_ansatz_2_copula_case_builds_bound_structured_circuit() -> None:
    benchmark_case = BenchmarkCase.load_from_database(
        BENCHMARK_CASES / "quantum_copula_ansatz_2_05_variables.instance.json"
    )
    runner = QcbmCopulaRunner()

    circuits = runner.get_benchmark_circuits(benchmark_case)

    assert len(circuits) == 1
    assert circuits[0].num_qubits == benchmark_case.num_qubits
    assert circuits[0].num_parameters == 0
    ops = circuits[0].count_ops()
    assert ops.get("cx", 0) > 0
    assert ops.get("rzz", 0) > 0
    assert ops.get("rxx", 0) == 0


def test_copula_case_matrix_is_explicit_and_matches_reference_assets() -> None:
    cases: dict[tuple[str, int], BenchmarkCase] = {}
    for path in sorted(BENCHMARK_CASES.glob("quantum_copula_*.instance.json")):
        benchmark_case = BenchmarkCase.load_from_database(path)
        family = benchmark_case.data["ansatz"]["family"]
        num_variables = benchmark_case.data["portfolio"]["num_variables"]
        cases[(family, num_variables)] = benchmark_case

    assert len(cases) == 12
    assert set(cases) == {
        (family, num_variables)
        for family in ("ansatz_1", "ansatz_2")
        for num_variables in range(5, 11)
    }

    for num_variables in range(5, 11):
        dense_assets = cases[("ansatz_1", num_variables)].data["portfolio"]["asset_symbols"]
        structured_assets = cases[("ansatz_2", num_variables)].data["portfolio"]["asset_symbols"]
        assert dense_assets == structured_assets

    assert cases[("ansatz_1", 8)].data["portfolio"]["asset_symbols"] == [
        "AAPL",
        "ADBE",
        "AMZN",
        "FORD",
        "INTC",
        "JPM",
        "KO",
        "MSFT",
    ]
    assert cases[("ansatz_1", 9)].data["portfolio"]["asset_symbols"] == [
        "AAPL",
        "ADBE",
        "AMZN",
        "FORD",
        "INTC",
        "JPM",
        "KO",
        "MCD",
        "MSFT",
    ]
    assert cases[("ansatz_2", 10)].data["portfolio"]["asset_symbols"] == [
        "AAPL",
        "ADBE",
        "AMZN",
        "FORD",
        "INTC",
        "JPM",
        "KO",
        "MCD",
        "MSFT",
        "PFE",
    ]


def test_copula_merit_figures_use_stored_reference_var() -> None:
    runner = QcbmCopulaRunner()
    runner._shots = 100

    counts = {"00": 100}
    copula_samples = get_copula_samples(counts, num_qubits=2, bits_per_variable=2, seed=0)
    real_space = convert_to_real_space(
        copula_samples,
        np.asarray([[5.0, 0.0, 1.0]], dtype=float),
        np.asarray([0.0], dtype=float),
        np.asarray([1.0], dtype=float),
    )
    generated_var = float(np.quantile(-real_space[:, 0], 0.95))

    benchmark_case = BenchmarkCase(
        benchmark_category="optimization",
        problem_type="quantum_copula_risk_assessment",
        instance_name="unit_test_copula_case",
        instance_id="copula_unit_case",
        num_qubits=2,
        solution_algorithms=["qcbm_copula"],
        data={
            "ansatz": {"family": "ansatz_1"},
            "portfolio": {"num_variables": 1, "qubits_per_variable": 2},
            "benchmark_spec": {"var_confidence_level": 0.95},
            "marginals": {
                "fit_parameters": [[5.0, 0.0, 1.0]],
                "means": [0.0],
                "standard_deviations": [1.0],
            },
            "reference_metrics": {"reference_var_95": generated_var * 2.0},
        },
    )

    merit = runner.merit_figures_from_measurements([counts], benchmark_case)

    assert merit["alpha"] == 0.95
    assert merit["ansatz_family"] == "ansatz_1"
    assert merit["VaR_ratio"] == 0.5
    assert merit["score"] == 0.5
    assert merit["reference_VaR"] == generated_var * 2.0
    assert merit["reference_VaR_95"] == generated_var * 2.0
    assert merit["generated_VaR"] == generated_var
    assert merit["generated_VaR_95"] == generated_var


def test_copula_runner_executes_end_to_end_with_mock_backend() -> None:
    benchmark_case = BenchmarkCase.load_from_database(
        BENCHMARK_CASES / "quantum_copula_ansatz_1_05_variables.instance.json"
    )
    runner = QcbmCopulaRunner()

    record = runner.run_benchmark(benchmark_case, MockBackend(deterministic=True), shots=100)

    assert record.solution_algorithm == "qcbm_copula"
    assert record.job_id.startswith("mock_job_")
    assert record.score >= 0.0
    assert record.problem_specific_data["ansatz_family"] == "ansatz_1"
    assert record.problem_specific_data["reference_VaR"] > 0.0
    assert record.problem_specific_data["generated_VaR"] > 0.0


def test_builtin_loading_and_registry_discover_copula_runner_and_cases() -> None:
    runner = _load_builtin_runner("optimization", "qcbm_copula")
    assert isinstance(runner, QcbmCopulaRunner)

    builtin_benchmarks = list_builtin_benchmarks()
    optimization = builtin_benchmarks["optimization"]
    assert "qcbm_copula" in optimization["runners"]

    copied_cases = [
        Path(entry["file"]).name
        for entry in optimization["benchmark_cases"]
        if Path(entry["file"]).name.startswith("quantum_copula_")
    ]
    assert len(copied_cases) == 12
    assert "quantum_copula_ansatz_1_05_variables.instance.json" in copied_cases
    assert "quantum_copula_ansatz_2_10_variables.instance.json" in copied_cases


def test_varqite_cases_are_tagged_as_open_benchmark_algorithms() -> None:
    builtin_benchmarks = list_builtin_benchmarks()
    optimization = builtin_benchmarks["optimization"]

    varqite_cases = [
        entry
        for entry in optimization["benchmark_cases"]
        if "varqite" in entry.get("open_solution_algorithms", [])
    ]

    assert len(varqite_cases) == 10
    assert all(entry["all_solutions_open"] is False for entry in varqite_cases)
