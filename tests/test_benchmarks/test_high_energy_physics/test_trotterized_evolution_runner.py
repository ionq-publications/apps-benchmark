"""Smoke tests for the high-energy-physics benchmark family."""

from __future__ import annotations

from pathlib import Path

import pytest

from apps_benchmark.benchmarks.high_energy_physics.algorithms.trotterized_evolution_runner import (
    TrotterizedEvolutionRunner,
)
from apps_benchmark.cli import _load_builtin_runner
from apps_benchmark.core.registry import _discover_builtin_benchmarks
from apps_benchmark.primitives.benchmark_case import BenchmarkCase

REPO_ROOT = Path(__file__).resolve().parents[3]
HEP_CASES = REPO_ROOT / "apps_benchmark" / "benchmarks" / "high_energy_physics" / "benchmark_cases"


def load_case(name: str) -> BenchmarkCase:
    """Load one HEP benchmark case by filename."""
    return BenchmarkCase.load_from_database(HEP_CASES / name)


def test_hep_registry_lists_runner_and_cases() -> None:
    benchmarks = _discover_builtin_benchmarks()

    assert benchmarks["high_energy_physics"]["runners"] == ["trotterized_evolution"]
    assert len(benchmarks["high_energy_physics"]["benchmark_cases"]) == 8


def test_hep_loader_imports_runner() -> None:
    runner = _load_builtin_runner("high_energy_physics", "trotterized_evolution")

    assert runner.name() == "trotterized_evolution"
    assert runner.benchmark_category == "high_energy_physics"


def test_hep_runner_loads_qasm_independently_of_cwd(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    benchmark_case = load_case("hep_mM0_t0p5.json")
    circuit = TrotterizedEvolutionRunner().get_benchmark_circuits(benchmark_case)[0]

    assert circuit.num_qubits == benchmark_case.num_qubits
    assert circuit.num_clbits == 0
    assert all(inst.operation.name != "measure" for inst in circuit.data)


@pytest.mark.parametrize(
    ("case_name", "expected_l", "expected_qe"),
    [
        ("hep_mM0_t0p5.json", 0.0, -0.45),
        ("hep_mM0_t1p0.json", 0.0, -1.1),
        ("hep_mM0_t1p5.json", 0.0, -1.2),
        ("hep_mM0_t2p0.json", 0.0, -1.09),
        ("hep_mM1p7_t0p5.json", 0.01, -0.44),
        ("hep_mM1p7_t1p0.json", 0.15, -1.08),
        ("hep_mM1p7_t1p5.json", 0.59, -1.26),
        ("hep_mM1p7_t2p0.json", 1.31, -1.34),
    ],
)
def test_hep_cases_preserve_baseline_reference_values(
    case_name: str,
    expected_l: float,
    expected_qe: float,
) -> None:
    benchmark_case = load_case(case_name)

    assert benchmark_case.data["L"] == expected_l
    assert benchmark_case.data["Qe"] == expected_qe
    assert benchmark_case.open_solution_algorithms == ["trotterized_evolution"]


def test_hep_score_preserves_signed_deviation() -> None:
    benchmark_case = load_case("hep_mM0_t0p5.json")
    result = TrotterizedEvolutionRunner().merit_figures_from_measurements(
        [{"1" * benchmark_case.num_qubits: 10}],
        benchmark_case,
    )

    assert result["score"] == 4.0
    assert result["signed_lepton_number_error"] == -4.0
    assert result["reference_lepton_number"] == 0.0
    assert result["reference_lepton_charge"] == -0.45
