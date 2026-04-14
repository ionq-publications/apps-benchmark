"""Smoke tests for the data-first QLBM benchmark case."""

from __future__ import annotations

from pathlib import Path

from apps_benchmark.core.registry import _discover_builtin_benchmarks
from apps_benchmark.primitives.benchmark_case import BenchmarkCase

REPO_ROOT = Path(__file__).resolve().parents[3]
QLBM_ROOT = REPO_ROOT / "apps_benchmark" / "benchmarks" / "computational_fluid_dynamics"
QLBM_CASES = QLBM_ROOT / "benchmark_cases"
QLBM_CASE = QLBM_CASES / "qlbm_16by16_2d_advection_diffusion.json"


def test_qlbm_case_loads() -> None:
    benchmark_case = BenchmarkCase.load_from_database(QLBM_CASE)

    assert benchmark_case.benchmark_category == "computational_fluid_dynamics"
    assert benchmark_case.problem_type == "quantum_lattice_boltzmann_method"
    assert benchmark_case.instance_name == "16by16_2d_advection_diffusion"
    assert benchmark_case.num_qubits == 15
    assert benchmark_case.solution_algorithms == ["qlbm"]
    assert benchmark_case.open_solution_algorithms == ["qlbm"]
    assert benchmark_case.instance_id == "fe2e221a"
    assert benchmark_case.data["grid size"] == 16
    assert benchmark_case.data["spatial dimension"] == 2
    assert benchmark_case.data["directions"] == 5
    assert benchmark_case.data["e"] == [0, -1, 1, -1, 1]
    assert benchmark_case.data["u"] == [0, 0.2, 0.2, 0.15, 0.15]
    assert len(benchmark_case.data["initial state"]) == 256


def test_qlbm_registry_lists_case_without_runners() -> None:
    benchmarks = _discover_builtin_benchmarks()

    assert benchmarks["computational_fluid_dynamics"]["runners"] == []
    assert benchmarks["computational_fluid_dynamics"]["benchmark_cases"] == [
        {
            "uuid": "fe2e221a",
            "name": "16by16_2d_advection_diffusion",
            "problem_type": "quantum_lattice_boltzmann_method",
            "file": str(QLBM_CASE),
            "builtin": True,
            "open_solution_algorithms": ["qlbm"],
            "all_solutions_open": True,
        }
    ]


def test_qlbm_support_files_exist() -> None:
    assert (QLBM_ROOT / "README.md").exists()
    assert (QLBM_ROOT / "qlbm_schema.json").exists()
