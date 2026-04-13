"""
Tests for hidden-shift runner discovery and execution.

Copyright (c) 2025 IonQ, Inc. All rights reserved.
Licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0
International (CC BY-NC-ND 4.0) License.
For details, visit: https://creativecommons.org/licenses/by-nc-nd/4.0/
"""

import json
from pathlib import Path

import numpy as np
import pytest
from apps_benchmark.benchmarks.hidden_shift.algorithms.hidden_shift_runner import HiddenShiftRunner
from apps_benchmark.cli import _find_benchmark_case_by_uuid, _load_builtin_runner
from apps_benchmark.core.registry import _discover_builtin_benchmarks
from apps_benchmark.primitives.benchmark_case import BenchmarkCase
from qiskit.quantum_info import Statevector


def assert_unmeasured(circuit) -> None:
    """Assert a circuit has no classical bits and no measurement gates."""
    assert circuit.num_clbits == 0
    assert all(inst.operation.name != "measure" for inst in circuit.data)


class SeededSamplingBackend:
    """Deterministic statevector sampler for runner contract tests."""

    def __init__(self, seed: int):
        self._rng = np.random.default_rng(seed)
        self._calls = 0

    def name(self) -> str:
        return "seeded-sampling"

    def run(self, circuits, shots: int = 1000, job_name: str | None = None):
        histograms = []
        for qc in circuits:
            assert_unmeasured(qc)
            probs = Statevector.from_instruction(qc).probabilities_dict()
            bitstrings = sorted(probs)
            weights = np.array([probs[b] for b in bitstrings], dtype=float)
            weights /= weights.sum()
            draws = self._rng.choice(len(bitstrings), size=int(shots), p=weights)
            unique_idx, counts = np.unique(draws, return_counts=True)
            histograms.append({bitstrings[i]: int(c) for i, c in zip(unique_idx, counts, strict=True)})
        self._calls += 1
        return histograms, f"job-{self._calls}", {"calls": self._calls}


REPO_ROOT = Path(__file__).resolve().parents[3]
HIDDEN_SHIFT_CASES = REPO_ROOT / "apps_benchmark" / "benchmarks" / "hidden_shift" / "benchmark_cases"


def load_case(name: str) -> BenchmarkCase:
    return BenchmarkCase.load_from_database(HIDDEN_SHIFT_CASES / name)


def iter_hidden_shift_case_paths() -> list[Path]:
    return sorted(HIDDEN_SHIFT_CASES.glob("hsbp_*.json"))


def iter_hidden_shift_cases() -> list[BenchmarkCase]:
    return [BenchmarkCase.load_from_database(path) for path in iter_hidden_shift_case_paths()]


def expected_shifts(benchmark_case: BenchmarkCase) -> list[str]:
    shifts = list(benchmark_case.data["shifts"])
    if benchmark_case.data["permutation"] == "random":
        return shifts * len(benchmark_case.data["permutation_cx_pairs"])
    return shifts


def ideal_probabilities_for_expected_shifts(
    runner: HiddenShiftRunner, benchmark_case: BenchmarkCase
) -> list[float]:
    probabilities = []
    circuits = runner.get_benchmark_circuits(benchmark_case, decompose=False)
    for qc, shift in zip(circuits, expected_shifts(benchmark_case), strict=True):
        probs = Statevector.from_instruction(qc).probabilities_dict()
        probabilities.append(float(probs.get(shift, 0.0)))
    return probabilities


def test_hidden_shift_registry_lists_runner() -> None:
    benchmarks = _discover_builtin_benchmarks()

    assert benchmarks["hidden_shift"]["runners"] == ["hidden_shift"]
    assert any(case["uuid"] == "83af7cf5" for case in benchmarks["hidden_shift"]["benchmark_cases"])
    assert all(
        "/hidden_shift/benchmark_cases/" in case["file"]
        for case in benchmarks["hidden_shift"]["benchmark_cases"]
    )


def test_hidden_shift_loader_imports_runner() -> None:
    runner = _load_builtin_runner("hidden_shift", "hidden_shift")

    assert runner.name() == "hidden_shift"
    assert runner.benchmark_category == "hidden_shift"


def test_hidden_shift_runner_builds_unmeasured_circuits() -> None:
    runner = HiddenShiftRunner()
    benchmark_case = load_case("hsbp_06_qubit_cx_ladder_challenge.json")

    circuits = runner.get_benchmark_circuits(benchmark_case)

    assert len(circuits) == len(benchmark_case.data["shifts"])
    assert all(qc.num_qubits == benchmark_case.num_qubits for qc in circuits)
    for circuit in circuits:
        assert_unmeasured(circuit)


def test_hidden_shift_runner_executes_with_seeded_backend() -> None:
    runner = HiddenShiftRunner()
    benchmark_case = load_case("hsbp_06_qubit_cx_ladder_challenge.json")

    record = runner.run_benchmark(
        benchmark_case,
        SeededSamplingBackend(seed=123),
        shots=128,
    )

    assert record.benchmark_category == "hidden_shift"
    assert record.solution_algorithm == "hidden_shift"
    assert record.num_qubits == benchmark_case.num_qubits
    assert np.isfinite(record.score)
    assert 0.0 <= record.score <= 1.0


def test_hidden_shift_case_loads_from_new_category() -> None:
    problem = load_case("hsbp_06_qubit_cx_ladder_challenge.json")

    assert problem.benchmark_category == "hidden_shift"
    assert problem.problem_type == "hidden_shift_cx_ladder"
    assert problem.instance_name == "hsbp_06q_cx_ladder"
    assert problem.instance_id == "83af7cf5"


def test_hidden_shift_uuid_lookup_resolves_new_category() -> None:
    result = _find_benchmark_case_by_uuid("83af7cf5")

    assert result is not None
    problem_path, category, runner_name = result
    assert problem_path == HIDDEN_SHIFT_CASES / "hsbp_06_qubit_cx_ladder_challenge.json"
    assert category == "hidden_shift"
    assert runner_name == "hidden_shift"


def test_hidden_shift_benchmark_cases_are_well_formed() -> None:
    benchmark_cases = iter_hidden_shift_cases()

    assert len(benchmark_cases) == 170

    for benchmark_case in benchmark_cases:
        assert benchmark_case.benchmark_category == "hidden_shift"
        assert benchmark_case.solution_algorithms == ["hidden_shift"]
        assert benchmark_case.problem_type.startswith("hidden_shift_")
        assert benchmark_case.num_qubits % 2 == 0
        assert benchmark_case.instance_id

        shifts = benchmark_case.data["shifts"]
        assert shifts
        assert all(len(shift) == benchmark_case.num_qubits for shift in shifts)
        assert all(set(shift) <= {"0", "1"} for shift in shifts)

        permutation = benchmark_case.data["permutation"]
        if permutation == "random":
            pairs_by_permutation = benchmark_case.data["permutation_cx_pairs"]
            n_half = benchmark_case.num_qubits // 2
            assert benchmark_case.data["num_random_permutations"] == len(pairs_by_permutation)
            for pairs in pairs_by_permutation:
                for ctrl, tgt in pairs:
                    assert 0 <= ctrl < n_half
                    assert 0 <= tgt < n_half
                    assert ctrl != tgt


def test_hidden_shift_small_cases_are_exactly_solved_ideally() -> None:
    runner = HiddenShiftRunner()
    benchmark_cases = [
        benchmark_case
        for benchmark_case in iter_hidden_shift_cases()
        if benchmark_case.num_qubits <= 12
    ]

    assert any(benchmark_case.data["permutation"] == "random" for benchmark_case in benchmark_cases)

    for benchmark_case in benchmark_cases:
        probabilities = ideal_probabilities_for_expected_shifts(runner, benchmark_case)
        assert probabilities
        assert all(
            pytest.approx(1.0, abs=1e-12) == probability for probability in probabilities
        ), benchmark_case.instance_name


def test_hidden_shift_small_cases_score_perfectly_with_exact_histograms() -> None:
    runner = HiddenShiftRunner()
    benchmark_cases = [
        benchmark_case
        for benchmark_case in iter_hidden_shift_cases()
        if benchmark_case.num_qubits <= 12
    ]

    for benchmark_case in benchmark_cases:
        perfect_histograms = [{shift: 256} for shift in expected_shifts(benchmark_case)]
        merit_figures = runner.merit_figures_from_measurements(perfect_histograms, benchmark_case)

        assert merit_figures["score"] == pytest.approx(1.0, abs=1e-12)
        assert merit_figures["score_std"] == pytest.approx(0.0, abs=1e-12)
        assert np.allclose(merit_figures["score_arr"], 1.0)


def test_hidden_shift_case_files_contain_no_qft_metadata() -> None:
    for path in iter_hidden_shift_case_paths():
        payload = json.loads(path.read_text())
        assert payload["benchmark_category"] == "hidden_shift"
        assert "qft" not in payload["problem_type"]
