"""
Tests for QFT runner discovery and execution.

Copyright (c) 2025 IonQ, Inc. All rights reserved.
Licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0
International (CC BY-NC-ND 4.0) License.
For details, visit: https://creativecommons.org/licenses/by-nc-nd/4.0/
"""

import math
from pathlib import Path

import numpy as np
import pytest
from apps_benchmark.benchmarks.qft.algorithms.cosine_qft_runner import CosineQftRunner
from apps_benchmark.benchmarks.qft.algorithms.hidden_phase_qft_runner import HiddenPhaseQftRunner
from apps_benchmark.cli import _load_builtin_runner
from apps_benchmark.core.registry import _discover_builtin_benchmarks
from apps_benchmark.primitives.benchmark_case import BenchmarkCase

from .conftest import SeededSamplingBackend, assert_unmeasured, exact_probabilities

REPO_ROOT = Path(__file__).resolve().parents[3]
QFT_CASES = REPO_ROOT / "apps_benchmark" / "benchmarks" / "qft" / "benchmark_cases"
ALL_QFT_CASE_NAMES = sorted(path.name for path in QFT_CASES.glob("qft_*.json"))


def load_case(name: str) -> BenchmarkCase:
    return BenchmarkCase.load_from_database(QFT_CASES / name)


def make_case(num_qubits: int, solution_algorithm: str, data: dict[str, int]) -> BenchmarkCase:
    return BenchmarkCase(
        benchmark_category="qft",
        problem_type="qft",
        instance_name=f"synthetic_{solution_algorithm}",
        num_qubits=num_qubits,
        solution_algorithms=[solution_algorithm],
        data=data,
    )


class TestQftRunnerDiscovery:
    """Tests for built-in qft runner discovery and loading."""

    def test_builtin_registry_lists_only_final_qft_runners(self) -> None:
        benchmarks = _discover_builtin_benchmarks()

        assert sorted(benchmarks["qft"]["runners"]) == [
            "cosine_qft",
            "hidden_phase_qft",
        ]
        assert len(benchmarks["qft"]["benchmark_cases"]) == 25

    @pytest.mark.parametrize(
        "runner_name",
        ["cosine_qft", "hidden_phase_qft"],
    )
    def test_builtin_loader_can_import_qft_runners(self, runner_name: str) -> None:
        runner = _load_builtin_runner("qft", runner_name)

        assert runner.name() == runner_name
        assert runner.benchmark_category == "qft"


@pytest.mark.parametrize(
    ("runner_cls", "extra_qubits"),
    [
        (CosineQftRunner, 0),
        (HiddenPhaseQftRunner, 1),
    ],
)
def test_qft_runners_build_unmeasured_circuits(runner_cls, extra_qubits: int) -> None:
    benchmark_case = load_case("qft_10_high_freq_challenge.json")
    runner = runner_cls()

    circuits = runner.get_benchmark_circuits(benchmark_case)

    assert len(circuits) == 1
    assert circuits[0].num_qubits == benchmark_case.num_qubits + extra_qubits
    assert_unmeasured(circuits[0])


@pytest.mark.parametrize("case_name", ALL_QFT_CASE_NAMES)
@pytest.mark.parametrize(
    ("runner_cls", "extra_qubits"),
    [
        (CosineQftRunner, 0),
        (HiddenPhaseQftRunner, 1),
    ],
)
def test_all_shipped_qft_cases_build_for_all_runners(
    case_name: str, runner_cls, extra_qubits: int
) -> None:
    benchmark_case = load_case(case_name)
    runner = runner_cls()

    circuits = runner.get_benchmark_circuits(benchmark_case)

    assert len(circuits) == 1
    assert circuits[0].num_qubits == benchmark_case.num_qubits + extra_qubits
    assert_unmeasured(circuits[0])


@pytest.mark.parametrize(
    ("runner_cls", "runner_name"),
    [
        (CosineQftRunner, "cosine_qft"),
        (HiddenPhaseQftRunner, "hidden_phase_qft"),
    ],
)
def test_qft_runners_execute_with_seeded_backend(runner_cls, runner_name: str) -> None:
    benchmark_case = load_case("qft_10_high_freq_challenge.json")
    runner = runner_cls()

    record = runner.run_benchmark(
        benchmark_case,
        SeededSamplingBackend(seed=123),
        shots=128,
    )

    assert record.benchmark_category == "qft"
    assert record.solution_algorithm == runner_name
    assert record.num_qubits == benchmark_case.num_qubits
    assert np.isfinite(record.score)
    assert record.score >= 0.0


@pytest.mark.parametrize(
    ("runner_cls", "case_name"),
    [
        (CosineQftRunner, "qft_10_high_freq_challenge.json"),
        (CosineQftRunner, "qft_12_high_freq_challenge.json"),
        (HiddenPhaseQftRunner, "qft_10_high_freq_challenge.json"),
        (HiddenPhaseQftRunner, "qft_12_high_freq_challenge.json"),
    ],
)
def test_builtin_qft_cases_score_perfectly_on_exact_statevector(runner_cls, case_name: str) -> None:
    benchmark_case = load_case(case_name)
    runner = runner_cls()

    probs = exact_probabilities(runner.get_benchmark_circuits(benchmark_case)[0])
    score = runner.merit_figures_from_measurements([probs], benchmark_case)["score"]

    assert score == pytest.approx(1.0, abs=1e-12)


def test_cosine_qft_exact_spectrum_tracks_frequency_index() -> None:
    benchmark_case = make_case(4, "cosine_qft", {"frequency_index": 5})
    runner = CosineQftRunner()

    probs = exact_probabilities(runner.get_benchmark_circuits(benchmark_case)[0])

    assert probs == pytest.approx({"0101": 0.5, "1011": 0.5}, abs=1e-12)
    assert runner.merit_figures_from_measurements([probs], benchmark_case)[
        "score"
    ] == pytest.approx(
        1.0,
        abs=1e-12,
    )

    wrong_case = make_case(4, "cosine_qft", {"frequency_index": 4})
    assert runner.merit_figures_from_measurements([probs], wrong_case)["score"] == pytest.approx(
        0.0,
        abs=1e-12,
    )


def test_cosine_qft_zero_frequency_collapses_to_dc_peak() -> None:
    benchmark_case = make_case(4, "cosine_qft", {"frequency_index": 0})
    runner = CosineQftRunner()

    probs = exact_probabilities(runner.get_benchmark_circuits(benchmark_case)[0])

    assert probs == pytest.approx({"0000": 1.0}, abs=1e-12)
    assert runner.merit_figures_from_measurements([probs], benchmark_case)[
        "score"
    ] == pytest.approx(
        1.0,
        abs=1e-12,
    )


def test_hidden_phase_qft_exact_distribution_tracks_phase_index() -> None:
    benchmark_case = make_case(4, "hidden_phase_qft", {"phase_index": 7})
    runner = HiddenPhaseQftRunner()

    probs = exact_probabilities(runner.get_benchmark_circuits(benchmark_case)[0])
    lam = math.pi * benchmark_case.data["phase_index"] / 2**benchmark_case.num_qubits
    target = {
        "0" * (benchmark_case.num_qubits + 1): math.cos(lam) ** 2,
        "1" + "0" * benchmark_case.num_qubits: math.sin(lam) ** 2,
    }

    assert probs == pytest.approx(target, abs=1e-12)
    assert runner.merit_figures_from_measurements([probs], benchmark_case)[
        "score"
    ] == pytest.approx(
        1.0,
        abs=1e-12,
    )

    wrong_case = make_case(4, "hidden_phase_qft", {"phase_index": 1})
    assert runner.merit_figures_from_measurements([probs], wrong_case)["score"] < 0.2
