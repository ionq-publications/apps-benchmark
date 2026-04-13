"""
Tests for VQE pUCCD runner measurement boundary and contract.

Ported from farmckon-ionq/apps-benchmarking2#6 with import paths adjusted
for the apps_benchmark package.

Copyright (c) 2025 IonQ, Inc. All rights reserved.
Licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0
International (CC BY-NC-ND 4.0) License.
For details, visit: https://creativecommons.org/licenses/by-nc-nd/4.0/
"""

from pathlib import Path

import numpy as np
import pytest

from apps_benchmark.backends.qiskit_aer_sim_backend import QiskitAerSimBackend
from apps_benchmark.benchmarks.chemistry.algorithms.vqe_puccd_runner import VqePuccdRunner
from apps_benchmark.primitives.benchmark_case import BenchmarkCase

from .conftest import SeededSamplingBackend, assert_unmeasured


class _ReplayBackend:
    """Returns pre-recorded histograms. Used by sync/async agreement test."""

    def __init__(self, histograms):
        self._histograms = histograms

    def name(self):
        return "replay"

    def run(self, circuits, shots=0, job_name=None):
        for qc in circuits:
            assert_unmeasured(qc)
        return self._histograms, "replay-job", {}


REPO_ROOT = Path(__file__).resolve().parents[3]
CHEMISTRY_CASES = REPO_ROOT / "apps_benchmark" / "benchmarks" / "chemistry" / "benchmark_cases"
CASE_NAMES = ["h002_chain_0_75.json", "h002_chain_1_00.json"]


def load_case(name: str) -> BenchmarkCase:
    return BenchmarkCase.load_from_database(CHEMISTRY_CASES / name)


def make_zero_param_case(**data_overrides) -> BenchmarkCase:
    data = {
        "paired_hamiltonian_dict": {
            "II": -1.0,
            "ZI": 0.5,
            "IZ": 0.25,
            "ZZ": 0.1,
        },
        "num_alpha": 0,
        "optimal_parameters": [],
        "reference_energy_doci": -0.15,
        "reference_energy_fci": -0.15,
        "hf_energy": -0.15,
        "vqe_final_energy": -0.15,
    }
    data.update(data_overrides)
    return BenchmarkCase(
        benchmark_category="chemistry",
        problem_type="hydrogen_lattice_vqe",
        instance_name="synthetic_zero_param",
        num_qubits=2,
        solution_algorithms=["vqe_puccd"],
        data=data,
    )


@pytest.mark.parametrize("case_name", CASE_NAMES)
def test_exact_inference_matches_benchmark_defined_vqe_energy(case_name: str) -> None:
    """Evaluating stored benchmark parameters reproduces the benchmark-defined pUCCD energy."""
    benchmark_case = load_case(case_name)
    runner = VqePuccdRunner()

    algo_inputs = runner.setup_algo_inputs(benchmark_case)
    result = runner.execute_benchmark_algo(
        algo_inputs,
        backend=SeededSamplingBackend(seed=123),
        shots=1000,
        exact_simulation=True,
        qiskit_primitive_version="v2",
    )

    assert result["optimizer_success"] is True
    assert result["nfev"] == 1
    assert result["final_energy_se"] == 0.0
    assert result["final_energy"] == pytest.approx(
        benchmark_case.data["vqe_final_energy"], rel=0.0, abs=1e-12
    )


@pytest.mark.parametrize("case_name", CASE_NAMES)
def test_exact_optimization_converges_to_benchmark_defined_vqe_energy(case_name: str) -> None:
    """Optimization path (not just stored-parameter evaluation) converges."""
    benchmark_case = load_case(case_name)
    benchmark_case.data = dict(benchmark_case.data)
    benchmark_case.data.pop("optimal_parameters", None)

    runner = VqePuccdRunner()
    algo_inputs = runner.setup_algo_inputs(benchmark_case)
    result = runner.execute_benchmark_algo(
        algo_inputs,
        backend=SeededSamplingBackend(seed=123),
        shots=1000,
        exact_simulation=True,
        qiskit_primitive_version="v2",
    )

    assert result["optimizer_success"] is True
    assert result["nfev"] > 1
    assert result["final_energy_se"] == 0.0
    assert result["final_energy"] == pytest.approx(
        benchmark_case.data["vqe_final_energy"], rel=0.0, abs=1e-8
    )


def test_exact_inference_keeps_zero_standard_errors_with_low_requested_shots() -> None:
    benchmark_case = load_case("h002_chain_0_75.json")
    runner = VqePuccdRunner()

    result = runner.execute_benchmark_algo(
        runner.setup_algo_inputs(benchmark_case),
        backend=SeededSamplingBackend(seed=123),
        shots=1,
        exact_simulation=True,
        qiskit_primitive_version="v2",
    )
    merit_figures = runner.compute_merit_figures(result, benchmark_case)

    assert result["final_energy_se"] == 0.0
    assert merit_figures["accuracy_vs_doci_se"] == 0.0
    assert merit_figures["score_se"] == 0.0


def test_shot_inference_merit_figures_have_finite_standard_errors() -> None:
    benchmark_case = load_case("h002_chain_0_75.json")
    runner = VqePuccdRunner()

    result = runner.execute_benchmark_algo(
        runner.setup_algo_inputs(benchmark_case),
        backend=SeededSamplingBackend(seed=123),
        shots=128,
        exact_simulation=False,
    )
    merit_figures = runner.compute_merit_figures(result, benchmark_case)

    assert np.isfinite(result["final_energy_se"])
    assert np.isfinite(merit_figures["accuracy_vs_doci_se"])
    assert np.isfinite(merit_figures["score_se"])


def test_single_shot_inference_reports_nan_standard_errors() -> None:
    benchmark_case = load_case("h002_chain_0_75.json")
    runner = VqePuccdRunner()

    with pytest.warns(
        UserWarning,
        match="Standard error is undefined for shot-based runs with fewer than 2 shots",
    ):
        result = runner.execute_benchmark_algo(
            runner.setup_algo_inputs(benchmark_case),
            backend=SeededSamplingBackend(seed=123),
            shots=1,
            exact_simulation=False,
        )

    merit_figures = runner.compute_merit_figures(result, benchmark_case)

    assert np.isnan(result["final_energy_se"])
    assert np.isnan(merit_figures["accuracy_vs_doci_se"])
    assert np.isnan(merit_figures["score_se"])


def test_shot_inference_with_local_qiskit_backend_returns_finite_energy() -> None:
    """Key shot-mode contract test for the bundled local backend."""
    pytest.importorskip("qiskit_aer", reason="qiskit-aer is required for local shot tests")

    benchmark_case = load_case("h002_chain_0_75.json")
    runner = VqePuccdRunner()
    algo_inputs = runner.setup_algo_inputs(benchmark_case)

    result = runner.execute_benchmark_algo(
        algo_inputs,
        backend=QiskitAerSimBackend(method="automatic"),
        shots=128,
        exact_simulation=False,
    )

    assert result["optimizer_success"] is True
    assert np.isfinite(result["final_energy"])
    assert result["final_energy_se"] is not None
    assert np.isfinite(result["final_energy_se"])
    assert result["nfev"] == 1
    assert result["num_circuits_per_eval"] == 3
    assert result["total_circuits"] == 3
    assert result["total_shots"] == 384


def test_get_benchmark_circuits_returns_bound_unmeasured_circuits() -> None:
    benchmark_case = load_case("h002_chain_0_75.json")
    runner = VqePuccdRunner()

    circuits = runner.get_benchmark_circuits(benchmark_case)

    assert len(circuits) > 0
    for circuit in circuits:
        assert circuit.num_qubits == benchmark_case.num_qubits
        assert circuit.num_parameters == 0
        assert_unmeasured(circuit)


def test_prepare_circuits_returns_matching_unmeasured_pairs() -> None:
    benchmark_case = load_case("h002_chain_0_75.json")
    runner = VqePuccdRunner()
    hamiltonian, ansatz = runner.setup_algo_inputs(benchmark_case)
    bound_ansatz = ansatz.assign_parameters(np.zeros(ansatz.num_parameters))

    circuits, observables = runner._prepare_circuits(bound_ansatz, hamiltonian)

    assert len(circuits) > 0
    assert len(circuits) == len(observables)
    for circuit in circuits:
        assert circuit.num_parameters == 0
        assert_unmeasured(circuit)


def test_async_normalized_histograms_match_counts_path() -> None:
    """Async merit-figure path should treat normalized probabilities and raw
    counts identically when shots_per_qc is known."""
    benchmark_case = load_case("h002_chain_0_75.json")
    benchmark_case.data = dict(benchmark_case.data)
    benchmark_case.data["shots_per_qc"] = 1000

    runner = VqePuccdRunner()
    circuits = runner.get_benchmark_circuits(benchmark_case)
    backend = SeededSamplingBackend(seed=777)
    counts, _, _ = backend.run(circuits, shots=1000)

    normalized = []
    for histogram in counts:
        total = sum(histogram.values())
        normalized.append({bitstring: value / total for bitstring, value in histogram.items()})

    counts_metrics = runner.merit_figures_from_measurements(counts, benchmark_case)
    normalized_metrics = runner.merit_figures_from_measurements(normalized, benchmark_case)

    for key in [
        "final_energy",
        "final_energy_se",
        "score",
        "score_se",
        "total_shots",
        "total_circuits",
        "accuracy_vs_doci",
        "accuracy_vs_doci_se",
    ]:
        assert normalized_metrics[key] == pytest.approx(
            counts_metrics[key], rel=0.0, abs=1e-12
        ), key


@pytest.mark.parametrize("exact_simulation", [False, True])
def test_zero_parameter_inference(exact_simulation: bool) -> None:
    """A Hamiltonian with no variational parameters must still produce a valid result."""
    benchmark_case = make_zero_param_case()
    runner = VqePuccdRunner()
    result = runner.execute_benchmark_algo(
        runner.setup_algo_inputs(benchmark_case),
        backend=SeededSamplingBackend(seed=17),
        shots=256,
        exact_simulation=exact_simulation,
        qiskit_primitive_version="v2",
    )

    assert result["mode"] == "inference"
    assert result["nfev"] == 1
    assert result["num_circuits_per_eval"] == 1
    assert result["final_energy"] == pytest.approx(-0.15, abs=1e-14)
    assert np.isfinite(result["final_energy_se"])


def test_compute_merit_figures_labels_shot_based_score_as_unbounded(capsys) -> None:
    benchmark_case = load_case("h002_chain_0_75.json")
    runner = VqePuccdRunner()

    merit_figures = runner.compute_merit_figures(
        {
            "final_energy": benchmark_case.data["reference_energy_doci"] - 0.001,
            "final_energy_se": 0.002,
            "shots_per_eval": 1000,
            "num_circuits_per_eval": 3,
            "total_circuits": 3,
            "total_shots": 3000,
            "mode": "inference",
            "simulation_mode": "shot-based_sampling",
        },
        benchmark_case,
    )

    captured = capsys.readouterr().out
    assert "Correlation Energy Ratio (Ec_VQE / Ec_DOCI)" in captured
    assert "unbounded under shot noise" in captured
    assert np.isfinite(merit_figures["score"])


def test_compute_merit_figures_keeps_exact_mode_fraction_label(capsys) -> None:
    benchmark_case = load_case("h002_chain_0_75.json")
    runner = VqePuccdRunner()

    runner.compute_merit_figures(
        {
            "final_energy": benchmark_case.data["reference_energy_doci"],
            "final_energy_se": 0.0,
            "shots_per_eval": 0,
            "num_circuits_per_eval": 1,
            "total_circuits": 1,
            "total_shots": 0,
            "mode": "inference",
            "simulation_mode": "exact_statevector",
            "primitive_version": "v2",
        },
        benchmark_case,
    )

    captured = capsys.readouterr().out
    assert "Fraction Correlation Energy Captured" in captured
    assert "unbounded under shot noise" not in captured
