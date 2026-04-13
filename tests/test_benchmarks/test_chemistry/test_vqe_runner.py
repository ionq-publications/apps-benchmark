"""
Tests for VQE PUCCD runner.

This module tests the VQE benchmark runner implementation.

Copyright (c) 2025 IonQ, Inc. All rights reserved.
Licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0
International (CC BY-NC-ND 4.0) License.
For details, visit: https://creativecommons.org/licenses/by-nc-nd/4.0/
"""

import copy
from pathlib import Path

import numpy as np
import pytest
from apps_benchmark.backends.mock_backend import MockBackend
from apps_benchmark.benchmarks.chemistry.algorithms.vqe_puccd_runner import VqePuccdRunner
from apps_benchmark.core.qc_benchmark_runner import QCBenchmarkSubmissionRecord
from apps_benchmark.primitives.benchmark_case import BenchmarkCase
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from .conftest import assert_unmeasured


class TestVqePuccdRunner:
    """Tests for VQE PUCCD algorithm runner."""

    @pytest.fixture
    def runner(self) -> VqePuccdRunner:
        """Create VQE runner instance."""
        return VqePuccdRunner()

    @pytest.fixture
    def benchmark_case_inference(self) -> BenchmarkCase:
        """Load the inference-capable chemistry benchmark case for testing."""
        problem_path = (
            Path(__file__).parent.parent.parent.parent
            / "apps_benchmark"
            / "benchmarks"
            / "chemistry"
            / "benchmark_cases"
            / "h002_chain_0_75.json"
        )
        return BenchmarkCase.load_from_database(problem_path)

    @pytest.fixture
    def benchmark_case_without_optimal_parameters(
        self,
        benchmark_case_inference: BenchmarkCase,
    ) -> BenchmarkCase:
        """Create a copy of the benchmark case without inference parameters."""
        benchmark_case = copy.deepcopy(benchmark_case_inference)
        benchmark_case.data.pop("optimal_parameters", None)
        return benchmark_case

    def test_runner_name(self, runner: VqePuccdRunner) -> None:
        """Test that runner returns correct name."""
        assert runner.name() == "vqe_puccd"

    def test_runner_benchmark_category(self, runner: VqePuccdRunner) -> None:
        """Test that runner identifies correct benchmark category."""
        assert runner.benchmark_category == "chemistry"

    def test_benchmark_case_inference_contains_optimal_parameters(
        self,
        benchmark_case_inference: BenchmarkCase,
    ) -> None:
        """Test the inference benchmark case includes optimal parameters."""
        assert "optimal_parameters" in benchmark_case_inference.data
        assert isinstance(benchmark_case_inference.data["optimal_parameters"], list)
        assert len(benchmark_case_inference.data["optimal_parameters"]) == 1

    def test_get_benchmark_circuits_returns_bound_circuits(
        self,
        runner: VqePuccdRunner,
        benchmark_case_inference: BenchmarkCase,
    ) -> None:
        """Test QC-runner contract returns parameter-bound circuits for inference mode."""
        circuits = runner.get_benchmark_circuits(benchmark_case_inference)

        assert len(circuits) > 0
        for circuit in circuits:
            assert isinstance(circuit, QuantumCircuit)
            assert circuit.num_qubits == benchmark_case_inference.num_qubits
            assert circuit.num_parameters == 0
            assert_unmeasured(circuit)

    def test_get_benchmark_circuits_requires_optimal_parameters(
        self,
        runner: VqePuccdRunner,
        benchmark_case_without_optimal_parameters: BenchmarkCase,
    ) -> None:
        """Test inference circuit generation fails without precomputed parameters."""
        with pytest.raises(ValueError, match="only supported in inference mode"):
            runner.get_benchmark_circuits(benchmark_case_without_optimal_parameters)

    def test_build_ansatz_creates_valid_circuit(self, runner: VqePuccdRunner) -> None:
        """Test that _build_ansatz creates a valid circuit."""
        ansatz = runner._build_ansatz(num_qubits=2, num_occ_pairs=1)

        assert ansatz.num_qubits == 2
        assert ansatz.num_parameters == 1

    def test_prepare_circuits_returns_circuit_observable_pairs(
        self,
        runner: VqePuccdRunner,
        benchmark_case_inference: BenchmarkCase,
    ) -> None:
        """Test helper returns one observable for each prepared circuit."""
        hamiltonian, ansatz = runner.setup_algo_inputs(benchmark_case_inference)
        bound_ansatz = ansatz.assign_parameters(np.zeros(ansatz.num_parameters))

        circuits, observables = runner._prepare_circuits(bound_ansatz, hamiltonian)

        assert len(circuits) > 0
        assert len(circuits) == len(observables)
        assert all(isinstance(observable, SparsePauliOp) for observable in observables)
        for circuit in circuits:
            assert_unmeasured(circuit)

    def test_compute_energy_from_results_returns_energy_and_standard_error(
        self,
        runner: VqePuccdRunner,
    ) -> None:
        """Test energy helper returns both an energy and standard error."""
        mock_results: list[dict[str, int | float]] = [{"00": 1000}]
        mock_observables = [SparsePauliOp(["ZZ"], coeffs=[1.0])]

        energy, standard_error = runner._compute_energy_from_results(
            mock_results,
            mock_observables,
            num_qubits=2,
            shots=1000,
        )

        assert isinstance(energy, float)
        assert isinstance(standard_error, float)
        assert not np.isnan(energy)
        assert standard_error >= 0.0

    def test_merit_figures_from_measurements_returns_required_fields(
        self,
        runner: VqePuccdRunner,
        benchmark_case_inference: BenchmarkCase,
    ) -> None:
        """Test async-compatible merit computation returns QC-runner fields."""
        circuits = runner.get_benchmark_circuits(benchmark_case_inference)
        measurements, _, _ = MockBackend(deterministic=True).run(circuits, shots=100)

        merit_figures = runner.merit_figures_from_measurements(
            measurements,
            benchmark_case_inference,
        )

        assert "score" in merit_figures
        assert "total_shots" in merit_figures
        assert "final_energy" in merit_figures
        assert "reference_doci" in merit_figures
        assert "reference_fci" in merit_figures
        assert "mode" in merit_figures
        assert "simulation_mode" in merit_figures
        assert merit_figures["total_shots"] == 100 * len(measurements)
        assert merit_figures["mode"] == "inference_async"
        assert merit_figures["simulation_mode"] == "shot-based_sampling"

    def test_merit_figures_from_measurements_requires_optimal_parameters(
        self,
        runner: VqePuccdRunner,
        benchmark_case_without_optimal_parameters: BenchmarkCase,
    ) -> None:
        """Test async-compatible merit computation requires inference parameters."""
        with pytest.raises(ValueError, match="requires 'optimal_parameters'"):
            runner.merit_figures_from_measurements(
                [],
                benchmark_case_without_optimal_parameters,
            )

    def test_run_benchmark_returns_qc_submission_record(
        self,
        runner: VqePuccdRunner,
        benchmark_case_inference: BenchmarkCase,
    ) -> None:
        """Test full QC-runner integration returns the enriched submission record."""
        record = runner.run_benchmark(
            benchmark_case_inference,
            MockBackend(deterministic=True),
            shots=100,
        )

        assert isinstance(record, QCBenchmarkSubmissionRecord)
        assert record.benchmark_category == "chemistry"
        assert record.problem_type == "hydrogen_lattice_vqe"
        assert record.instance_name == "h002_chain_0_75"
        assert record.solution_algorithm == "vqe_puccd"
        assert record.num_qubits == 2
        assert record.backend == "mock"
        assert record.shots_per_qc == 100
        assert record.total_shots > 0
        assert record.status == "done"
        assert isinstance(record.score, float)
        assert len(record.measurements) > 0
        assert isinstance(record.job_id, str)
        assert isinstance(record.job_data, dict)

    def test_ansatz_has_correct_structure(self, runner: VqePuccdRunner) -> None:
        """Test that ansatz has expected gate structure."""
        ansatz = runner._build_ansatz(num_qubits=2, num_occ_pairs=1)
        ops = ansatz.count_ops()
        x_instructions = [
            instruction for instruction in ansatz.data if instruction.operation.name == "x"
        ]
        cx_qubits = [
            tuple(ansatz.find_bit(qubit).index for qubit in instruction.qubits)
            for instruction in ansatz.data
            if instruction.operation.name == "cx"
        ]
        ry_instructions = [
            instruction for instruction in ansatz.data if instruction.operation.name == "ry"
        ]

        assert ansatz.num_parameters == 1
        assert ops["x"] == 1
        assert ops["barrier"] == 2
        assert ops["s"] == 2
        assert ops["h"] == 2
        assert ops["cx"] == 2
        assert ops["ry"] == 2
        assert ops["sdg"] == 2
        assert ansatz.find_bit(x_instructions[0].qubits[0]).index == 0
        assert cx_qubits == [(1, 0), (1, 0)]
        assert len(ry_instructions) == 2
        assert {
            ansatz.find_bit(instruction.qubits[0]).index for instruction in ry_instructions
        } == {0, 1}
        assert {instruction.operation.params[0] for instruction in ry_instructions} == set(
            ansatz.parameters
        )

    def test_different_num_qubits(self, runner: VqePuccdRunner) -> None:
        """Test ansatz building with different qubit counts."""
        ansatz_4q = runner._build_ansatz(num_qubits=4, num_occ_pairs=2)
        assert ansatz_4q.num_qubits == 4
        assert ansatz_4q.num_parameters == 4

        ansatz_6q = runner._build_ansatz(num_qubits=6, num_occ_pairs=2)
        assert ansatz_6q.num_qubits == 6
        assert ansatz_6q.num_parameters == 8
