"""
Tests for CircuitBenchmarkRunner behavior and records.

Copyright (c) 2025 IonQ, Inc. All rights reserved.
Licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0
International (CC BY-NC-ND 4.0) License.
For details, visit: https://creativecommons.org/licenses/by-nc-nd/4.0/
"""

from dataclasses import asdict

import pandas as pd
import pytest
from apps_benchmark.core.backend import AbstractAsyncBackend, AbstractBackend, JobStatus
from apps_benchmark.core.qc_benchmark_runner import (
    BaselineScore,
    CircuitBenchmarkRunner,
    CircuitStats,
    QCBenchmarkSubmissionRecord,
)
from apps_benchmark.primitives.benchmark_case import BenchmarkCase
from qiskit import QuantumCircuit


class ConcreteSyncBackend(AbstractBackend):
    """Minimal synchronous backend for QC runner tests."""

    def name(self) -> str:
        return "test_sync_backend"

    def run(
        self,
        circuits: list[QuantumCircuit],
        shots: int = 1000,
        job_name: str | None = None,
    ) -> tuple[list[dict], str, dict]:
        results = [{"00": shots // 2, "11": shots - (shots // 2)} for _ in circuits]
        job_id = "sync_job_123"
        job_data = self.serialize_job_data(circuits, shots, job_name or "")
        return results, job_id, job_data


class ConcreteAsyncBackend(AbstractAsyncBackend):
    """Minimal asynchronous backend for QC runner tests."""

    def __init__(self) -> None:
        self._jobs: dict[str, dict] = {}

    def name(self) -> str:
        return "test_async_backend"

    def submit(
        self,
        circuits: list[QuantumCircuit],
        shots: int = 1000,
        job_name: str | None = None,
    ) -> tuple[str, dict]:
        job_id = f"async_job_{len(self._jobs)}"
        job_data = self.serialize_job_data(circuits, shots, job_name or "")
        self._jobs[job_id] = {
            "status": JobStatus.QUEUED,
            "circuits": circuits,
            "shots": shots,
        }
        return job_id, job_data

    def job_status(self, job_id: str) -> JobStatus:
        return self._jobs[job_id]["status"]

    def retrieve_results(
        self,
        job_id: str,
        job_data: dict,
    ) -> tuple[list[dict], pd.Timestamp]:
        shots = int(job_data["shots"])
        results = [{"00": shots // 2, "11": shots - (shots // 2)}]
        return results, pd.Timestamp.now(tz="UTC")


class ConcreteQCRunner(CircuitBenchmarkRunner):
    """Concrete QC runner used to exercise inherited contract behavior."""

    def name(self) -> str:
        return "concrete_qc_runner"

    def get_benchmark_circuits(
        self,
        benchmark_case: BenchmarkCase,
    ) -> list[QuantumCircuit]:
        circuit = QuantumCircuit(benchmark_case.num_qubits, benchmark_case.num_qubits)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.measure([0, 1], [0, 1])
        return [circuit]

    def merit_figures_from_measurements(
        self,
        measurements: list[dict[str, float]],
        benchmark_case: BenchmarkCase,
    ) -> dict[str, float]:
        shots = sum(measurements[0].values()) if (measurements and measurements[0]) else 0
        score = measurements[0].get("00", 0) / shots if shots else 0.0
        return {
            "score": score,
            "dominant_state_probability": score,
        }


@pytest.fixture
def benchmark_case() -> BenchmarkCase:
    """Create a simple benchmark case for QC runner tests."""
    return BenchmarkCase(
        benchmark_category="test_category",
        problem_type="test_problem",
        instance_name="bell_pair",
        instance_id="qc_case_001",
        num_qubits=2,
        solution_algorithms=["concrete_qc_runner"],
        data={},
    )


@pytest.fixture
def runner() -> ConcreteQCRunner:
    """Create a concrete QC runner instance."""
    return ConcreteQCRunner()


@pytest.fixture
def async_backend() -> ConcreteAsyncBackend:
    """Create a concrete async backend instance."""
    return ConcreteAsyncBackend()


class TestQCBenchmarkSubmissionRecord:
    """Tests for QC benchmark submission record validation."""

    def test_job_data_must_be_json_serializable(self):
        """Test that non-serializable job data raises a clear ValueError."""
        start = pd.Timestamp.now(tz="UTC")

        with pytest.raises(ValueError, match="job_data must be JSON-serializable"):
            QCBenchmarkSubmissionRecord(
                benchmark_category="chemistry",
                problem_type="vqe",
                instance_name="h2",
                instance_id="abc123",
                solution_algorithm="vqe_puccd",
                num_qubits=2,
                backend="test_backend",
                shots_per_qc=1000,
                total_shots=1000,
                start_time=start,
                end_time=pd.NaT,
                time_to_soln=pd.NaT,
                adjusted_tts=pd.NaT,
                last_retrieval=start,
                status="submitted",
                score=float("nan"),
                problem_specific_data={},
                job_id="job_123",
                job_data={"bad": object()},
            )

    def test_job_data_serializable_record_creation(self):
        """Test that JSON-serializable job data is accepted."""
        start = pd.Timestamp.now(tz="UTC")

        record = QCBenchmarkSubmissionRecord(
            benchmark_category="chemistry",
            problem_type="vqe",
            instance_name="h2",
            instance_id="abc123",
            solution_algorithm="vqe_puccd",
            num_qubits=2,
            backend="test_backend",
            shots_per_qc=1000,
            total_shots=1000,
            start_time=start,
            end_time=pd.NaT,
            time_to_soln=pd.NaT,
            adjusted_tts=pd.NaT,
            last_retrieval=start,
            status="submitted",
            score=float("nan"),
            problem_specific_data={},
            job_id="job_123",
            job_data={"circuits": ["OPENQASM 3.0;"], "shots": 1000},
        )

        assert asdict(record)["job_id"] == "job_123"


class TestCircuitBenchmarkRunnerBehavior:
    """Tests for inherited CircuitBenchmarkRunner behavior."""

    def test_setup_algo_inputs_returns_circuits_and_instance_name(
        self,
        runner: ConcreteQCRunner,
        benchmark_case: BenchmarkCase,
    ) -> None:
        """Test QC runner setup returns generated circuits and instance name."""
        circuits, instance_name = runner.setup_algo_inputs(benchmark_case)

        assert isinstance(circuits, list)
        assert len(circuits) == 1
        assert isinstance(circuits[0], QuantumCircuit)
        assert instance_name == benchmark_case.instance_name

    def test_compute_merit_figures_adds_total_shots_and_job_metadata(
        self,
        runner: ConcreteQCRunner,
        benchmark_case: BenchmarkCase,
    ) -> None:
        """Test inherited compute_merit_figures adds QC-runner metadata."""
        runner._shots = 100
        algo_output = (
            [{"00": 60, "11": 40}],
            "job_123",
            {"shots": 100, "job_name": "qc_test"},
        )

        merit_figures = runner.compute_merit_figures(algo_output, benchmark_case)

        assert merit_figures["total_shots"] == 100
        assert merit_figures["benchmark_qc_hist"] == [{"00": 60, "11": 40}]
        assert merit_figures["job_id"] == "job_123"
        assert merit_figures["job_data"] == {"shots": 100, "job_name": "qc_test"}
        assert "score" in merit_figures

    def test_run_benchmark_returns_qc_submission_record(
        self,
        runner: ConcreteQCRunner,
        benchmark_case: BenchmarkCase,
    ) -> None:
        """Test inherited run_benchmark returns a QC submission record."""
        record = runner.run_benchmark(
            benchmark_case,
            ConcreteSyncBackend(),
            shots=100,
        )

        assert isinstance(record, QCBenchmarkSubmissionRecord)
        assert record.status == "done"
        assert record.backend == "test_sync_backend"
        assert record.total_shots == 100
        assert len(record.measurements) == 1
        assert record.job_id == "sync_job_123"
        assert isinstance(record.job_data, dict)

    def test_submit_benchmark_circuits_returns_submitted_record(
        self,
        runner: ConcreteQCRunner,
        benchmark_case: BenchmarkCase,
        async_backend: ConcreteAsyncBackend,
    ) -> None:
        """Test async submission returns a submitted QC record."""
        record = runner.submit_benchmark_circuits(
            benchmark_case,
            async_backend,
            shots=250,
        )

        assert isinstance(record, QCBenchmarkSubmissionRecord)
        assert record.status == "submitted"
        assert record.backend == "test_async_backend"
        assert record.total_shots == 250
        assert record.job_id.startswith("async_job_")
        assert isinstance(record.job_data, dict)

    def test_get_circuit_stats_returns_gate_counts(
        self,
        runner: ConcreteQCRunner,
        benchmark_case: BenchmarkCase,
    ) -> None:
        """Test inherited circuit stats helper returns per-circuit counts."""
        stats = runner.get_circuit_stats(benchmark_case)

        assert isinstance(stats, CircuitStats)
        assert stats.num_circuits == 1
        assert len(stats.num_1q_gates) == 1
        assert len(stats.num_2q_gates) == 1
        assert stats.num_2q_gates[0] > 0

    def test_est_rnd_baseline_score_returns_summary_record(
        self,
        runner: ConcreteQCRunner,
        benchmark_case: BenchmarkCase,
    ) -> None:
        """Test random baseline helper returns a populated summary record."""
        baseline = runner.est_rnd_baseline_score(
            benchmark_case,
            shots=10,
            num_draws=5,
        )

        assert isinstance(baseline, BaselineScore)
        assert baseline.solution_algorithm == "concrete_qc_runner"
        assert baseline.num_draws == 5
        assert baseline.shots_per_draw == 10
        assert 0.0 <= baseline.mean <= 1.0
