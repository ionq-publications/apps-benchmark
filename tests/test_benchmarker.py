"""
Tests for the interactive Benchmarker orchestration layer.
"""

from pathlib import Path

import pandas as pd
import pytest
from apps_benchmark.backends.mock_backend import MockBackend
from apps_benchmark.benchmarker import (
    ALGORITHM_COLUMNS,
    BENCHMARK_CASE_COLUMNS,
    STATS_COLUMNS,
    SUBMISSION_COLUMNS,
    Benchmarker,
)
from apps_benchmark.core.backend import AbstractAsyncBackend, JobData, JobStatus, MeasurementBatch
from apps_benchmark.errors import BackendValidationError, BenchmarkError, OpenBenchmarkError
from apps_benchmark.primitives.benchmark_case import BenchmarkCase
from qiskit import QuantumCircuit


class InstantAsyncBackend(AbstractAsyncBackend):
    """Small async backend that completes submitted jobs immediately."""

    def __init__(self, status: JobStatus = JobStatus.DONE) -> None:
        self.status = status
        self.jobs: dict[str, JobData] = {}

    def name(self) -> str:
        return "instant_async"

    def submit(
        self,
        circuits: list[QuantumCircuit],
        shots: int = 1000,
        job_name: str | None = None,
    ) -> tuple[str, JobData]:
        job_id = f"instant_job_{len(self.jobs)}"
        job_data = self.serialize_job_data(circuits, shots, job_name or "")
        job_data["num_qubits"] = [circuit.num_qubits for circuit in circuits]
        self.jobs[job_id] = job_data
        return job_id, job_data

    def job_status(self, job_id: str) -> JobStatus:
        return self.status

    def retrieve_results(
        self,
        job_id: str,
        job_data: JobData,
    ) -> tuple[MeasurementBatch, pd.Timestamp]:
        shots = int(job_data["shots"])
        num_qubits = [int(value) for value in job_data["num_qubits"]]
        measurements = [{"0" * nqubits: shots} for nqubits in num_qubits]
        return measurements, pd.Timestamp.now(tz="UTC")


@pytest.fixture
def benchmarker() -> Benchmarker:
    return Benchmarker(include_diy=False)


@pytest.fixture
def qft_case_frame(benchmarker: Benchmarker) -> pd.DataFrame:
    return benchmarker.get_benchmark_cases(
        benchmark_category="qft",
        max_qubits=10,
        runnable_only=True,
    ).head(1)


def test_get_benchmark_cases_returns_expected_schema_and_filters(
    benchmarker: Benchmarker,
) -> None:
    cases = benchmarker.get_benchmark_cases(
        benchmark_category="qft",
        instance_id="f75ae75f",
        solution_algorithm="cosine_qft",
        min_qubits=10,
        max_qubits=10,
    )

    assert list(cases.columns) == BENCHMARK_CASE_COLUMNS
    assert len(cases) == 1
    row = cases.iloc[0]
    assert row["benchmark_category"] == "qft"
    assert row["instance_name"] == "10_qubit_challenge"
    assert row["solution_algorithms"] == ["cosine_qft", "hidden_phase_qft"]
    assert isinstance(row["file"], Path)


def test_get_benchmark_cases_runnable_only_drops_open_only_cases(
    benchmarker: Benchmarker,
) -> None:
    all_cases = benchmarker.get_benchmark_cases(benchmark_category="high_energy_physics")
    runnable_cases = benchmarker.get_benchmark_cases(
        benchmark_category="high_energy_physics",
        runnable_only=True,
    )

    assert not all_cases.empty
    assert runnable_cases.empty


def test_get_algorithms_reports_shipped_and_open_statuses(benchmarker: Benchmarker) -> None:
    qft_algorithms = benchmarker.get_algorithms("qft")
    hep_algorithms = benchmarker.get_algorithms("high_energy_physics")
    closed_hep = benchmarker.get_algorithms("high_energy_physics", include_open=False)

    assert list(qft_algorithms.columns) == ALGORITHM_COLUMNS
    assert set(qft_algorithms["status"]) == {"shipped"}
    assert set(hep_algorithms["status"]) == {"open"}
    assert closed_hep.empty


def test_get_algorithms_fails_closed_for_unregistered_algorithms(
    benchmarker: Benchmarker,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    case_path = tmp_path / "unregistered_case.json"
    BenchmarkCase(
        benchmark_category="example",
        problem_type="demo",
        instance_name="unregistered_case",
        instance_id="case_001",
        num_qubits=2,
        solution_algorithms=["missing_runner"],
        data={},
    ).dump(case_path)
    monkeypatch.setattr(
        benchmarker,
        "_case_entries",
        lambda: [("example", {"file": case_path, "builtin": True})],
    )

    with pytest.raises(BenchmarkError, match="no registered runners"):
        benchmarker.get_algorithms("example")


def test_run_benchmarks_accepts_backend_instances_and_names(
    benchmarker: Benchmarker,
    qft_case_frame: pd.DataFrame,
) -> None:
    by_instance = benchmarker.run_benchmarks(
        qft_case_frame,
        MockBackend(),
        shots=8,
        solution_algorithms="cosine_qft",
    )
    by_name = benchmarker.run_benchmarks(
        qft_case_frame,
        "mock_backend",
        shots=8,
        solution_algorithms="cosine_qft",
    )

    assert by_instance.iloc[0]["status"] == "done"
    assert by_instance.iloc[0]["backend"] == "mock"
    assert by_instance.iloc[0]["shots_per_qc"] == 8
    assert by_instance.iloc[0]["solution_algorithm"] == "cosine_qft"
    assert by_name.iloc[0]["backend"] == "mock"


def test_run_benchmarks_raises_for_open_benchmark_cases(benchmarker: Benchmarker) -> None:
    open_cases = benchmarker.get_benchmark_cases("high_energy_physics").head(1)

    with pytest.raises(OpenBenchmarkError, match="open benchmark"):
        benchmarker.run_benchmarks(open_cases, MockBackend(), shots=4)


def test_submit_benchmarks_requires_async_backend(
    benchmarker: Benchmarker,
    qft_case_frame: pd.DataFrame,
    tmp_path: Path,
) -> None:
    with pytest.raises(BackendValidationError, match="async submission"):
        benchmarker.submit_benchmarks(
            qft_case_frame,
            MockBackend(),
            record_db=tmp_path / "record.json",
            shots=8,
            solution_algorithms="cosine_qft",
        )


def test_submit_benchmarks_requires_record_db(
    benchmarker: Benchmarker,
    qft_case_frame: pd.DataFrame,
) -> None:
    with pytest.raises(TypeError, match="record_db"):
        benchmarker.submit_benchmarks(
            qft_case_frame,
            InstantAsyncBackend(),
            shots=8,
            solution_algorithms="cosine_qft",
        )


def test_submit_load_and_update_submission_record(
    benchmarker: Benchmarker,
    qft_case_frame: pd.DataFrame,
    tmp_path: Path,
) -> None:
    record_db = tmp_path / "benchmark-record.json"
    backend = InstantAsyncBackend()

    submitted = benchmarker.submit_benchmarks(
        qft_case_frame,
        backend,
        shots=8,
        record_db=record_db,
        solution_algorithms="cosine_qft",
    )
    loaded = benchmarker.load_submission_record(record_db)
    updated = benchmarker.update_submission_record(record_db, backend)
    reloaded = benchmarker.load_submission_record(record_db)

    assert list(submitted.columns) == SUBMISSION_COLUMNS
    assert record_db.exists()
    assert loaded.iloc[0]["status"] == "submitted"
    assert pd.api.types.is_datetime64_any_dtype(loaded["start_time"])
    assert updated.iloc[0]["status"] == "done"
    assert updated.iloc[0]["backend"] == "instant_async"
    assert updated.iloc[0]["measurements"] == [{"0" * 10: 8}]
    assert isinstance(updated.iloc[0]["problem_specific_data"], dict)
    assert not pd.isna(updated.iloc[0]["end_time"])
    assert reloaded.iloc[0]["status"] == "done"


def test_update_submission_record_leaves_running_jobs_submitted(
    benchmarker: Benchmarker,
    qft_case_frame: pd.DataFrame,
    tmp_path: Path,
) -> None:
    record_db = tmp_path / "benchmark-record.json"
    backend = InstantAsyncBackend(status=JobStatus.RUNNING)

    benchmarker.submit_benchmarks(
        qft_case_frame,
        backend,
        shots=8,
        record_db=record_db,
        solution_algorithms="cosine_qft",
    )
    updated = benchmarker.update_submission_record(record_db, backend)

    assert updated.iloc[0]["status"] == "running"
    assert pd.isna(updated.iloc[0]["end_time"])
    assert updated.iloc[0]["measurements"] == []


def test_get_benchmark_circuit_stats_returns_stats_frame(
    benchmarker: Benchmarker,
    qft_case_frame: pd.DataFrame,
) -> None:
    stats = benchmarker.get_benchmark_circuit_stats(
        qft_case_frame,
        solution_algorithms="cosine_qft",
    )

    assert list(stats.columns) == STATS_COLUMNS
    assert stats.iloc[0]["solution_algorithm"] == "cosine_qft"
    assert stats.iloc[0]["num_circuits"] == 1
    assert len(stats.iloc[0]["num_1q_gates"]) == 1
