"""
Tests for abstract backend interfaces.

This module tests the AbstractBackend and AbstractAsyncBackend classes.

Copyright (c) 2025 IonQ, Inc. All rights reserved.
Licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0
International (CC BY-NC-ND 4.0) License.
For details, visit: https://creativecommons.org/licenses/by-nc-nd/4.0/
"""

import pandas as pd
import pytest
import qiskit.qasm2 as qasm2
from apps_benchmark.core.backend import (
    AbstractAsyncBackend,
    AbstractBackend,
    JobStatus,
)
from apps_benchmark.errors import BackendError
from qiskit import QuantumCircuit


class ConcreteBackend(AbstractBackend):
    """
    Minimal concrete backend for testing.
    """

    def name(self) -> str:
        return "test_backend"

    def run(
        self,
        circuits: list[QuantumCircuit],
        shots: int = 1000,
        job_name: str | None = None,
    ) -> tuple[list[dict], str, dict]:
        # Simple deterministic results
        results = [{"00": shots // 2, "11": shots // 2} for _ in circuits]
        job_id = "test_job_123"
        job_data = self.serialize_job_data(circuits, shots, job_name or "")
        return results, job_id, job_data


class ConcreteAsyncBackend(AbstractAsyncBackend):
    """
    Minimal async backend for testing.
    """

    def __init__(self):
        self._jobs = {}

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

        # Store job info
        self._jobs[job_id] = {
            "status": JobStatus.QUEUED,
            "circuits": circuits,
            "shots": shots,
            "results": None,
        }

        return job_id, job_data

    def job_status(self, job_id: str) -> JobStatus:
        if job_id not in self._jobs:
            raise BackendError(f"Job {job_id} not found")
        return self._jobs[job_id]["status"]

    def retrieve_results(self, job_id: str, job_data: dict) -> tuple[list[dict], pd.Timestamp]:
        if job_id not in self._jobs:
            raise BackendError(f"Job {job_id} not found")

        job = self._jobs[job_id]
        if job["status"] != JobStatus.DONE:
            raise BackendError(f"Job {job_id} is not complete")

        results = job["results"]
        completion_time = pd.Timestamp.now(tz="UTC")
        return results, completion_time

    def mark_done(self, job_id: str, results: list[dict]) -> None:
        """Helper method for testing - mark job as done."""
        if job_id in self._jobs:
            self._jobs[job_id]["status"] = JobStatus.DONE
            self._jobs[job_id]["results"] = results


class TestJobStatus:
    """Tests for JobStatus enum."""

    def test_job_status_values(self):
        """Test that JobStatus enum has expected values."""
        assert JobStatus.SUBMITTED.value == "submitted"
        assert JobStatus.QUEUED.value == "queued"
        assert JobStatus.RUNNING.value == "running"
        assert JobStatus.DONE.value == "done"
        assert JobStatus.FAILED.value == "failed"

    def test_job_status_comparison(self):
        """Test JobStatus equality comparison."""
        assert JobStatus.DONE == JobStatus.DONE
        assert JobStatus.DONE != JobStatus.FAILED


class TestAbstractBackend:
    """Tests for AbstractBackend class."""

    def test_concrete_backend_instantiation(self):
        """Test that a concrete backend can be instantiated."""
        backend = ConcreteBackend()
        assert isinstance(backend, AbstractBackend)

    def test_backend_name(self):
        """Test backend name method."""
        backend = ConcreteBackend()
        assert backend.name() == "test_backend"

    def test_backend_run(self):
        """Test backend run method returns correct structure."""
        backend = ConcreteBackend()
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        results, job_id, job_data = backend.run([qc], shots=1000)

        # Check return types
        assert isinstance(results, list)
        assert isinstance(job_id, str)
        assert isinstance(job_data, dict)

        # Check results structure
        assert len(results) == 1
        assert isinstance(results[0], dict)
        assert "00" in results[0] or "11" in results[0]

        # Check job_data structure
        assert "circuits" in job_data
        assert "shots" in job_data
        assert job_data["shots"] == 1000

    def test_backend_run_multiple_circuits(self):
        """Test running multiple circuits."""
        backend = ConcreteBackend()
        circuits = [QuantumCircuit(2) for _ in range(3)]

        results, job_id, job_data = backend.run(circuits, shots=500)

        assert len(results) == 3
        assert len(job_data["circuits"]) == 3

    def test_serialize_job_data(self):
        """Test job data serialization."""
        backend = ConcreteBackend()
        qc = QuantumCircuit(2)
        qc.h(0)

        job_data = backend.serialize_job_data([qc], 1000, "test_job")

        assert "circuits" in job_data
        assert "shots" in job_data
        assert "job_name" in job_data
        assert job_data["shots"] == 1000
        assert job_data["job_name"] == "test_job"
        assert isinstance(job_data["circuits"], list)
        assert isinstance(job_data["circuits"][0], str)  # QASM string
        assert job_data["circuits"][0].lstrip().startswith("OPENQASM 3")

    def test_deserialize_job_data(self):
        """Test job data deserialization."""
        backend = ConcreteBackend()
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        # Serialize then deserialize
        job_data = backend.serialize_job_data([qc], 1000, "test_job")
        hydrated = backend.de_serialize_job_data(job_data)

        assert "circuits" in hydrated
        assert "shots" in hydrated
        assert "job_name" in hydrated
        assert hydrated["shots"] == 1000
        assert hydrated["job_name"] == "test_job"
        assert isinstance(hydrated["circuits"], list)
        assert isinstance(hydrated["circuits"][0], QuantumCircuit)

    def test_deserialize_legacy_qasm2_job_data(self):
        """Test job data deserialization remains compatible with QASM2."""
        backend = ConcreteBackend()
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        job_data = {
            "circuits": [qasm2.dumps(qc)],
            "shots": 1000,
            "job_name": "legacy_job",
        }
        hydrated = backend.de_serialize_job_data(job_data)

        assert hydrated["shots"] == 1000
        assert hydrated["job_name"] == "legacy_job"
        assert isinstance(hydrated["circuits"][0], QuantumCircuit)

    def test_validate_connection_default(self):
        """Test default validate_connection returns True."""
        backend = ConcreteBackend()
        assert backend.validate_connection() is True

    def test_cannot_instantiate_abstract_backend(self):
        """Test that AbstractBackend cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            AbstractBackend()


class TestAbstractAsyncBackend:
    """Tests for AbstractAsyncBackend class."""

    def test_concrete_async_backend_instantiation(self):
        """Test that a concrete async backend can be instantiated."""
        backend = ConcreteAsyncBackend()
        assert isinstance(backend, AbstractAsyncBackend)
        assert isinstance(backend, AbstractBackend)

    def test_async_backend_submit(self):
        """Test async backend submit method."""
        backend = ConcreteAsyncBackend()
        qc = QuantumCircuit(2)
        qc.h(0)

        job_id, job_data = backend.submit([qc], shots=1000, job_name="test")

        assert isinstance(job_id, str)
        assert isinstance(job_data, dict)
        assert "circuits" in job_data
        assert "shots" in job_data

    def test_async_backend_job_status(self):
        """Test job status tracking."""
        backend = ConcreteAsyncBackend()
        qc = QuantumCircuit(2)

        job_id, job_data = backend.submit([qc], shots=1000)

        # Initially queued
        status = backend.job_status(job_id)
        assert status == JobStatus.QUEUED

        # Mark as done
        backend.mark_done(job_id, [{"00": 500, "11": 500}])
        status = backend.job_status(job_id)
        assert status == JobStatus.DONE

    def test_async_backend_job_status_not_found(self):
        """Test job_status raises error for unknown job."""
        backend = ConcreteAsyncBackend()

        with pytest.raises(BackendError, match="Job unknown_job not found"):
            backend.job_status("unknown_job")

    def test_async_backend_retrieve_results(self):
        """Test retrieving results from completed job."""
        backend = ConcreteAsyncBackend()
        qc = QuantumCircuit(2)

        job_id, job_data = backend.submit([qc], shots=1000)
        backend.mark_done(job_id, [{"00": 500, "11": 500}])

        results, completion_time = backend.retrieve_results(job_id, job_data)

        assert isinstance(results, list)
        assert len(results) == 1
        assert results[0] == {"00": 500, "11": 500}
        assert isinstance(completion_time, pd.Timestamp)
        assert completion_time.tz is not None  # Has timezone

    def test_async_backend_retrieve_results_not_complete(self):
        """Test retrieving results from incomplete job raises error."""
        backend = ConcreteAsyncBackend()
        qc = QuantumCircuit(2)

        job_id, job_data = backend.submit([qc], shots=1000)

        with pytest.raises(BackendError, match="is not complete"):
            backend.retrieve_results(job_id, job_data)

    def test_async_backend_run_method(self):
        """Test that async backend run method works end-to-end."""

        class QuickAsyncBackend(ConcreteAsyncBackend):
            """Backend that completes jobs immediately."""

            def submit(self, circuits, shots=1000, job_name=None):
                job_id, job_data = super().submit(circuits, shots, job_name)
                # Immediately mark as done
                results = [{"00": shots // 2, "11": shots // 2} for _ in circuits]
                self.mark_done(job_id, results)
                return job_id, job_data

        backend = QuickAsyncBackend()
        qc = QuantumCircuit(2)

        results, job_id, job_data = backend.run([qc], shots=1000)

        assert isinstance(results, list)
        assert isinstance(job_id, str)
        assert isinstance(job_data, dict)
        assert len(results) == 1

    def test_async_backend_run_with_failed_job(self):
        """Test that run method raises error for failed jobs."""

        class FailingBackend(ConcreteAsyncBackend):
            """Backend that marks jobs as failed."""

            def submit(self, circuits, shots=1000, job_name=None):
                job_id, job_data = super().submit(circuits, shots, job_name)
                # Mark as failed
                self._jobs[job_id]["status"] = JobStatus.FAILED
                return job_id, job_data

        backend = FailingBackend()
        qc = QuantumCircuit(2)

        with pytest.raises(BackendError, match="failed"):
            backend.run([qc], shots=1000)

    def test_cannot_instantiate_abstract_async_backend(self):
        """Test that AbstractAsyncBackend cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            AbstractAsyncBackend()


class TestBackendSerialization:
    """Tests for backend serialization/deserialization."""

    def test_round_trip_serialization(self):
        """Test that serialize/deserialize round trip works."""
        backend = ConcreteBackend()

        # Create circuit with multiple gates
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.measure_all()

        # Round trip
        job_data = backend.serialize_job_data([qc], 2000, "round_trip")
        hydrated = backend.de_serialize_job_data(job_data)

        # Verify structure
        assert hydrated["shots"] == 2000
        assert hydrated["job_name"] == "round_trip"
        assert len(hydrated["circuits"]) == 1

        # Verify circuit can be used
        restored_qc = hydrated["circuits"][0]
        assert isinstance(restored_qc, QuantumCircuit)
        assert restored_qc.num_qubits == 3

    def test_serialize_multiple_circuits(self):
        """Test serialization of multiple circuits."""
        backend = ConcreteBackend()

        circuits = []
        for i in range(3):
            qc = QuantumCircuit(i + 1)
            qc.h(0)
            circuits.append(qc)

        job_data = backend.serialize_job_data(circuits, 1000, "multi")
        hydrated = backend.de_serialize_job_data(job_data)

        assert len(hydrated["circuits"]) == 3
        assert hydrated["circuits"][0].num_qubits == 1
        assert hydrated["circuits"][1].num_qubits == 2
        assert hydrated["circuits"][2].num_qubits == 3
