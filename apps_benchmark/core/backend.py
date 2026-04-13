"""
Abstract base classes for quantum backends.

This module defines the interface that all quantum backends must implement.

Copyright (c) 2025 IonQ, Inc. All rights reserved.
Licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0
International (CC BY-NC-ND 4.0) License.
For details, visit: https://creativecommons.org/licenses/by-nc-nd/4.0/
"""

import time
from abc import ABC, abstractmethod
from enum import Enum
from typing import TypeAlias, cast

import pandas as pd
import qiskit.qasm3 as qasm3
from qiskit import QuantumCircuit

from apps_benchmark.errors import BackendError

MeasurementValue: TypeAlias = int | float
MeasurementHistogram: TypeAlias = dict[str, MeasurementValue]
MeasurementBatch: TypeAlias = list[MeasurementHistogram]
JobData: TypeAlias = dict[str, object]


class JobStatus(Enum):
    """Job status enumeration."""

    SUBMITTED = "submitted"  # Job submitted but not yet queued
    QUEUED = "queued"  # Job in queue waiting to run
    RUNNING = "running"  # Job is currently executing
    DONE = "done"  # Job completed successfully
    FAILED = "failed"  # Job failed with error


class AbstractBackend(ABC):
    """
    Abstract base class for quantum backends.

    This is the base interface for all backends. Synchronous backends
    (like local simulators) can implement this directly. Asynchronous
    backends (like cloud QPUs) should extend AbstractAsyncBackend instead.
    """

    @abstractmethod
    def name(self) -> str:
        """
        Return the backend identifier.

        Returns:
            str: Backend name (e.g., 'ionq.forte', 'qiskit.aer', 'my_simulator')

        Example:
            >>> backend.name()
            'ionq.forte'
        """
        pass

    @abstractmethod
    def run(
        self,
        circuits: list[QuantumCircuit],
        shots: int = 1000,
        job_name: str | None = None,
    ) -> tuple[MeasurementBatch, str, JobData]:
        """
        Execute quantum circuits and return results (blocking).

        This method submits circuits for execution and waits for completion.
        For synchronous backends, this executes immediately. For async backends,
        this should block until results are ready.

        Args:
            circuits: list of quantum circuits to execute
            shots: Number of measurement shots per circuit
            job_name: Optional name for the job (for tracking/debugging)

        Returns:
            tuple containing:
                - results (list[dict]): list of measurement histograms, one per circuit.
                  Each dict maps bitstring to count, e.g., {'00': 480, '11': 520}
                - job_id (str): Unique job identifier
                - job_data (dict): Serialized job data for potential re-execution

        Raises:
            BackendConnectionError: If backend is unreachable
            BackendCredentialError: If credentials are missing or invalid
            BackendError: For other backend-specific errors

        Example:
            >>> circuits = [QuantumCircuit(2), QuantumCircuit(3)]
            >>> results, job_id, job_data = backend.run(circuits, shots=1000)
            >>> results[0]
            {'00': 480, '01': 20, '10': 23, '11': 477}
        """
        pass

    def serialize_job_data(
        self,
        circuits: list[QuantumCircuit],
        shots: int,
        job_name: str,
    ) -> JobData:
        """
        Serialize quantum job data for later re-execution.

        This method should serialize all data needed to re-run the same job at
        a later time.

        Default implementation converts circuits to QASM3. Override if needed.

        Args:
            circuits: Quantum circuits
            shots: Number of shots
            job_name: Job name

        Returns:
            dict: Serialized job data

        Example:
            >>> job_data = backend.serialize_job_data(circuits, 1000, "test_job")
            >>> job_data.keys()
            dict_keys(['circuits', 'shots', 'job_name'])
        """
        return {
            "circuits": [qasm3.dumps(qc) for qc in circuits],
            "shots": shots,
            "job_name": job_name,
        }

    def de_serialize_job_data(self, job_data: JobData) -> JobData:
        """
        Deserialize quantum job data.

        Counterpart to serialize_job_data. Hydrates serialized job data
        back into usable Python objects.

        Default implementation reads QASM3 records while remaining compatible
        with older QASM2 records.

        Args:
            job_data: Serialized job data

        Returns:
            dict: Hydrated job data with QuantumCircuit objects

        Example:
            >>> hydrated = backend.de_serialize_job_data(job_data)
            >>> isinstance(hydrated["circuits"][0], QuantumCircuit)
            True
        """
        serialized_circuits = cast(list[str], job_data["circuits"])
        circuits: list[QuantumCircuit] = []
        for qc in serialized_circuits:
            try:
                circuits.append(QuantumCircuit.from_qasm_str(qc))
            except Exception:
                if qc.lstrip().startswith("OPENQASM 3"):
                    circuits.append(qasm3.loads(qc))
                else:
                    raise

        return {
            "circuits": circuits,
            "shots": int(cast(int | float | str, job_data["shots"])),
            "job_name": cast(str, job_data["job_name"]),
        }

    def validate_connection(self) -> bool:
        """
        Test backend connectivity (for --self-test).

        This method should perform a minimal check to ensure the backend
        is reachable and credentials are valid. Should NOT execute circuits.

        Returns:
            bool: True if connection is valid

        Raises:
            BackendConnectionError: If backend is unreachable
            BackendCredentialError: If credentials are missing or invalid

        Example:
            >>> backend.validate_connection()
            True

        Implementation notes:
            - Check API endpoint is reachable (for cloud backends)
            - Verify API key is valid (if applicable)
            - Do NOT submit actual circuits
            - Should complete in < 5 seconds
        """
        # Default implementation: no-op for local simulators
        return True


class AbstractAsyncBackend(AbstractBackend):
    """
    Abstract base class for asynchronous quantum backends.

    Use this for backends where job submission and result retrieval
    are separate operations (e.g., cloud QPUs, remote simulators).

    The base run() method is automatically implemented by calling
    submit() followed by polling job_status() and retrieve_results().
    """

    @abstractmethod
    def submit(
        self,
        circuits: list[QuantumCircuit],
        shots: int = 1000,
        job_name: str | None = None,
    ) -> tuple[str, JobData]:
        """
        Submit circuits to the backend (non-blocking).

        Args:
            circuits: list of quantum circuits to execute
            shots: Number of measurement shots per circuit
            job_name: Optional name for the job

        Returns:
            tuple containing:
                - job_id (str): Unique job identifier
                - job_data (dict): Serialized job data (from serialize_job_data)

        Raises:
            BackendConnectionError: If backend is unreachable
            BackendCredentialError: If credentials are missing or invalid
            BackendError: For other backend-specific errors

        Example:
            >>> job_id, job_data = backend.submit(circuits, shots=1000)
            >>> job_id
            '550e8400-e29b-41d4-a716-446655440000'

        Note:
            This method should call self.serialize_job_data() before returning.
        """
        pass

    @abstractmethod
    def job_status(self, job_id: str) -> JobStatus:
        """
        Check the status of a submitted job.

        Args:
            job_id: Job identifier returned by submit()

        Returns:
            JobStatus: Current status of the job

        Raises:
            BackendError: If job_id is not found or status check fails

        Example:
            >>> status = backend.job_status(job_id)
            >>> status
            <JobStatus.RUNNING: 'running'>
            >>> status == JobStatus.DONE
            False
        """
        pass

    @abstractmethod
    def retrieve_results(
        self,
        job_id: str,
        job_data: JobData,
    ) -> tuple[MeasurementBatch, pd.Timestamp]:
        """
        Retrieve results from a completed job.

        Args:
            job_id: Job identifier
            job_data: Serialized job data (from submit())

        Returns:
            tuple containing:
                - results (list[dict]): Measurement histograms
                - completion_time (pd.Timestamp): When job completed (UTC)

        Raises:
            BackendError: If job is not complete or results unavailable

        Example:
            >>> results, completion_time = backend.retrieve_results(job_id, job_data)
            >>> results[0]
            {'00': 480, '11': 520}
            >>> completion_time
            Timestamp('2026-03-31 14:30:00+0000', tz='UTC')

        Note:
            If completion time is not available from the backend, use
            retrieval time instead.
        """
        pass

    def run(
        self,
        circuits: list[QuantumCircuit],
        shots: int = 1000,
        job_name: str | None = None,
    ) -> tuple[MeasurementBatch, str, JobData]:
        """
        Execute circuits and wait for results (blocking).

        This method is automatically implemented by:
        1. Calling submit()
        2. Polling job_status() until DONE or FAILED
        3. Calling retrieve_results()

        You do not need to override this method.

        Returns:
            tuple containing:
                - results (list[dict]): Measurement histograms
                - job_id (str): Job identifier
                - job_data (dict): Serialized job data
        """
        job_id, job_data = self.submit(circuits, shots, job_name)

        # Poll until done (implementers can customize polling logic)
        while True:
            status = self.job_status(job_id)
            if status == JobStatus.DONE:
                break
            elif status == JobStatus.FAILED:
                raise BackendError(f"Job {job_id} failed")
            time.sleep(1)  # Poll every second

        results, completion_time = self.retrieve_results(job_id, job_data)
        return results, job_id, job_data
