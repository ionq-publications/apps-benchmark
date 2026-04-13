"""
IonQ Cloud backend for quantum circuit execution.

This backend provides access to IonQ's cloud quantum computers and simulators
through the qiskit_ionq provider.

Copyright (c) 2025 IonQ, Inc. All rights reserved.
Licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0
International (CC BY-NC-ND 4.0) License.
For details, visit: https://creativecommons.org/licenses/by-nc-nd/4.0/
"""

import logging
from datetime import datetime
from os import getpid
from time import sleep
from typing import Any, Callable, TypeVar

import pandas as pd
import requests
from qiskit import QuantumCircuit, transpile
from qiskit_ionq.exceptions import IonQAPIError
from qiskit_ionq.ionq_backend import Backend as IonQBackend
from qiskit_ionq.ionq_job import IonQJob, jobstatus
from requests.exceptions import RequestException
from urllib3.exceptions import HTTPError, PoolError

from apps_benchmark.core.backend import AbstractAsyncBackend, JobData, JobStatus, MeasurementBatch
from apps_benchmark.errors import BackendError

T = TypeVar("T")


def robust_backend_call(
    fn: Callable[..., T],
    args: tuple[object, ...] = (),
    kwargs: dict[str, object] | None = None,
    wait_time: int = 100,
    max_attempts: int = 25,
) -> T:
    """
    Robustly call a backend function with retry logic.

    Try to evaluate fn(*args, **kwargs) every wait_time seconds, up to
    max_attempts times, handling network and server errors.

    This method provides a means of calling quantum backend methods that is
    robust against RequestException, HTTPError, PoolError, and IonQAPIError.

    Args:
        fn: Function to call
        args: Positional arguments
        kwargs: Keyword arguments
        wait_time: Wait time in seconds between retries
        max_attempts: Maximum number of attempts

    Returns:
        Result of fn(*args, **kwargs)

    Raises:
        RuntimeError: If all attempts fail
    """
    if kwargs is None:
        kwargs = {}

    suffix = {2: "nd", 3: "rd"}
    for k in range(max_attempts):
        try:
            if k > 0:
                print(f"Trying to execute {fn} for the {k + 1}{suffix.get(k + 1, 'th')} time...")
            res = fn(*args, **kwargs)
            return res
        except (HTTPError, IonQAPIError, PoolError, RequestException) as e:
            logging.warning(f"\nProcess {getpid()} ran into {type(e)}:{e} at {datetime.today()}.")
            # Wait before trying again
            sleep(wait_time)
            continue
    raise RuntimeError(f"Robust backend call {fn} failed on process {getpid()}")


class IonqCloudBackend(AbstractAsyncBackend):
    """
    Asynchronous backend wrapper for IonQ Cloud execution targets.

    This backend wraps IonQ quantum computers and simulators accessible through
    the qiskit_ionq.IonQProvider class.

    Attributes:
        _target: The underlying IonQBackend instance
        _optimization_level: Qiskit transpilation optimization level (0-3)
        _do_transpile: Whether to transpile circuits before submission
    """

    def __init__(self, target: IonQBackend, optimization_level: int = 0):
        """
        Initialize IonQ Cloud backend.

        Args:
            target: IonQ backend instance from IonQProvider
            optimization_level: Transpilation optimization level (0-3, default 0)

        Example:
            >>> from qiskit_ionq import IonQProvider
            >>> provider = IonQProvider(token="your_api_key")
            >>> ionq_target = provider.get_backend("ionq_simulator")
            >>> backend = IonqCloudBackend(ionq_target, optimization_level=1)
        """
        self._optimization_level = optimization_level
        self._target = target
        self._do_transpile = target.gateset() != "native"

    def name(self) -> str:
        """
        Get the name of the backend.

        Returns:
            str: Backend name (e.g., 'ionq_simulator', 'ionq_qpu.forte')
        """
        return str(self._target.name)

    def submit(
        self,
        circuits: list[QuantumCircuit],
        shots: int = 1000,
        job_name: str | None = None,
    ) -> tuple[str, JobData]:
        """
        Submit circuits to IonQ Cloud (non-blocking).

        Args:
            circuits: List of quantum circuits to execute
            shots: Number of measurement shots
            job_name: Optional job name for tracking

        Returns:
            Tuple containing:
                - job_id: IonQ job identifier
                - job_data: Serialized job data

        Raises:
            BackendError: If submission fails

        Example:
            >>> job_id, job_data = backend.submit([circuit], shots=1000)
            >>> print(f"Submitted job: {job_id}")
        """
        try:
            # Ensure all circuits have measurements
            for circuit in circuits:
                if not any(instr.operation.name == "measure" for instr in circuit.data):
                    circuit.measure_all()

            # Transpile if needed
            if self._do_transpile:
                circuits = [
                    transpile(
                        circ, backend=self._target, optimization_level=self._optimization_level
                    )
                    for circ in circuits
                ]

            # Submit job with retry logic
            quantum_job = robust_backend_call(
                self._target.run, args=(circuits,), kwargs={"shots": int(shots), "name": job_name}
            )

            job_id = quantum_job.job_id()
            job_data = self.serialize_job_data(circuits, shots, job_name or "")

            return job_id, job_data

        except Exception as exc:
            raise BackendError(f"Failed to submit job to IonQ: {exc}") from exc

    def job_status(self, job_id: str) -> JobStatus:
        """
        Check the status of a submitted job.

        Args:
            job_id: IonQ job identifier

        Returns:
            JobStatus: Current status of the job

        Raises:
            BackendError: If status check fails

        Example:
            >>> status = backend.job_status(job_id)
            >>> if status == JobStatus.DONE:
            ...     print("Job completed!")
        """
        try:
            job = IonQJob(self._target, job_id=job_id)
            status = job.status()

            # Map IonQ status to our JobStatus enum
            if (
                status == jobstatus.JobStatus.INITIALIZING
                or status == jobstatus.JobStatus.VALIDATING
            ):
                return JobStatus.SUBMITTED
            if status == jobstatus.JobStatus.CANCELLED or status == jobstatus.JobStatus.ERROR:
                return JobStatus.FAILED

            if isinstance(status, dict):
                status_name: Any = status.get("name")
            else:
                status_name = getattr(status, "name", None)

            if not isinstance(status_name, str):
                raise BackendError(f"Unexpected IonQ job status for {job_id}: {status!r}")

            return JobStatus[status_name]

        except Exception as exc:
            raise BackendError(f"Failed to check job status for {job_id}: {exc}") from exc

    def retrieve_results(
        self,
        job_id: str,
        job_data: JobData,
    ) -> tuple[MeasurementBatch, pd.Timestamp]:
        """
        Retrieve results from a completed job.

        Args:
            job_id: IonQ job identifier
            job_data: Serialized job data (unused but required by interface)

        Returns:
            Tuple containing:
                - results: List of measurement histograms
                - completion_time: When the job completed (UTC)

        Raises:
            BackendError: If retrieval fails

        Example:
            >>> results, completion_time = backend.retrieve_results(job_id, job_data)
            >>> print(f"Results: {results[0]}")
            >>> print(f"Completed at: {completion_time}")

        Note:
            If the completion time is not available from the API, the retrieval
            time is used instead.
        """
        try:
            job = IonQJob(self._target, job_id=job_id)

            # Get results with retry logic
            res = robust_backend_call(job.result).get_counts()

            # Ensure we return a list of measurements
            if isinstance(res, dict):
                res = [res]

            # Get the completion time from IonQ API
            completion_time = robust_backend_call(self._get_completion_time, args=(job_id,))

            return res, completion_time

        except Exception as exc:
            raise BackendError(f"Failed to retrieve results for {job_id}: {exc}") from exc

    def _get_completion_time(self, job_id: str) -> pd.Timestamp:
        """
        Get the completion time for a job from IonQ API.

        Args:
            job_id: IonQ job identifier

        Returns:
            pd.Timestamp: Job completion time in UTC

        Raises:
            BackendError: If API call fails
        """
        try:
            ionq_api_key = self._target.provider.credentials["token"]
            headers = {
                "Authorization": f"apiKey {ionq_api_key}",
                "Content-Type": "application/json",
            }
            response = requests.get(
                f"https://api.ionq.co/v0.4/jobs/{job_id}", headers=headers, timeout=30
            )
            response.raise_for_status()

            completion_time = response.json()["completed_at"]
            timestamp = pd.Timestamp(completion_time)
            if timestamp.tzinfo is None:
                return timestamp.tz_localize("UTC")
            return timestamp.tz_convert("UTC")

        except Exception as exc:
            # If we can't get completion time, return current time
            logging.warning(f"Could not retrieve completion time for {job_id}: {exc}")
            return pd.Timestamp.now(tz="UTC")

    def validate_connection(self) -> bool:
        """
        Test IonQ backend connectivity.

        Returns:
            bool: True if connection is valid

        Raises:
            BackendError: If connection validation fails

        Example:
            >>> if backend.validate_connection():
            ...     print("Backend is ready")
        """
        try:
            # Make a lightweight API call to verify credentials and connectivity
            # This works for both simulators and QPU backends just gets a list of backends available
            self._target.client.get_with_retry(self._target.client.make_path("backends"))
            return True
        except Exception as exc:
            raise BackendError(f"IonQ backend validation failed: {exc}") from exc
