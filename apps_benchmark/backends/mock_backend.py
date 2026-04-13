"""
Mock backend for testing.

This backend provides deterministic results without any actual quantum execution.
It's useful for testing the framework and for quick development iterations.

Copyright (c) 2025 IonQ, Inc. All rights reserved.
Licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0
International (CC BY-NC-ND 4.0) License.
For details, visit: https://creativecommons.org/licenses/by-nc-nd/4.0/
"""

from qiskit import QuantumCircuit

from apps_benchmark.core.backend import (
    AbstractBackend,
    JobData,
    MeasurementBatch,
    MeasurementHistogram,
)


class MockBackend(AbstractBackend):
    """
    Mock backend that returns deterministic fake results.

    This backend doesn't execute any circuits - it just returns
    deterministic results based on the number of qubits and shots.
    Useful for testing and development.

    Attributes:
        deterministic (bool): If True, always return same results. If False,
                             add some controlled randomness.
        fail_on_execution (bool): If True, raise an error when run() is called.
                                 Useful for testing error handling.
    """

    def __init__(self, deterministic: bool = True, fail_on_execution: bool = False) -> None:
        """
        Initialize mock backend.

        Args:
            deterministic: If True, always return same results
            fail_on_execution: If True, fail when run() is called
        """
        self.deterministic = deterministic
        self.fail_on_execution = fail_on_execution
        self._execution_count = 0

    def name(self) -> str:
        """Return backend name."""
        return "mock"

    def run(
        self,
        circuits: list[QuantumCircuit],
        shots: int = 1000,
        job_name: str | None = None,
    ) -> tuple[MeasurementBatch, str, JobData]:
        """
        Return mock results without executing circuits.

        Args:
            circuits: List of quantum circuits (not actually executed)
            shots: Number of shots to simulate
            job_name: Optional job name

        Returns:
            Tuple of (results, job_id, job_data)

        Raises:
            RuntimeError: If fail_on_execution is True

        Example:
            >>> backend = MockBackend()
            >>> qc = QuantumCircuit(2)
            >>> results, job_id, job_data = backend.run([qc], shots=1000)
            >>> results[0]
            {'00': 500, '11': 500}
        """
        if self.fail_on_execution:
            raise RuntimeError("Mock backend configured to fail on execution")

        self._execution_count += 1

        results: MeasurementBatch = []
        for circuit in circuits:
            num_qubits = circuit.num_qubits

            if self.deterministic:
                # Deterministic results: split shots evenly between |0...0> and |1...1>
                zero_state = "0" * num_qubits
                one_state = "1" * num_qubits
                results.append({zero_state: shots // 2, one_state: shots // 2})
            else:
                # Semi-random results: split between first few basis states
                # Still deterministic per execution count for reproducibility
                import hashlib

                result: MeasurementHistogram = {}
                remaining_shots = shots
                # Generate up to 4 basis states
                num_states = min(4, 2**num_qubits)

                for i in range(num_states - 1):
                    # Use hash of execution count and index for reproducible "randomness"
                    hash_input = f"{self._execution_count}_{i}_{num_qubits}"
                    hash_val = int(hashlib.md5(hash_input.encode()).hexdigest()[:8], 16)
                    count = (hash_val % remaining_shots) // num_states
                    state = format(i, f"0{num_qubits}b")
                    result[state] = count
                    remaining_shots -= count

                # Give remaining shots to last state
                last_state = format(num_states - 1, f"0{num_qubits}b")
                result[last_state] = remaining_shots

                results.append(result)

        # Generate mock job ID
        job_id = f"mock_job_{self._execution_count}"

        # Serialize job data
        job_data = self.serialize_job_data(circuits, shots, job_name or "")

        return results, job_id, job_data

    def validate_connection(self) -> bool:
        """
        Validate mock backend connection (always succeeds).

        Returns:
            True (mock backend always available)
        """
        return True

    def get_execution_count(self) -> int:
        """
        Get number of times run() was called.

        Returns:
            Execution count

        Note:
            This is a test helper method, not part of AbstractBackend interface.
        """
        return self._execution_count

    def reset_execution_count(self) -> None:
        """
        Reset execution counter.

        Note:
            This is a test helper method, not part of AbstractBackend interface.
        """
        self._execution_count = 0
