"""
Qiskit Aer backend for local simulation.

This backend uses Qiskit's AerSimulator for local quantum circuit simulation.
No credentials required - runs entirely on local hardware.

Copyright (c) 2025 IonQ, Inc. All rights reserved.
Licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0
International (CC BY-NC-ND 4.0) License.
For details, visit: https://creativecommons.org/licenses/by-nc-nd/4.0/
"""

from typing import Any

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

from apps_benchmark.core.backend import AbstractBackend, JobData, MeasurementBatch
from apps_benchmark.errors import BackendError


class QiskitAerSimBackend(AbstractBackend):
    """
    Backend using Qiskit's AerSimulator.

    This backend provides fast local simulation using Qiskit Aer.
    No API keys or cloud credentials required.

    Attributes:
        simulator: The AerSimulator instance
        method: Simulation method ('automatic', 'statevector', 'density_matrix', etc.)
        optimization_level: Transpilation optimization level (0-3)
    """

    def __init__(
        self,
        method: str = "automatic",
        optimization_level: int = 1,
        **simulator_options: Any,
    ) -> None:
        """
        Initialize Qiskit Aer backend.

        Args:
            method: Simulation method. Options include:
                   - 'automatic' (default): Automatically select best method
                   - 'statevector': State vector simulation
                   - 'density_matrix': Density matrix simulation
                   - 'stabilizer': Stabilizer simulation (Clifford circuits only)
                   - 'matrix_product_state': MPS simulation
            optimization_level: Qiskit transpilation level (0-3, default 1)
            **simulator_options: Additional options passed to AerSimulator

        Example:
            >>> backend = QiskitAerSimBackend(method="statevector")
            >>> backend = QiskitAerSimBackend(method="automatic", optimization_level=2)
        """
        self.method = method
        self.optimization_level = optimization_level
        self.simulator_options = simulator_options

        # Initialize simulator
        try:
            self.simulator = AerSimulator(method=method, **simulator_options)
        except Exception as exc:
            raise BackendError(f"Failed to initialize AerSimulator: {exc}") from exc

    def name(self) -> str:
        """Return the underlying AerSimulator backend name.

        For example, the default simulator reports ``"aer_simulator"``,
        while method-specific simulator configurations may report a
        method-qualified name from Aer.
        """
        return str(self.simulator.name)

    def run(
        self,
        circuits: list[QuantumCircuit],
        shots: int = 1000,
        job_name: str | None = None,
    ) -> tuple[MeasurementBatch, str, JobData]:
        """
        Execute circuits on Qiskit Aer simulator.

        Args:
            circuits: List of quantum circuits to execute
            shots: Number of measurement shots
            job_name: Optional job name for tracking

        Returns:
            Tuple containing:
                - results: List of measurement histograms (one per circuit)
                - job_id: Job identifier
                - job_data: Serialized job data

        Raises:
            BackendError: If execution fails

        Example:
            >>> backend = QiskitAerSimBackend()
            >>> qc = QuantumCircuit(2)
            >>> qc.h(0)
            >>> qc.cx(0, 1)
            >>> results, job_id, job_data = backend.run([qc], shots=1000)
            >>> results[0]
            {'00': 512, '11': 488}
        """
        # Reject circuits that already contain measurement operations.
        # Backends are responsible for adding terminal measurements.
        for idx, qc in enumerate(circuits):
            if any(inst.operation.name == "measure" for inst in qc.data):
                raise BackendError(
                    f"Measurement gates are unsupported. "
                    f"Check circuit {idx} and try again."
                )

        circuits_to_run = [qc.measure_all(inplace=False) for qc in circuits]

        try:
            # Transpile circuits for simulator
            transpiled_circuits = transpile(
                circuits_to_run,
                backend=self.simulator,
                optimization_level=self.optimization_level,
            )

            # Run simulation
            job = self.simulator.run(transpiled_circuits, shots=shots)
            result = job.result()

            # Extract measurement results
            results: MeasurementBatch = []
            for i in range(len(circuits_to_run)):
                counts = result.get_counts(i)
                # Convert Qiskit Counts object to plain dict
                results.append(dict(counts))

            # Generate job ID
            job_id = job.job_id()

            # Serialize job data (use original circuits, not transpiled)
            job_data = self.serialize_job_data(circuits, shots, job_name or "")

            return results, job_id, job_data

        except Exception as exc:
            raise BackendError(f"Qiskit Aer execution failed: {exc}") from exc

    def validate_connection(self) -> bool:
        """
        Validate that simulator is available.

        For local simulators, this just checks that the simulator
        object is initialized correctly.

        Returns:
            bool: True if simulator is available

        Raises:
            BackendError: If simulator is not available
        """
        try:
            # Try to get simulator configuration
            config = self.simulator.configuration()
            return config is not None
        except Exception as exc:
            raise BackendError(f"Qiskit Aer simulator not available: {exc}") from exc

    def get_simulator_info(self) -> dict[str, object]:
        """
        Get information about the simulator.

        Returns:
            dict: Simulator configuration info

        Note:
            This is a helper method, not part of AbstractBackend interface.
        """
        config = self.simulator.configuration()
        return {
            "backend_name": config.backend_name,
            "backend_version": config.backend_version,
            "max_shots": config.max_shots,
            "simulator": True,
            "local": True,
            "conditional": config.conditional,
            "memory": config.memory,
            "method": self.method,
            "optimization_level": self.optimization_level,
        }
