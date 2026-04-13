"""
Cosine QFT benchmark runner.

Copyright (c) 2025 IonQ, Inc. All rights reserved.
Licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0
International (CC BY-NC-ND 4.0) License.
For details, visit: https://creativecommons.org/licenses/by-nc-nd/4.0/
"""

import math
from typing import Any

from qiskit import QuantumCircuit
from qiskit.circuit.library import QFT
from qiskit.quantum_info.analysis import hellinger_fidelity

from apps_benchmark.core.backend import MeasurementBatch
from apps_benchmark.core.qc_benchmark_runner import CircuitBenchmarkRunner
from apps_benchmark.primitives.benchmark_case import BenchmarkCase


class CosineQftRunner(CircuitBenchmarkRunner):
    """
    Load a cosine wave, apply a hidden shift, and recover its supported spectrum.
    """

    def name(self) -> str:
        return "cosine_qft"

    def get_benchmark_circuits(self, benchmark_case: BenchmarkCase) -> list[QuantumCircuit]:
        """
        Build the hidden-shift QFT circuit that prepares a cosine spectrum.
        """
        frequency_index = int(benchmark_case.data["frequency_index"])
        num_qubits = benchmark_case.num_qubits

        # A zero-frequency cosine is constant, so its QFT support is the
        # single all-zero basis state regardless of the hidden shift.
        if frequency_index == 0:
            return [QuantumCircuit(num_qubits)]

        register_size = 2**num_qubits
        qft_gate = QFT(num_qubits)

        shift = frequency_index + 1
        init_ket = (1 - 2 * shift) % register_size
        init_ket_bs = f"{init_ket:0{num_qubits}b}"

        qc = QuantumCircuit(num_qubits)
        qc.h(0)
        for q in range(1, num_qubits):
            qc.cx(0, q)

        qc.x(num_qubits - 1)
        for q, bit in enumerate(init_ket_bs[::-1]):
            if bit == "1":
                qc.cx(num_qubits - 1, q)
        qc.x(num_qubits - 1)
        qc.barrier()

        qc.append(qft_gate, qc.qubits)

        for qubit in range(num_qubits):
            divisor = 2 ** (num_qubits - 1 - qubit)
            qc.rz(shift * math.pi / divisor, qubit)

        qc.append(qft_gate, qc.qubits)
        return [qc.decompose(reps=2)]

    def merit_figures_from_measurements(
        self,
        measurements: MeasurementBatch,
        benchmark_case: BenchmarkCase,
    ) -> dict[str, Any]:
        """
        Score the recovered spectrum against the known cosine support.
        """
        hist = measurements[0]
        tot_mass = sum(hist.values())
        if tot_mass > 1:
            hist = {bs: val / tot_mass for bs, val in hist.items()}

        num_qubits = benchmark_case.num_qubits
        frequency_index = int(benchmark_case.data["frequency_index"])
        frequency_bs = f"{frequency_index:0{num_qubits}b}"
        if frequency_index == 0:
            target_state = {frequency_bs: 1.0}
        else:
            twos_comp_bs = f"{(2**num_qubits - frequency_index):0{num_qubits}b}"
            target_state = {frequency_bs: 0.5, twos_comp_bs: 0.5}
        return {"score": hellinger_fidelity(hist, target_state)}
