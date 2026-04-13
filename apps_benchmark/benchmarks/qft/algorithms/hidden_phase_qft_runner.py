"""
Hidden phase QFT benchmark runner.

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


class HiddenPhaseQftRunner(CircuitBenchmarkRunner):
    """
    Recover a hidden phase encoded through controlled shifts on QFT basis states.
    """

    def name(self) -> str:
        return "hidden_phase_qft"

    def get_benchmark_circuits(self, benchmark_case: BenchmarkCase) -> list[QuantumCircuit]:
        """
        Build the ancilla-assisted hidden-phase QFT circuit from the benchmark case.
        """
        shift = int(benchmark_case.data["phase_index"])
        num_qubits = benchmark_case.num_qubits
        qft_gate = QFT(num_qubits)

        qc = QuantumCircuit(num_qubits + 1)
        qc.h(range(num_qubits + 1))
        qc.barrier()

        qc.append(qft_gate, qc.qubits[:-1])
        qc.barrier()

        for i_q in range(num_qubits):
            divisor = 2 ** (num_qubits - 1 - i_q)
            qc.crz(shift * math.pi / divisor, num_qubits, i_q)

        qc.x(num_qubits)
        for i_q in range(num_qubits):
            divisor = 2 ** (num_qubits - 1 - i_q)
            qc.crz(-shift * math.pi / divisor, num_qubits, i_q)
        qc.x(num_qubits)
        qc.barrier()

        qc.append(qft_gate, qc.qubits[:-1])
        qc.barrier()

        qc.h(range(num_qubits + 1))
        return [qc.decompose(reps=2)]

    def merit_figures_from_measurements(
        self,
        measurements: MeasurementBatch,
        benchmark_case: BenchmarkCase,
    ) -> dict[str, Any]:
        """
        Score the ancilla/readout distribution against the known hidden-phase target.
        """
        hist = measurements[0]
        tot_mass = sum(hist.values())
        if tot_mass > 1:
            hist = {bs: val / tot_mass for bs, val in hist.items()}

        num_qubits = benchmark_case.num_qubits
        hidden_phase = int(benchmark_case.data["phase_index"])
        lam = (2 * math.pi / 2**num_qubits) * (hidden_phase / 2)
        target_state = {
            "0" * (num_qubits + 1): math.cos(lam) ** 2,
            "1" + "0" * num_qubits: math.sin(lam) ** 2,
        }
        fidelity = hellinger_fidelity(hist, target_state)
        return {"score": fidelity}
