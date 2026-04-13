#
# Copyright (c)2025. IonQ, Inc. All rights reserved.
#
from __future__ import annotations

from typing import Any

###############################################################################
# MIT License
# portions of this are Copyright (c) 2020 alibaba-edu
# portions of this are Copyright (c) 2021 MIT
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
###############################################################################
# Generate the polynonial to fit the Heavyside step function using pyqsp
# The following code assumes pyqsp version 0.1.6
import pyqsp  # type: ignore[import-untyped]  # noqa: F401
from pyqsp.angle_sequence import (  # type: ignore[import-untyped]  # noqa: F401
    Polynomial,
    QuantumSignalProcessingPhases,
)
from qiskit import QuantumCircuit
from qiskit.circuit import Gate

from apps_benchmark.core.backend import MeasurementBatch
from apps_benchmark.core.qc_benchmark_runner import CircuitBenchmarkRunner
from apps_benchmark.primitives.benchmark_case import BenchmarkCase


class FaaRunner(CircuitBenchmarkRunner):
    """
    A class that runs a fixed point amplitude amplification for a single
    bitstring using Quantum Singular Value Transforms
    from John M Martyn, Zane M Rossi, Andrew K Tan, and Isaac L Chuang.
    Grand unification of quantum algorithms. PRX quantum, 2(4):040203, 2021.
    """

    def name(self) -> str:
        """
        Get the name of this algorithm.
        """
        return "faa"

    def get_benchmark_circuits(self, benchmark_case: BenchmarkCase) -> list[QuantumCircuit]:
        # Import and define the inputs from the benchmark_case file
        target = benchmark_case.data["target"]
        num_system_qubits = benchmark_case.data["num_qubits"]
        phase_option = benchmark_case.data["phase_option"]

        # QSVT for Fixed Point Amplitude Amplification!
        num_qubits = num_system_qubits + 1 + (num_system_qubits - 2)
        sys_range = list(range(0, num_system_qubits))
        full_range = list(range(0, num_qubits))
        phases = generate_qsp_angles(option=phase_option)
        num_iter = len(phases) // 2

        u = U(num_system_qubits)
        udag = Udag(u)

        qc = QuantumCircuit(num_qubits)
        qc.rz(phases[0], num_system_qubits)

        phase_idx = 1
        for i in range(num_iter):
            qc.append(u, sys_range)
            qc.append(
                proj_cnot_t(num_system_qubits, phases[phase_idx], target, phase_idx), full_range
            )
            phase_idx += 1
            if i < num_iter - 1:
                qc.append(udag, sys_range)
                qc.append(
                    proj_cnot_0(num_system_qubits, phases[phase_idx], phase_idx), full_range
                )
                phase_idx += 1

        return [qc]

    def merit_figures_from_measurements(
        self,
        measurements: MeasurementBatch,
        benchmark_case: BenchmarkCase,
    ) -> dict[str, Any]:
        """
        Compute probability of the measuring the target bitstring from FAA results.
        """
        # Import the required data from the benchmark_case file
        target = benchmark_case.data["target"]
        num_qubits = benchmark_case.data["num_qubits"]

        # Measurement outcomes from FAA circuit
        counts_dict = measurements[0]
        total_shots = sum(counts_dict.values())
        probs: dict[str, int | float] = {}
        for state, counts in counts_dict.items():
            state_trunc = state[-num_qubits:]
            if state_trunc in probs:
                probs[state_trunc] += counts
            else:
                probs[state_trunc] = counts

        # Compute the probability of the target bitstring
        score = 0.0
        if target in probs:
            score = probs[target] / total_shots  # probability of target bitstring

        return {"score": score}


"""
FAA circuit for unstructured search
"""


# Define unitary to switch from 0 basis to target basis
def U(nq: int) -> Gate:
    qc = QuantumCircuit(nq)
    for q in range(nq):
        qc.h(q)
    U_gate = qc.to_gate()
    U_gate.name = "U"
    return U_gate


# Define unitary to switch from target basis to 0 basis
def Udag(gate: Gate) -> Gate:
    gate_rev = gate.reverse_ops()
    gate_rev.name = gate.name + "$^†$"
    return gate_rev


# Define projector controlled CNOT for the target subspace
def proj_cnot_t(nq: int, phi: float, target: str = "", idx: int = 0) -> Gate:
    num_qubits = nq + 1 + nq - 2
    qc = QuantumCircuit(num_qubits)
    if target == "":
        target = "1" * nq

    controls = list(range(0, nq))
    control_state = "1" * nq
    ancilla_qubits = list(range(nq + 1, num_qubits))
    control_idxs = [idx for (idx, bit) in enumerate(target[::-1]) if bit == "0"]
    for q in control_idxs:
        qc.x(q)
    qc.mcx(controls, nq, ancilla_qubits=ancilla_qubits, mode="v-chain", ctrl_state=control_state)
    qc.rz(phi, nq)
    qc.mcx(controls, nq, ancilla_qubits=ancilla_qubits, mode="v-chain", ctrl_state=control_state)
    for q in control_idxs:
        qc.x(q)
    p_tgate = qc.to_gate()
    p_tgate.name = f"P$_{{target}}(Φ_{idx})$"
    return p_tgate


# Define projector controlled CNOT for the '0' subspace
def proj_cnot_0(nq: int, phi: float, idx: int = 0) -> Gate:
    num_qubits = nq + 1 + nq - 2
    qc = QuantumCircuit(num_qubits)
    controls = list(range(0, nq))
    control_state = "1" * nq
    ancilla_qubits = list(range(nq + 1, num_qubits))
    for q in range(nq):
        qc.x(q)
    qc.mcx(controls, nq, ancilla_qubits=ancilla_qubits, mode="v-chain", ctrl_state=control_state)
    qc.rz(phi, nq)
    qc.mcx(controls, nq, ancilla_qubits=ancilla_qubits, mode="v-chain", ctrl_state=control_state)
    for q in range(nq):
        qc.x(q)
    p_0gate = qc.to_gate()
    p_0gate.name = f"P$_0(Φ_{idx})$"
    return p_0gate


# Generate the QSP phases from the polymial fit
def generate_qsp_angles(option: int = 2) -> list[float]:
    """
    DEGREE = 19
    X_BASIS = True

    pg = pyqsp.poly.PolySign()
    pcoefs, scale = pg.generate(
        degree=DEGREE, delta=25, ensure_bounded=True, return_scale=True
    )

    poly = Polynomial(pcoefs)
    measurement = "x"
    ang_seq = QuantumSignalProcessingPhases(
        poly, signal_operator="Wx", method="laurent", measurement=measurement
    )
    #pyqsp.response.PlotQSPResponse(ang_seq, signal_operator="Wx", measurement=measurement)

    #### change the R(x) to W(x), as the phases are in the W(x) conventions
    phases = np.array(ang_seq[::-1])
    phases[1:-1] = phases[1:-1] - np.pi / 2
    phases[0] = phases[0] - np.pi / 4
    phases[-1] = phases[-1] + (2 * (len(phases) - 1) - 1) * np.pi / 4
    phases = (-2 * phases)  # -2 is due to exp(-i*phi/2*z) in qiskit
    """
    if option == 1:
        # Option 1
        phases = [
            -1.44174911,
            2.96208034,
            3.64950635,
            2.62339909,
            5.22425252,
            5.22425252,
            2.62339909,
            3.64950635,
            2.96208034,
            -26.57449034,
        ]
    elif option == 2:
        # Option 2
        phases = [
            -1.97544876,
            2.63015536,
            2.7205025,
            4.44718084,
            5.15133251,
            4.37493862,
            2.3165292,
            2.16381159,
            1.44505292,
            4.54842497,
            4.54842497,
            1.44505292,
            2.16381159,
            2.3165292,
            4.37493862,
            5.15133251,
            4.44718084,
            2.7205025,
            2.63015536,
            -58.52411653,
        ]

    return phases
