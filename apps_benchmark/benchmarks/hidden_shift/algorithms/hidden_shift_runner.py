"""
Hidden Shift algorithm benchmark runner.

Copyright (c) 2025 IonQ, Inc. All rights reserved.
Licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0
International (CC BY-NC-ND 4.0) License.
For details, visit: https://creativecommons.org/licenses/by-nc-nd/4.0/
"""
from typing import Any

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import XGate

from apps_benchmark.core.backend import MeasurementBatch
from apps_benchmark.core.qc_benchmark_runner import CircuitBenchmarkRunner
from apps_benchmark.primitives.benchmark_case import BenchmarkCase


class HiddenShiftRunner(CircuitBenchmarkRunner):
    def name(self) -> str:
        """
        Return the name of the benchmark.

        :return: The string identifier for the hidden shift benchmark.
        :rtype: str
        """
        return "hidden_shift"

    def _build_pi(
        self,
        n_half: int,
        permutation: str,
        *,
        cx_pairs: list[tuple[int, int]] | None = None,
    ) -> QuantumCircuit:
        """
        Build the permutation circuit pi on n_half qubits.

        :param n_half: The number of qubits in the half-register.
        :type n_half: int
        :param permutation: The type of permutation to build.
        :type permutation: str
        :param cx_pairs: A list of (control, target) pairs.
        :type cx_pairs: list[tuple[int, int]] | None
        :return: A QuantumCircuit representing the permutation pi.
        :rtype: QuantumCircuit
        """
        pi = QuantumCircuit(n_half, name=r"$\pi$")
        pi.barrier()

        if permutation == "ccx_ladder":
            for k in range(n_half - 2):
                pi.ccx(k, k + 1, k + 2)

        elif permutation == "cx_ladder":
            for k in range(n_half - 1):
                pi.cx(k, k + 1)

        elif permutation == "mcx":
            mcx = XGate().control(n_half - 1)
            pi.append(mcx, pi.qubits)

        elif permutation == "random":
            if cx_pairs is None:
                raise ValueError(
                    "Random permutation requires 'cx_pairs' list of (control, target) pairs."
                )
            for ctrl, tgt in cx_pairs:
                pi.cx(int(ctrl), int(tgt))

        else:
            raise NotImplementedError(f"Unrecognized permutation {permutation}!")

        pi.barrier()
        return pi

    def get_benchmark_circuits(
        self, benchmark_case: BenchmarkCase, decompose: bool = True
    ) -> list[QuantumCircuit]:
        """
        Build benchmark circuits for the hidden shift algorithm.
        """
        num_qubits = benchmark_case.num_qubits
        if num_qubits % 2 != 0:
            raise ValueError("hidden_shift requires an even num_qubits.")

        n_half = num_qubits // 2
        data = benchmark_case.data
        perm_spec = data["permutation"]
        shifts = list(data["shifts"])

        if perm_spec == "random":
            cx_list = data.get("permutation_cx_pairs")
            if cx_list is None:
                raise ValueError(
                    "hidden_shift_random instances must store 'permutation_cx_pairs' in their data."
                )

            pis = [self._build_pi(n_half, "random", cx_pairs=pairs) for pairs in cx_list]
        else:
            pis = [self._build_pi(n_half, perm_spec)]

        benchmark_qc: list[QuantumCircuit] = []

        # Give each circuit a deterministic name to make QPY serialization reproducible.
        for perm_idx, pi in enumerate(pis):
            for shift_idx, shift in enumerate(shifts):
                hidden_shift_qc = QuantumCircuit(num_qubits)
                hidden_shift_qc.name = (
                    f"hidden_shift_{benchmark_case.instance_name}_perm{perm_idx}_shift{shift_idx}"
                )

                hidden_shift_qc.h(range(num_qubits))
                hidden_shift_qc.barrier()

                for q, b in enumerate(shift[::-1]):
                    if b == "1":
                        hidden_shift_qc.x(q)

                hidden_shift_qc.append(pi, hidden_shift_qc.qubits[::2])
                for q in range(n_half):
                    hidden_shift_qc.cz(2 * q, 2 * q + 1)
                hidden_shift_qc.append(pi.inverse(), hidden_shift_qc.qubits[::2])

                for q, b in enumerate(shift[::-1]):
                    if b == "1":
                        hidden_shift_qc.x(q)

                hidden_shift_qc.barrier()
                hidden_shift_qc.h(range(num_qubits))

                hidden_shift_qc.append(pi.inverse(), hidden_shift_qc.qubits[1::2])
                for q in range(n_half):
                    hidden_shift_qc.cz(2 * q, 2 * q + 1)
                hidden_shift_qc.append(pi, hidden_shift_qc.qubits[1::2])

                hidden_shift_qc.h(range(num_qubits))

                if decompose:
                    hidden_shift_qc = hidden_shift_qc.decompose()

                benchmark_qc.append(hidden_shift_qc)

        return benchmark_qc

    def merit_figures_from_measurements(
        self,
        measurements: MeasurementBatch,
        benchmark_case: BenchmarkCase,
    ) -> dict[str, Any]:
        """
        Compute the benchmark score as the success probability of measuring the hidden shift.
        """
        data = benchmark_case.data
        perm_spec = data.get("permutation")
        shifts = list(data["shifts"])

        if perm_spec == "random":
            cx_list = data.get("permutation_cx_pairs")
            if cx_list is not None:
                num_perms = len(cx_list)
            else:
                num_perms = int(data.get("num_random_permutations", 3))
            expected_shifts = shifts * num_perms
        else:
            expected_shifts = shifts

        if len(measurements) != len(expected_shifts):
            raise ValueError(
                f"Expected {len(expected_shifts)} measurement histograms, got {len(measurements)}."
            )

        scores = np.zeros(len(measurements), dtype=float)
        for k, (hist, shift) in enumerate(zip(measurements, expected_shifts, strict=False)):
            shots = sum(hist.values())
            scores[k] = 0.0 if shots == 0 else hist.get(shift, 0) / shots

        return {"score": scores.mean(), "score_std": scores.std(), "score_arr": scores}
