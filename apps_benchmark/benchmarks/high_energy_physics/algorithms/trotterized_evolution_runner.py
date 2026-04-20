"""
Trotterized Evolution benchmark runner for High Energy Physics.

This runner evaluates a trotterized quantum circuit describing a neutrino-less
double Beta decay process in lattice Quantum Chromo Dynamics.

NOTE: All benchmark cases are 32 qubits in size and cannot be executed using
a simulator. These benchmarks are designed for execution on quantum hardware.

Copyright (c) 2025 IonQ, Inc. All rights reserved.
Licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0
International (CC BY-NC-ND 4.0) License.
For details, visit: https://creativecommons.org/licenses/by-nc-nd/4.0/
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from qiskit import QuantumCircuit
from qiskit.qasm3 import load

from apps_benchmark.core.backend import MeasurementBatch
from apps_benchmark.core.qc_benchmark_runner import CircuitBenchmarkRunner
from apps_benchmark.primitives.benchmark_case import BenchmarkCase


class TrotterizedEvolutionRunner(CircuitBenchmarkRunner):
    """
    A class that runs a trotterized quantum circuit describing a neutrino-less
    double Beta decay process in lattice Quantum Chromo Dynamics.
    """

    def name(self) -> str:
        """
        Get the name of this algorithm.
        """
        return "trotterized_evolution"

    def get_benchmark_circuits(self, benchmark_case: BenchmarkCase) -> list[QuantumCircuit]:
        """
        Load the trotterized evolution circuit from the QASM file specified in the benchmark case.

        :param benchmark_case: The benchmark case containing circuit path and parameters
        :return: List containing the loaded quantum circuit
        """
        # Import and define the inputs from the benchmark_case file
        circuit_path = Path(str(benchmark_case.data["circuit"]))
        if not circuit_path.exists():
            circuit_path = Path(__file__).parent.parent / "benchmark_cases" / circuit_path.name

        circuit = load(circuit_path)

        return [circuit]

    def merit_figures_from_measurements(
        self,
        measurements: MeasurementBatch,
        benchmark_case: BenchmarkCase,
    ) -> dict[str, Any]:
        """
        Compute probability of the measuring the target bitstring from trotterized evolution results.

        The score is calculated as the deviation of the computed lepton number (L)
        from the expected simulation value (Lsim).

        :param measurements: List of measurement histograms from circuit execution
        :param benchmark_case: The benchmark case containing reference values
        :return: Dictionary with score and related metrics
        """
        # Import the required data from the benchmark_case file
        Lsim = benchmark_case.data["L"]
        Qesim = benchmark_case.data["Qe"]

        # Measurement outcomes from trotterized evolution circuit - reverse bitstrings
        counts_dict = {state[::-1]: counts for state, counts in measurements[0].items()}
        total_shots = sum(counts_dict.values())

        # Expectation values of Lepton number and charge from HEP circuit
        L = round(lepton_number(counts_dict), 10)
        Qe = round(lepton_charge(counts_dict), 10)

        # Compute the deviation from noiseless simulation values
        signed_lepton_number_error = L - Lsim
        score = abs(signed_lepton_number_error)

        return {
            "score": score,
            "signed_lepton_number_error": signed_lepton_number_error,
            "lepton_number": L,
            "lepton_charge": Qe,
            "reference_lepton_number": Lsim,
            "reference_lepton_charge": Qesim,
            "total_shots": total_shots,
        }


def z_expectation(counts: dict[str, int], qubit: int, num_shots: int) -> float:
    """
    Calculate the expectation value of the Z operator on a specific qubit.

    :param counts: Dictionary of measurement outcomes
    :param qubit: Index of the qubit to measure
    :param num_shots: Total number of shots
    :return: Expectation value in range [-1, 1]
    """
    expectation_value = 0.0
    for outcome, count in counts.items():
        z_value = 1 if outcome[qubit] == "0" else -1
        expectation_value += z_value * count / num_shots

    return expectation_value


def lepton_number(data: dict[str, int]) -> float:
    """
    Calculate the lepton number from measurement data.

    The lepton number is computed from Z expectation values on specific qubits
    that correspond to lepton states in the trotterized QCD simulation.

    :param data: Dictionary of measurement outcomes
    :return: Computed lepton number
    """
    num_shots = sum(data.values())
    L1 = 0.5 * sum(z_expectation(data, 24 + 2 * n, num_shots) for n in range(4))
    L2 = 0.5 * sum(z_expectation(data, 25 + 2 * n, num_shots) for n in range(4))
    return L1 + L2


def lepton_charge(data: dict[str, int]) -> float:
    """
    Calculate the lepton charge from measurement data.

    :param data: Dictionary of measurement outcomes
    :return: Computed lepton charge
    """
    num_shots = sum(data.values())
    return -0.5 * sum(z_expectation(data, 25 + 2 * n, num_shots) for n in range(4))
