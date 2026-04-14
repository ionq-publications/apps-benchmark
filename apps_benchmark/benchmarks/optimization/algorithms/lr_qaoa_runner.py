#
# Copyright (c)2025. IonQ, Inc. All rights reserved.
#
from functools import partial
from pathlib import Path
from typing import Any, Mapping, Sequence

import networkx as nx
import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

from apps_benchmark.core.backend import MeasurementBatch
from apps_benchmark.core.qc_benchmark_runner import CircuitBenchmarkRunner
from apps_benchmark.primitives.benchmark_case import BenchmarkCase


class LrQaoaRunner(CircuitBenchmarkRunner):
    """
    A class that runs a p-layer LR-QAOA algorithm
    to solve the MaxCut problem on 3- and 4-regular graphs.
    """

    def name(self) -> str:
        """
        Get the name of this algorithm.
        """
        return "lr_qaoa"

    def get_benchmark_circuits(self, benchmark_case: BenchmarkCase) -> list[QuantumCircuit]:
        # Import and define the graph from the benchmark_case file
        graph_file = (
            Path(__file__).parent.parent / "benchmark_cases" / benchmark_case.data["graph_file"]
        )
        graph = nx.read_gml(graph_file, destringizer=int)

        # Degree of the regular graph
        degrees = [deg for _, deg in graph.degree()]
        if len(set(degrees)) == 1:
            d = degrees[0]
        else:
            raise ValueError("The graph is not regular.")

        # Get the QAOA ansatz parameters from a Linear Ramp
        p = benchmark_case.data["p"]  # No. of layers

        # Values of the LR-QAOA slope for fully-connected weighted MaxCut instances
        fc_delta_values = {8: 1.35, 12: 1.25, 16: 1.2, 20: 1.0, 24: 0.9, 30: 0.8, 36: 0.7}

        if d == len(graph) - 1:
            delta = fc_delta_values[len(graph)]
        else:
            delta = 1.25  # Slope value optimized from ideal simulations

        delta_gamma = delta
        delta_beta = delta
        gamma = np.arange(1, p + 1) * delta_gamma / p
        beta = np.arange(1, p + 1)[::-1] * delta_beta / p

        # Construct the QAOA circuit
        ansatz = get_qaoa_circuit(graph, gamma, beta, save_statevector=False)

        return [ansatz]

    def merit_figures_from_measurements(
        self,
        measurements: MeasurementBatch,
        benchmark_case: BenchmarkCase,
    ) -> dict[str, Any]:
        """
        Compute approximation ratio from QAOA results.
        """
        # Measurement outcomes from QAOA circuit
        counts_dict = invert_counts(measurements[0])

        # Import and define the graph from the benchmark_case file
        graph_file = (
            Path(__file__).parent.parent / "benchmark_cases" / benchmark_case.data["graph_file"]
        )
        graph = nx.read_gml(graph_file, destringizer=int)
        adj_matrix = get_adjacency_matrix(graph)

        # Load optimal cut value from benchmark_case
        optimal_cut = benchmark_case.data["opt"]

        # Compute MaxCut merit figure: approximation ratio
        obj = partial(maxcut_obj, w=adj_matrix)  # Cost (objective) per bitstring
        total_energy = 0.0
        total_shots: float = 0.0
        probs = False
        for bitstring, count in counts_dict.items():
            probs = True if count < 1.0 else False
            bitstring_str = str(bitstring)[: benchmark_case.num_qubits]
            state_array = np.array([int(bit) for bit in bitstring_str])

            cost_value = obj(state_array)
            total_energy += cost_value * count
            total_shots += count

        # Calculate average energy
        if not probs:
            avg_energy = total_energy / total_shots
        else:
            avg_energy = total_energy

        score = avg_energy / optimal_cut  # approximation ratio
        return {"score": score}


###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################

"""
QAOA circuit for MAXCUT
"""


def append_zz_term(qc: QuantumCircuit, q1: int, q2: int, gamma: float) -> None:
    qc.rzz(-gamma / 2, q1, q2)


def append_maxcut_cost_operator_circuit(qc: QuantumCircuit, G: nx.Graph, gamma: float) -> None:
    for i, j in G.edges():
        if nx.is_weighted(G):
            append_zz_term(qc, i, j, gamma * G[i][j]["weight"])
        else:
            append_zz_term(qc, i, j, gamma)


def append_x_term(qc: QuantumCircuit, q1: int, beta: float) -> None:
    qc.rx(2 * beta, q1)


def append_mixer_operator_circuit(qc: QuantumCircuit, G: nx.Graph, beta: float) -> None:
    for n in G.nodes():
        append_x_term(qc, n, beta)


def get_qaoa_circuit(
    G: nx.Graph,
    gammas: Sequence[float],
    betas: Sequence[float],
    save_statevector: bool = False,
    qr: QuantumRegister | None = None,
    cr: ClassicalRegister | None = None,
) -> QuantumCircuit:
    """Generates a circuit for weighted MaxCut on graph G.
    Parameters
    ----------
    G : networkx.Graph
        Graph to solve MaxCut on
    beta : list-like
        QAOA parameter beta
    gamma : list-like
        QAOA parameter gamma
    save_statevector : bool, default True
        Add save state instruction to the end of the circuit
    qr : qiskit.QuantumRegister, default None
        Registers to use for the circuit.
        Useful when one has to compose circuits in a complicated way
        By default, G.number_of_nodes() registers are used
    cr : qiskit.ClassicalRegister, default None
        Classical registers, useful if measuring
        By default, no classical registers are added
    Returns
    -------
    qc : qiskit.QuantumCircuit
        Quantum circuit implementing QAOA
    """
    assert len(betas) == len(gammas)
    p = len(betas)  # infering number of QAOA steps from the parameters passed
    N = G.number_of_nodes()
    if qr is not None:
        assert qr.size >= N
    else:
        qr = QuantumRegister(N)

    if cr is not None:
        qc = QuantumCircuit(qr, cr)
    else:
        qc = QuantumCircuit(qr)

    # first, apply a layer of Hadamards
    qc.h(range(N))
    # second, apply p alternating operators
    for i in range(p):
        append_maxcut_cost_operator_circuit(qc, G, gammas[i])
        append_mixer_operator_circuit(qc, G, betas[i])
    if save_statevector:
        qc.save_statevector()
    return qc


"""
Helper functions for the Maximum Cut (MaxCut) problem
"""


def maxcut_obj(x: np.ndarray, w: np.ndarray) -> float:
    """Compute the value of a cut.
    Args:
        x (numpy.ndarray): binary string as numpy array.
        w (numpy.ndarray): adjacency matrix returned by get_adjacency_matrix
    Returns:
        float: value of the cut.
    """
    X = np.outer(x, (1 - x))
    return np.sum(w * X)  # type: ignore


def get_adjacency_matrix(G: nx.Graph) -> np.ndarray:
    """Get adjacency matrix to be used in maxcut_obj
    Args:
        G (nx.Graph) : graph
    Returns:
        w (numpy.ndarray): adjacency matrix
    """
    n = G.number_of_nodes()
    w = np.zeros([n, n])

    for e in G.edges():
        if nx.is_weighted(G):
            w[e[0], e[1]] = G[e[0]][e[1]]["weight"]
            w[e[1], e[0]] = G[e[0]][e[1]]["weight"]
        else:
            w[e[0], e[1]] = 1
            w[e[1], e[0]] = 1
    return w


def invert_counts(counts: Mapping[str, int | float]) -> dict[str, int | float]:
    """Convert from lsb to msb ordering and vice versa"""
    return {k[::-1]: v for k, v in counts.items()}
