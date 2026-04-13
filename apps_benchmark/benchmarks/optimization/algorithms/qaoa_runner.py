"""
QAOA Runner for MaxCut optimization benchmarks.

Copyright (c) 2025 IonQ, Inc. All rights reserved.
Licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0
International (CC BY-NC-ND 4.0) License.
For details, visit: https://creativecommons.org/licenses/by-nc-nd/4.0/
"""
from functools import cache, partial
from pathlib import Path
from typing import Any, Sequence

import networkx as nx
import numpy as np
import pandas as pd
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

from apps_benchmark.core.backend import MeasurementBatch
from apps_benchmark.core.qc_benchmark_runner import CircuitBenchmarkRunner
from apps_benchmark.primitives.benchmark_case import BenchmarkCase


class QaoaRunner(CircuitBenchmarkRunner):
    """
    A class that runs a p-layer QAOA ansatz using fixed parameters obtained
    from arXiv:2107.00677 to solve the MaxCut problem on 3-regular graphs.
    """

    def name(self) -> str:
        """
        Get the name of this algorithm.
        """
        return "qaoa"

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

        # Get the QAOA ansatz parameters for MaxCut on regular graphs from arXiv:2107.00677
        # Number of QAOA layers in the circuit
        p = benchmark_case.data["p"]
        gamma, beta = get_fixed_gamma_beta(d, p)

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
        total_shots = 0
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


def append_zz_term(qc, q1, q2, gamma):
    qc.rzz(-gamma / 2, q1, q2)


def append_maxcut_cost_operator_circuit(qc, G, gamma):
    for i, j in G.edges():
        if nx.is_weighted(G):
            append_zz_term(qc, i, j, gamma * G[i][j]["weight"])
        else:
            append_zz_term(qc, i, j, gamma)


def append_x_term(qc, q1, beta):
    qc.rx(2 * beta, q1)


def append_mixer_operator_circuit(qc, G, beta):
    for n in G.nodes():
        append_x_term(qc, n, beta)


def get_qaoa_circuit(
    G: nx.Graph,
    gammas: Sequence,
    betas: Sequence,
    save_statevector: bool = False,
    qr: QuantumRegister = None,
    cr: ClassicalRegister = None,
):
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


"""
Utilities for parameter initialization
"""


@cache
def _get_gamma_beta_from_file():
    """
    Caches the dataframe after the first call to load JSON.
    Subsequent calls will get the dataframe from the cache to avoid extra I/O.

    Returns
    -------
    df : pandas.DataFrame
    """
    json_path = (
        Path(__file__).parent.parent
        / "benchmark_cases"
        / "maxcut_datasets"
        / "fixed_angles_for_regular_graphs.json"
    )

    return pd.read_json(json_path, orient="index")


def get_fixed_gamma_beta(d, p, return_AR=False):
    """
    Returns the parameters for QAOA for MaxCut on regular graphs from arXiv:2107.00677

    Parameters
    ----------
    d : int
        Degree of the graph
    p : int
        QAOA depth
    return_AR : bool
        return the guaranteed approximation ratio

    Returns
    -------
    gamma, beta : (list, list)
        Parameters as two separate lists in a tuple
    AR : float
        Only returned is flag return_AR is raised
    """
    df = _get_gamma_beta_from_file()
    row = df[(df["d"] == d) & (df["p"] == p)]
    if len(row) != 1:
        raise ValueError(f"Failed to retrieve fixed angles for d={d}, p={p}")
    row = row.squeeze()
    if return_AR:
        return row["gamma"], row["beta"], row["AR"]
    else:
        return row["gamma"], row["beta"]


def invert_counts(counts):
    """Convert from lsb to msb ordering and vice versa"""
    return {k[::-1]: v for k, v in counts.items()}
