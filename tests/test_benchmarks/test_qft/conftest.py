"""
Shared test helpers for QFT benchmark tests.

Copyright (c) 2025 IonQ, Inc. All rights reserved.
Licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0
International (CC BY-NC-ND 4.0) License.
For details, visit: https://creativecommons.org/licenses/by-nc-nd/4.0/
"""

import numpy as np
from qiskit.quantum_info import Statevector


def assert_unmeasured(circuit) -> None:
    """Assert a circuit has no classical bits and no measurement gates."""
    assert circuit.num_clbits == 0
    assert all(inst.operation.name != "measure" for inst in circuit.data)


def exact_probabilities(circuit, cutoff: float = 1e-12) -> dict[str, float]:
    """Return the exact nonzero output probabilities for a circuit."""
    probs = Statevector.from_instruction(circuit).probabilities_dict()
    return {bitstring: float(prob) for bitstring, prob in probs.items() if prob > cutoff}


class SeededSamplingBackend:
    """Deterministic statevector sampler for runner contract tests."""

    def __init__(self, seed: int):
        self._rng = np.random.default_rng(seed)
        self._calls = 0

    def name(self) -> str:
        return "seeded-sampling"

    def run(self, circuits, shots: int = 1000, job_name: str | None = None):
        histograms = []
        for qc in circuits:
            assert_unmeasured(qc)
            probs = Statevector.from_instruction(qc).probabilities_dict()
            bitstrings = sorted(probs)
            weights = np.array([probs[b] for b in bitstrings], dtype=float)
            weights /= weights.sum()
            draws = self._rng.choice(len(bitstrings), size=int(shots), p=weights)
            unique_idx, counts = np.unique(draws, return_counts=True)
            histograms.append({bitstrings[i]: int(c) for i, c in zip(unique_idx, counts)})
        self._calls += 1
        return histograms, f"job-{self._calls}", {"calls": self._calls}
