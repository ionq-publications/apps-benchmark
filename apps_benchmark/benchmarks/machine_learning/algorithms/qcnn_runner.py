"""Quantum Convolutional Neural Network benchmark runner."""

from __future__ import annotations

from typing import Any

from qiskit import QuantumCircuit

from apps_benchmark.benchmarks.machine_learning.qcnn_support import (
    build_feature_map_circuit,
    classify_observable_triplets,
    confusion_matrix,
    get_num_examples,
    get_observables,
    load_encoded_images,
    load_test_labels,
    load_trained_ansatz_circuit,
    observable_triplets_from_measurements,
)
from apps_benchmark.core.backend import MeasurementBatch
from apps_benchmark.core.qc_benchmark_runner import CircuitBenchmarkRunner
from apps_benchmark.primitives.benchmark_case import BenchmarkCase


class QcnnRunner(CircuitBenchmarkRunner):
    """Run the QCNN benchmark with the stored model files."""

    def name(self) -> str:
        """Return the algorithm identifier."""
        return "qcnn"

    def get_benchmark_circuits(self, benchmark_case: BenchmarkCase) -> list[QuantumCircuit]:
        """
        Build all unmeasured QCNN inference circuits for the benchmark case.

        Each evaluation example contributes one circuit per configured observable
        head, so these cases produce 150 circuits in total.
        """
        encoded_images = load_encoded_images(benchmark_case)
        observables = get_observables(benchmark_case)
        trained_ansatz_dir = benchmark_case.data["quantum_model"]["trained_ansatz_dir"]

        circuits: list[QuantumCircuit] = []
        for image_index, encoded_image in enumerate(encoded_images):
            feature_map = build_feature_map_circuit(encoded_image)
            for observable in observables:
                trained_ansatz = load_trained_ansatz_circuit(trained_ansatz_dir, observable)
                circuit = feature_map.compose(trained_ansatz, range(feature_map.num_qubits))
                circuit.name = (
                    f"qcnn_{benchmark_case.instance_name}_img{image_index:03d}_{observable.lower()}"
                )
                circuits.append(circuit)
        return circuits

    def merit_figures_from_measurements(
        self,
        measurements: MeasurementBatch,
        benchmark_case: BenchmarkCase,
    ) -> dict[str, Any]:
        """Compute QCNN accuracy and supporting evaluation diagnostics."""
        labels = load_test_labels(benchmark_case)
        observable_triplets = observable_triplets_from_measurements(measurements, benchmark_case)
        predictions, classifier_outputs = classify_observable_triplets(
            observable_triplets,
            benchmark_case,
        )

        num_classes = int(benchmark_case.data["classical_model"]["output_classes"])
        confusion = confusion_matrix(labels, predictions, num_classes)
        score = float((predictions == labels).mean())

        return {
            "score": score,
            "num_examples": get_num_examples(benchmark_case),
            "predictions": predictions.tolist(),
            "labels": labels.tolist(),
            "observable_triplets": observable_triplets.tolist(),
            "classifier_outputs": classifier_outputs.tolist(),
            "confusion_matrix": confusion.tolist(),
        }
