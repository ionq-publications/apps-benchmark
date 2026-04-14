"""Shared helpers for the QCNN benchmark in the machine learning category."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import cast

import numpy as np
from PIL import Image
from qiskit import QuantumCircuit, QuantumRegister, qpy
from qiskit.quantum_info import SparsePauliOp

from apps_benchmark.core.backend import MeasurementBatch, MeasurementHistogram
from apps_benchmark.primitives.benchmark_case import BenchmarkCase

_ASSET_ROOT = Path(__file__).resolve().parent / "assets"


def get_asset_root() -> Path:
    """Return the QCNN asset directory."""
    return _ASSET_ROOT


def resolve_asset_path(asset_name: str) -> Path:
    """Resolve a QCNN asset relative to the family asset root."""
    path = _ASSET_ROOT / asset_name
    if not path.is_file():
        raise FileNotFoundError(f"QCNN asset not found: {path}")
    return path


def get_observables(benchmark_case: BenchmarkCase) -> tuple[str, ...]:
    """Return the ordered observable heads for the benchmark case."""
    observables = tuple(benchmark_case.data["quantum_model"]["measurement_observables"])
    if not observables:
        raise ValueError("QCNN benchmark case must define at least one observable.")
    return observables


def get_num_examples(benchmark_case: BenchmarkCase) -> int:
    """Return the number of evaluation examples for the benchmark case."""
    return int(benchmark_case.data["dataset"]["evaluation_examples"])


def get_resize_shape(benchmark_case: BenchmarkCase) -> tuple[int, int]:
    """Return the target image shape for the benchmark case."""
    height, width = benchmark_case.data["dataset"]["resize_to"]
    return int(height), int(width)


def load_test_labels(benchmark_case: BenchmarkCase) -> np.ndarray:
    """Load the benchmark labels slice for the benchmark case."""
    labels_file = benchmark_case.data["assets"]["test_labels_file"]
    labels = np.loadtxt(resolve_asset_path(labels_file), dtype=np.float64).astype(int)
    num_examples = get_num_examples(benchmark_case)
    if labels.shape[0] < num_examples:
        raise ValueError(
            f"QCNN labels asset has {labels.shape[0]} entries but requires {num_examples}."
        )
    return labels[:num_examples]


def load_encoded_images(benchmark_case: BenchmarkCase) -> np.ndarray:
    """
    Load, resize, normalize, and flatten the QCNN evaluation images.

    Returns an array of shape ``(num_examples, benchmark_case.num_qubits)``.
    """
    images_file = benchmark_case.data["assets"]["test_images_file"]
    images = np.load(resolve_asset_path(images_file)).astype(np.float64)
    num_examples = get_num_examples(benchmark_case)
    target_shape = get_resize_shape(benchmark_case)

    image_stack = _coerce_image_stack(images)
    if image_stack.shape[0] < num_examples:
        raise ValueError(
            f"QCNN image asset has {image_stack.shape[0]} examples but requires {num_examples}."
        )
    image_stack = image_stack[:num_examples]
    if image_stack.shape[1:] != target_shape:
        image_stack = resize_image_stack(image_stack, target_shape)

    encoded_images = np.zeros((num_examples, benchmark_case.num_qubits), dtype=np.float64)
    for image_index, image in enumerate(image_stack):
        encoded_images[image_index] = _normalize_image(image).reshape(-1)

    if encoded_images.shape[1] != benchmark_case.num_qubits:
        raise ValueError(
            "QCNN encoded image width does not match the benchmark case qubit count: "
            f"{encoded_images.shape[1]} != {benchmark_case.num_qubits}."
        )
    return encoded_images


def resize_image_stack(images: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Resize a stack of grayscale images using float-preserving bilinear Pillow resize."""
    target_height, target_width = target_shape
    if images.shape[1:] == target_shape:
        return images.astype(np.float64)

    resized = []
    for image in images:
        pil_image = Image.fromarray(image.astype(np.float32))
        pil_image = pil_image.resize((target_width, target_height), resample=Image.Resampling.BILINEAR)
        resized.append(np.asarray(pil_image, dtype=np.float64))
    return np.stack(resized, axis=0)


def build_feature_map_circuit(encoded_image: np.ndarray) -> QuantumCircuit:
    """Build the fixed QCNN image-encoding circuit for one flattened image."""
    num_qubits = int(encoded_image.shape[0])
    qreg = QuantumRegister(num_qubits)
    circuit = QuantumCircuit(qreg)

    for qubit, value in enumerate(encoded_image):
        circuit.ry(np.pi * float(value), qubit)
    for qubit in range(num_qubits - 1):
        circuit.cx(qubit, qubit + 1)

    return circuit


@lru_cache(maxsize=None)
def load_trained_ansatz_circuit(ansatz_dir: str, observable: str) -> QuantumCircuit:
    """Load one bound QCNN ansatz circuit from the QPY assets."""
    if observable not in {"X", "Y", "Z"}:
        raise ValueError(f"Unsupported QCNN observable: {observable}")

    path = resolve_asset_path(f"{ansatz_dir}/trained_circ_ansatz_obs_{observable}_.qpy")
    with path.open("rb") as handle:
        circuits = qpy.load(handle)

    if len(circuits) != 1:
        raise ValueError(f"Expected one QCNN ansatz circuit in {path}, found {len(circuits)}.")

    return circuits[0].decompose().decompose()


def get_classifier_parameters(benchmark_case: BenchmarkCase) -> np.ndarray:
    """
    Load the classical classifier weights consumed during QCNN scoring.

    The parameter files include additional entries that are unused at
    benchmark runtime because the quantum circuits are already bound.
    """
    model_spec = benchmark_case.data["classical_model"]
    parameter_file = cast(str, model_spec["parameter_file"])
    total_entries = int(model_spec["total_parameter_file_entries"])
    consumed_entries = int(model_spec["consumed_parameter_count"])

    parameters = np.loadtxt(resolve_asset_path(parameter_file), dtype=np.float64)
    if parameters.size != total_entries:
        raise ValueError(
            f"QCNN parameter asset {parameter_file} has {parameters.size} entries, "
            f"expected {total_entries}."
        )

    unused_prefix = int(model_spec["unused_parameter_prefix_count"])
    if unused_prefix != total_entries - consumed_entries:
        raise ValueError(
            "QCNN classifier parameter metadata is inconsistent: "
            f"{unused_prefix=} != {total_entries - consumed_entries}."
        )
    return parameters[-consumed_entries:]


def observable_triplets_from_measurements(
    measurements: MeasurementBatch, benchmark_case: BenchmarkCase
) -> np.ndarray:
    """Convert per-circuit histograms into one observable triplet per evaluation image."""
    observables = get_observables(benchmark_case)
    num_examples = get_num_examples(benchmark_case)
    expected_histograms = num_examples * len(observables)
    if len(measurements) != expected_histograms:
        raise ValueError(
            f"QCNN expected {expected_histograms} measurement histograms, got {len(measurements)}."
        )

    triplets = np.zeros((num_examples, len(observables)), dtype=np.float64)
    for image_index in range(num_examples):
        base_index = image_index * len(observables)
        for observable_index, observable in enumerate(observables):
            triplets[image_index, observable_index] = expectation_from_histogram(
                measurements[base_index + observable_index],
                benchmark_case.num_qubits,
                observable,
            )
    return triplets


def expectation_from_histogram(
    histogram: MeasurementHistogram,
    num_qubits: int,
    observable: str,
) -> float:
    """
    Reproduce the QCNN observable extraction from sampled bitstrings.

    This intentionally keeps the benchmark's histogram-to-expectation path
    instead of inferring the expectation directly from the circuit state.
    """
    if not histogram:
        raise ValueError("QCNN histogram is empty.")

    cleaned = {bitstring.replace(" ", ""): float(weight) for bitstring, weight in histogram.items()}
    weights = np.fromiter(cleaned.values(), dtype=np.float64)
    total_weight = weights.sum()
    if total_weight <= 0.0:
        raise ValueError("QCNN histogram total weight must be positive.")

    bitstrings = list(cleaned)
    states = np.asarray([list(bitstring) for bitstring in bitstrings], dtype=int)
    pauli_mask, coeffs = get_pauli_energy_data(num_qubits, observable)
    energies = (-1.0) ** (states @ pauli_mask.T) @ coeffs
    return float(np.dot(energies, weights) / total_weight)


@lru_cache(maxsize=None)
def get_pauli_energy_data(num_qubits: int, observable: str) -> tuple[np.ndarray, np.ndarray]:
    """Return vectorized energy-evaluation data for the requested last-qubit Pauli observable."""
    operator = SparsePauliOp("I" * (num_qubits - 1) + observable)
    pauli_mask = np.array([list(label) for label, _ in operator.label_iter()]) != "I"
    coeffs = operator.coeffs.real.astype(np.float64)
    return pauli_mask, coeffs


def classify_observable_triplets(
    observable_triplets: np.ndarray,
    benchmark_case: BenchmarkCase,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the 3->2->2 QCNN classifier head on the observable triplets."""
    model_spec = benchmark_case.data["classical_model"]
    input_features = int(model_spec["input_features"])
    hidden_units = int(model_spec["hidden_units"])
    output_classes = int(model_spec["output_classes"])

    if observable_triplets.shape[1] != input_features:
        raise ValueError(
            "QCNN observable width does not match the classifier input width: "
            f"{observable_triplets.shape[1]} != {input_features}."
        )

    classifier_parameters = get_classifier_parameters(benchmark_case)
    expected_parameters = input_features * hidden_units
    expected_parameters += output_classes * hidden_units
    expected_parameters += hidden_units
    expected_parameters += output_classes
    if classifier_parameters.size != expected_parameters:
        raise ValueError(
            f"QCNN classifier expects {expected_parameters} parameters, "
            f"got {classifier_parameters.size}."
        )

    weight_1_end = input_features * hidden_units
    weight_2_end = weight_1_end + output_classes * hidden_units
    bias_1_end = weight_2_end + hidden_units

    weight_1 = classifier_parameters[:weight_1_end].reshape(hidden_units, input_features)
    weight_2 = classifier_parameters[weight_1_end:weight_2_end].reshape(output_classes, hidden_units)
    bias_1 = classifier_parameters[weight_2_end:bias_1_end]
    bias_2 = classifier_parameters[bias_1_end:]

    hidden = observable_triplets @ weight_1.T + bias_1
    logits = hidden @ weight_2.T + bias_2
    activated = np.maximum(logits, 0.0)
    predictions = np.argmax(activated, axis=1)
    return predictions.astype(int), activated


def confusion_matrix(
    labels: np.ndarray,
    predictions: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """Compute a compact integer confusion matrix."""
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for label, prediction in zip(labels, predictions, strict=True):
        matrix[int(label), int(prediction)] += 1
    return matrix


def _coerce_image_stack(images: np.ndarray) -> np.ndarray:
    """Convert raw QCNN image assets into a stack shaped ``(num_images, height, width)``."""
    if images.ndim == 4:
        if images.shape[1] != 1:
            raise ValueError(
                "QCNN image assets must have a singleton channel dimension when 4D, "
                f"got shape {images.shape}."
            )
        return images.squeeze(1)
    if images.ndim == 3:
        return images
    raise ValueError(f"Unsupported QCNN image asset shape: {images.shape}")


def _normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize one image with the QCNN L2 rule."""
    norm = np.linalg.norm(image)
    if norm <= 0.0:
        raise ValueError("QCNN image norm must be positive.")
    return image / norm
