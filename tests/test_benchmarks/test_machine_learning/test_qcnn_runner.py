"""Tests for the QCNN runner in the machine learning benchmark family."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from apps_benchmark.backends.mock_backend import MockBackend
from apps_benchmark.benchmarks.machine_learning.algorithms.qcnn_runner import QcnnRunner
from apps_benchmark.benchmarks.machine_learning.qcnn_support import (
    get_asset_root,
    load_encoded_images,
)
from apps_benchmark.cli import _find_benchmark_case_by_uuid, _load_builtin_runner
from apps_benchmark.core.registry import _discover_builtin_benchmarks
from apps_benchmark.primitives.benchmark_case import BenchmarkCase
from qiskit.quantum_info import Statevector

REPO_ROOT = Path(__file__).resolve().parents[3]
QCNN_CASES = REPO_ROOT / "apps_benchmark" / "benchmarks" / "machine_learning" / "benchmark_cases"
EXPECTED_CASES = {
    "qcnn_09q_digits_1_0.instance.json": "6d522c6b",
    "qcnn_09q_digits_1_7.instance.json": "22fa7a5e",
    "qcnn_16q_digits_1_0.instance.json": "3299eec7",
    "qcnn_16q_digits_1_7.instance.json": "59fc67f5",
}


def load_case(name: str) -> BenchmarkCase:
    """Load one QCNN benchmark case by filename."""
    return BenchmarkCase.load_from_database(QCNN_CASES / name)


def iter_qcnn_case_paths() -> list[Path]:
    """Return all QCNN case files in sorted order."""
    return sorted(QCNN_CASES.glob("*.json"))


def assert_unmeasured(circuit) -> None:
    """Assert a circuit has no measurements or classical bits."""
    assert circuit.num_clbits == 0
    assert all(inst.operation.name != "measure" for inst in circuit.data)


def exact_histograms(runner: QcnnRunner, benchmark_case: BenchmarkCase) -> list[dict[str, float]]:
    """Build exact probability histograms for one QCNN benchmark case."""
    histograms = []
    for circuit in runner.get_benchmark_circuits(benchmark_case):
        probabilities = Statevector.from_instruction(circuit).probabilities_dict()
        histograms.append(
            {bitstring: float(prob) for bitstring, prob in probabilities.items() if prob > 1e-16}
        )
    return histograms


def test_qcnn_registry_lists_runner_and_cases() -> None:
    benchmarks = _discover_builtin_benchmarks()

    assert benchmarks["machine_learning"]["runners"] == ["qcnn"]

    case_ids = {
        Path(case["file"]).name: case["uuid"]
        for case in benchmarks["machine_learning"]["benchmark_cases"]
    }
    assert case_ids == EXPECTED_CASES


def test_qcnn_loader_imports_runner() -> None:
    runner = _load_builtin_runner("machine_learning", "qcnn")

    assert runner.name() == "qcnn"
    assert runner.benchmark_category == "machine_learning"


def test_qcnn_uuid_lookup_resolves_new_category() -> None:
    result = _find_benchmark_case_by_uuid("6d522c6b")

    assert result is not None
    problem_path, category, runner_name = result
    assert problem_path == QCNN_CASES / "qcnn_09q_digits_1_0.instance.json"
    assert category == "machine_learning"
    assert runner_name == "qcnn"


def test_qcnn_cases_are_well_formed() -> None:
    paths = iter_qcnn_case_paths()

    assert [path.name for path in paths] == sorted(EXPECTED_CASES)

    asset_root = get_asset_root()
    for path in paths:
        benchmark_case = BenchmarkCase.load_from_database(path)

        assert benchmark_case.benchmark_category == "machine_learning"
        assert benchmark_case.problem_type == "qcnn_mnist_binary_classification"
        assert benchmark_case.solution_algorithms == ["qcnn"]
        assert benchmark_case.instance_id == EXPECTED_CASES[path.name]
        assert benchmark_case.num_qubits in {9, 16}

        dataset = benchmark_case.data["dataset"]
        quantum_model = benchmark_case.data["quantum_model"]
        classical_model = benchmark_case.data["classical_model"]
        assets = benchmark_case.data["assets"]

        assert dataset["evaluation_examples"] == 50
        assert dataset["resize_to"][0] * dataset["resize_to"][1] == benchmark_case.num_qubits
        assert dataset["label_values"] == [0, 1]
        assert dataset["resize_mode"] in {
            "pillow_float_bilinear",
            "identity_or_pillow_float_bilinear",
        }
        assert sum(dataset["evaluation_label_histogram"]) == dataset["evaluation_examples"]

        assert quantum_model["measurement_observables"] == ["X", "Y", "Z"]
        assert quantum_model["trained_ansatz_bound"] is True
        assert (asset_root / quantum_model["trained_ansatz_dir"]).is_dir()

        assert classical_model["parameter_source"] == "parameter_file_tail"
        assert classical_model["activation"] == "relu_after_output_layer"
        assert (
            classical_model["consumed_parameter_count"]
            + classical_model["unused_parameter_prefix_count"]
            == classical_model["total_parameter_file_entries"]
        )

        assert (asset_root / assets["test_images_file"]).is_file()
        assert (asset_root / assets["test_labels_file"]).is_file()
        assert (asset_root / classical_model["parameter_file"]).is_file()


@pytest.mark.parametrize(
    ("case_name", "expected_shape"),
    [
        ("qcnn_09q_digits_1_0.instance.json", (50, 9)),
        ("qcnn_16q_digits_1_7.instance.json", (50, 16)),
    ],
)
def test_qcnn_encoded_images_have_expected_shape_and_unit_norms(
    case_name: str,
    expected_shape: tuple[int, int],
) -> None:
    encoded_images = load_encoded_images(load_case(case_name))

    assert encoded_images.shape == expected_shape
    assert np.allclose(np.linalg.norm(encoded_images, axis=1), 1.0)


def test_qcnn_runner_builds_150_unmeasured_circuits_for_each_case() -> None:
    runner = QcnnRunner()

    for case_name in EXPECTED_CASES:
        benchmark_case = load_case(case_name)
        circuits = runner.get_benchmark_circuits(benchmark_case)

        assert len(circuits) == 150
        assert all(circuit.num_qubits == benchmark_case.num_qubits for circuit in circuits)
        for circuit in circuits:
            assert_unmeasured(circuit)


@pytest.mark.parametrize(
    "case_name",
    [
        "qcnn_09q_digits_1_0.instance.json",
        "qcnn_09q_digits_1_7.instance.json",
    ],
)
def test_qcnn_9q_merit_figures_match_exact_reference(case_name: str) -> None:
    benchmark_case = load_case(case_name)
    runner = QcnnRunner()

    merit = runner.merit_figures_from_measurements(exact_histograms(runner, benchmark_case), benchmark_case)

    assert merit["score"] == pytest.approx(0.98, abs=1e-12)
    assert merit["num_examples"] == 50
    assert merit["confusion_matrix"] == [[49, 1], [0, 0]]
    assert merit["predictions"].count(0) == 49
    assert merit["predictions"].count(1) == 1


def test_qcnn_runner_executes_end_to_end_with_mock_backend() -> None:
    runner = QcnnRunner()
    benchmark_case = load_case("qcnn_16q_digits_1_7.instance.json")

    record = runner.run_benchmark(benchmark_case, MockBackend(deterministic=True), shots=128)

    assert record.benchmark_category == "machine_learning"
    assert record.solution_algorithm == "qcnn"
    assert record.num_qubits == 16
    assert record.total_shots == 128 * 150
    assert 0.0 <= record.score <= 1.0
    assert len(record.problem_specific_data["predictions"]) == 50
    assert len(record.problem_specific_data["observable_triplets"]) == 50
    assert len(record.problem_specific_data["classifier_outputs"]) == 50
    assert record.problem_specific_data["num_examples"] == 50
