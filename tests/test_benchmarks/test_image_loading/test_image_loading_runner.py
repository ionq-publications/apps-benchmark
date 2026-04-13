"""
Tests for the image-loading benchmark runner.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from apps_benchmark.cli import _find_benchmark_case_by_uuid
from apps_benchmark.backends.mock_backend import MockBackend
from apps_benchmark.benchmarks.image_loading.algorithms.image_loading_runner import (
    ImageLoadingRunner,
)
from apps_benchmark.core.registry import list_builtin_benchmarks
from apps_benchmark.primitives.benchmark_case import BenchmarkCase
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
IMAGE_LOADING_ROOT = REPO_ROOT / "apps_benchmark" / "benchmarks" / "image_loading"
IMAGE_LOADING_ASSETS = IMAGE_LOADING_ROOT / "assets"
IMAGE_LOADING_CASES = (
    IMAGE_LOADING_ROOT / "benchmark_cases"
)
IMAGE_LOADING_MANIFEST = IMAGE_LOADING_ROOT / "asset_manifest.json"


def load_case(name: str) -> BenchmarkCase:
    return BenchmarkCase.load_from_database(IMAGE_LOADING_CASES / name)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_image_loading_category_is_discovered() -> None:
    builtin = list_builtin_benchmarks()

    assert "image_loading" in builtin
    assert "image_loading" in builtin["image_loading"]["runners"]
    discovered_case_names = {case["name"] for case in builtin["image_loading"]["benchmark_cases"]}
    assert discovered_case_names == {"mnist_5", "imagenet_sketch_shark"}


def test_case_surface_matches_two_image_benchmark() -> None:
    mnist_case = load_case("mnist_5.json")
    shark_case = load_case("imagenet_sketch_shark.json")

    assert mnist_case.instance_name == "mnist_5"
    assert shark_case.instance_name == "imagenet_sketch_shark"
    assert mnist_case.instance_id == "8f0d4a61"
    assert shark_case.instance_id == "9a17b3ce"
    assert mnist_case.data["recommended_minimum_shots_per_qc"] == 10_000
    assert shark_case.data["recommended_minimum_shots_per_qc"] == 10_000
    assert mnist_case.data["image_source_dataset"] == "MNIST"
    assert shark_case.data["image_source_dataset"] == "ImageNet-Sketch"
    assert [entry["depth"] for entry in mnist_case.data["depth_variants"]] == [1, 2, 3, 4, 5, 6]
    assert [entry["depth"] for entry in shark_case.data["depth_variants"]] == [3, 4, 5, 6, 7, 9, 11, 13]


def test_case_uuid_lookup_is_stable() -> None:
    mnist_lookup = _find_benchmark_case_by_uuid("8f0d4a61")
    shark_lookup = _find_benchmark_case_by_uuid("9a17b3ce")

    assert mnist_lookup is not None
    assert shark_lookup is not None

    mnist_path, mnist_category = mnist_lookup
    shark_path, shark_category = shark_lookup

    assert mnist_path == IMAGE_LOADING_CASES / "mnist_5.json"
    assert shark_path == IMAGE_LOADING_CASES / "imagenet_sketch_shark.json"
    assert mnist_category == "image_loading"
    assert shark_category == "image_loading"


def test_image_loading_asset_hashes_match_manifest() -> None:
    manifest = json.loads(IMAGE_LOADING_MANIFEST.read_text())

    for image_info in manifest["images"].values():
        local_image = IMAGE_LOADING_ASSETS / image_info["image_name"]
        assert local_image.exists()
        assert sha256_file(local_image) == image_info["sha256"]

        for variant in image_info["depth_cases"]:
            local_circuit = IMAGE_LOADING_ASSETS / variant["circuit_name"]
            assert local_circuit.exists()
            assert sha256_file(local_circuit) == variant["sha256"]

    for asset in manifest["extra_assets"]:
        local_asset = IMAGE_LOADING_ASSETS / asset["name"]
        assert local_asset.exists()
        assert sha256_file(local_asset) == asset["sha256"]


def test_runner_loads_one_circuit_per_depth() -> None:
    runner = ImageLoadingRunner()
    mnist_case = load_case("mnist_5.json")
    shark_case = load_case("imagenet_sketch_shark.json")

    mnist_circuits = runner.get_benchmark_circuits(mnist_case)
    shark_circuits = runner.get_benchmark_circuits(shark_case)

    assert len(mnist_circuits) == 6
    assert len(shark_circuits) == 8
    assert runner.name() == "image_loading"
    assert runner.benchmark_category == "image_loading"


def test_target_distribution_uses_l2_normalization(tmp_path: Path) -> None:
    image_path = tmp_path / "tiny.png"
    Image.fromarray(np.array([[1, 2], [3, 4]], dtype=np.uint8)).save(image_path)

    benchmark_case = BenchmarkCase(
        benchmark_category="image_loading",
        problem_type="image_loading",
        instance_name="tiny",
        instance_id="tiny_case",
        num_qubits=2,
        solution_algorithms=["image_loading"],
        data={
            "image_dimensions": [2, 2],
            "image_asset": str(image_path),
        },
    )
    runner = ImageLoadingRunner()
    runner._resolve_asset_path = lambda relative_path: Path(relative_path)  # type: ignore[method-assign]

    distribution = runner._load_target_distribution(benchmark_case)

    assert np.isclose(np.linalg.norm(distribution), 1.0)
    assert np.allclose(distribution, np.array([1.0, 2.0, 3.0, 4.0]) / np.sqrt(30.0))


def test_shipped_mnist_numeric_contract_is_stable() -> None:
    runner = ImageLoadingRunner()
    benchmark_case = load_case("mnist_5.json")
    benchmark_case.data["depth_variants"] = [
        {"depth": 1, "circuit": "unused.qpy"},
        {"depth": 6, "circuit": "unused.qpy"},
    ]

    merit = runner.merit_figures_from_measurements(
        [
            {
                "0000000000": 1000,
                "0000000001": 250,
                "0000011111": 125,
                "0000100000": 400,
                "0111111111": 300,
                "1010111100": 175,
                "1111111111": 50,
            },
            {
                "0000000101": 1000,
                "0000010001": 500,
                "0000100001": 250,
                "0010000000": 125,
                "1100001001": 300,
                "1110000100": 200,
                "1111111111": 50,
            },
        ],
        benchmark_case,
    )

    assert merit["best_depth"] == 6
    assert np.isclose(merit["per_depth_metrics"][0]["mse"], 2.0, atol=1e-12)
    assert np.isclose(merit["per_depth_metrics"][1]["mse"], 1.9651675044232726, atol=1e-12)
    assert np.isclose(merit["score"], 1.9651675044232726, atol=1e-12)


def test_merit_figures_choose_lowest_mse_depth(tmp_path: Path) -> None:
    image_path = tmp_path / "tiny.png"
    Image.fromarray(np.array([[10, 0], [0, 0]], dtype=np.uint8)).save(image_path)

    benchmark_case = BenchmarkCase(
        benchmark_category="image_loading",
        problem_type="image_loading",
        instance_name="tiny",
        instance_id="tiny_case",
        num_qubits=2,
        solution_algorithms=["image_loading"],
        data={
            "image_dimensions": [2, 2],
            "image_asset": str(image_path),
            "depth_variants": [
                {"depth": 1, "circuit": "unused.qpy"},
                {"depth": 2, "circuit": "unused.qpy"},
            ],
            "pass_threshold_mse": 0.1,
        },
    )
    runner = ImageLoadingRunner()
    runner._resolve_asset_path = lambda relative_path: Path(relative_path)  # type: ignore[method-assign]

    merit = runner.merit_figures_from_measurements(
        [{"00": 10}, {"11": 10}],
        benchmark_case,
    )

    assert np.isclose(merit["score"], 0.0)
    assert merit["best_depth"] == 1
    assert merit["passes_threshold"] is True
    assert [entry["depth"] for entry in merit["per_depth_metrics"]] == [1, 2]


def test_missing_depth_variants_is_rejected(tmp_path: Path) -> None:
    image_path = tmp_path / "tiny.png"
    Image.fromarray(np.array([[1, 2], [3, 4]], dtype=np.uint8)).save(image_path)

    benchmark_case = BenchmarkCase(
        benchmark_category="image_loading",
        problem_type="image_loading",
        instance_name="tiny",
        instance_id="tiny_case",
        num_qubits=2,
        solution_algorithms=["image_loading"],
        data={
            "image_dimensions": [2, 2],
            "image_asset": str(image_path),
        },
    )

    with pytest.raises(ValueError, match="depth_variants"):
        ImageLoadingRunner().get_benchmark_circuits(benchmark_case)


def test_image_dimensions_must_match_qubits(tmp_path: Path) -> None:
    image_path = tmp_path / "tiny.png"
    Image.fromarray(np.array([[1, 2], [3, 4]], dtype=np.uint8)).save(image_path)

    benchmark_case = BenchmarkCase(
        benchmark_category="image_loading",
        problem_type="image_loading",
        instance_name="tiny",
        instance_id="tiny_case",
        num_qubits=3,
        solution_algorithms=["image_loading"],
        data={
            "image_dimensions": [2, 2],
            "image_asset": str(image_path),
        },
    )
    runner = ImageLoadingRunner()
    runner._resolve_asset_path = lambda relative_path: Path(relative_path)  # type: ignore[method-assign]

    with pytest.raises(ValueError, match="Image dimensions do not match"):
        runner._load_target_distribution(benchmark_case)


def test_zero_norm_target_distribution_is_rejected(tmp_path: Path) -> None:
    image_path = tmp_path / "tiny.png"
    Image.fromarray(np.zeros((2, 2), dtype=np.uint8)).save(image_path)

    benchmark_case = BenchmarkCase(
        benchmark_category="image_loading",
        problem_type="image_loading",
        instance_name="tiny",
        instance_id="tiny_case",
        num_qubits=2,
        solution_algorithms=["image_loading"],
        data={
            "image_dimensions": [2, 2],
            "image_asset": str(image_path),
        },
    )
    runner = ImageLoadingRunner()
    runner._resolve_asset_path = lambda relative_path: Path(relative_path)  # type: ignore[method-assign]

    with pytest.raises(ValueError, match="non-positive norm"):
        runner._load_target_distribution(benchmark_case)


def test_measurement_count_must_match_depth_variants(tmp_path: Path) -> None:
    image_path = tmp_path / "tiny.png"
    Image.fromarray(np.array([[1, 0], [0, 0]], dtype=np.uint8)).save(image_path)

    benchmark_case = BenchmarkCase(
        benchmark_category="image_loading",
        problem_type="image_loading",
        instance_name="tiny",
        instance_id="tiny_case",
        num_qubits=2,
        solution_algorithms=["image_loading"],
        data={
            "image_dimensions": [2, 2],
            "image_asset": str(image_path),
            "depth_variants": [
                {"depth": 1, "circuit": "unused.qpy"},
                {"depth": 2, "circuit": "unused.qpy"},
            ],
        },
    )
    runner = ImageLoadingRunner()
    runner._resolve_asset_path = lambda relative_path: Path(relative_path)  # type: ignore[method-assign]

    with pytest.raises(ValueError, match="expected one measurement histogram per configured depth"):
        runner.merit_figures_from_measurements([{"00": 10}], benchmark_case)


def test_non_numeric_pass_threshold_is_rejected(tmp_path: Path) -> None:
    image_path = tmp_path / "tiny.png"
    Image.fromarray(np.array([[1, 0], [0, 0]], dtype=np.uint8)).save(image_path)

    benchmark_case = BenchmarkCase(
        benchmark_category="image_loading",
        problem_type="image_loading",
        instance_name="tiny",
        instance_id="tiny_case",
        num_qubits=2,
        solution_algorithms=["image_loading"],
        data={
            "image_dimensions": [2, 2],
            "image_asset": str(image_path),
            "depth_variants": [{"depth": 1, "circuit": "unused.qpy"}],
            "pass_threshold_mse": "0.1",
        },
    )
    runner = ImageLoadingRunner()
    runner._resolve_asset_path = lambda relative_path: Path(relative_path)  # type: ignore[method-assign]

    with pytest.raises(ValueError, match="pass_threshold_mse"):
        runner.merit_figures_from_measurements([{"00": 10}], benchmark_case)


def test_end_to_end_mock_backend_runs_image_loading_cases() -> None:
    runner = ImageLoadingRunner()

    for case_name, expected_depths in [
        ("mnist_5.json", 6),
        ("imagenet_sketch_shark.json", 8),
    ]:
        benchmark_case = load_case(case_name)
        record = runner.run_benchmark(benchmark_case, MockBackend(), shots=1000)

        assert record.status == "done"
        assert record.solution_algorithm == "image_loading"
        assert len(record.measurements) == expected_depths
        assert np.isfinite(record.score)
        assert record.problem_specific_data["best_depth"] in [
            entry["depth"] for entry in benchmark_case.data["depth_variants"]
        ]
