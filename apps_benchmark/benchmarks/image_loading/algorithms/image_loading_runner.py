"""
Image-loading benchmark runner.

Copyright (c) 2025 IonQ, Inc. All rights reserved.
Licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0
International (CC BY-NC-ND 4.0) License.
For details, visit: https://creativecommons.org/licenses/by-nc-nd/4.0/
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import qiskit.qpy as qpy
from PIL import Image
from qiskit import QuantumCircuit, transpile

from apps_benchmark.core.backend import MeasurementBatch
from apps_benchmark.core.qc_benchmark_runner import CircuitBenchmarkRunner
from apps_benchmark.primitives.benchmark_case import BenchmarkCase


class ImageLoadingRunner(CircuitBenchmarkRunner):
    """
    Evaluate image-loading instances using precompiled MPS-based depth sweeps.
    """

    def name(self) -> str:
        return "image_loading"

    def _resolve_asset_path(self, relative_path: str) -> Path:
        return (Path(__file__).resolve().parent / relative_path).resolve()

    @staticmethod
    def _depth_variants(benchmark_case: BenchmarkCase) -> list[dict[str, Any]]:
        variants = benchmark_case.data.get("depth_variants")
        if not isinstance(variants, list) or not variants:
            raise ValueError("Image-loading benchmark case is missing non-empty 'depth_variants'.")
        return variants

    @staticmethod
    def _normalize_l2(values: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(values))
        if norm <= 0:
            raise ValueError("Cannot normalize an image or histogram with non-positive norm.")
        return values / norm

    def _load_target_distribution(self, benchmark_case: BenchmarkCase) -> np.ndarray:
        image_dimensions = benchmark_case.data.get("image_dimensions")
        if not isinstance(image_dimensions, list) or len(image_dimensions) != 2:
            raise ValueError("Image-loading benchmark case must define 'image_dimensions'.")

        width, height = image_dimensions
        if not isinstance(width, int) or not isinstance(height, int):
            raise ValueError("'image_dimensions' entries must be integers.")
        if width * height != 2**benchmark_case.num_qubits:
            raise ValueError(
                "Image dimensions do not match the benchmark qubit count for dense amplitude encoding."
            )

        image_asset = benchmark_case.data.get("image_asset")
        if not isinstance(image_asset, str):
            raise ValueError("Image-loading benchmark case must define an 'image_asset' path.")

        image_path = self._resolve_asset_path(image_asset)
        with Image.open(image_path) as image:
            grayscale = image.convert("L")
            if grayscale.size != (width, height):
                raise ValueError(
                    f"Image asset {image_path.name} has size {grayscale.size}, expected {(width, height)}."
                )
            pixels = np.asarray(grayscale, dtype=float).reshape(-1)

        return self._normalize_l2(pixels)

    @staticmethod
    def _measurement_distribution(
        histogram: dict[str, int | float],
        num_qubits: int,
    ) -> tuple[np.ndarray, float]:
        distribution = np.zeros(2**num_qubits, dtype=float)
        total = 0.0
        for bitstring, count in histogram.items():
            index = int(bitstring.replace(" ", ""), 2)
            value = float(count)
            distribution[index] += value
            total += value

        return ImageLoadingRunner._normalize_l2(distribution), total

    @staticmethod
    def _depth_mse(target_distribution: np.ndarray, measured_distribution: np.ndarray) -> float:
        if target_distribution.shape != measured_distribution.shape:
            raise ValueError("Target and measured distributions must have the same shape.")
        return float(np.sum((measured_distribution - target_distribution) ** 2))

    def get_benchmark_circuits(self, benchmark_case: BenchmarkCase) -> list[QuantumCircuit]:
        circuits: list[QuantumCircuit] = []
        for variant in self._depth_variants(benchmark_case):
            circuit_asset = variant.get("circuit")
            if not isinstance(circuit_asset, str):
                raise ValueError("Each depth variant must define a 'circuit' asset path.")

            circuit_path = self._resolve_asset_path(circuit_asset)
            with circuit_path.open("rb") as handle:
                [loaded_circuit] = qpy.load(handle)

            circuits.append(
                transpile(
                    loaded_circuit,
                    basis_gates=["rx", "ry", "rz", "cx"],
                    optimization_level=1,
                )
            )
        return circuits

    def merit_figures_from_measurements(
        self,
        measurements: MeasurementBatch,
        benchmark_case: BenchmarkCase,
    ) -> dict[str, Any]:
        depth_variants = self._depth_variants(benchmark_case)
        if len(measurements) != len(depth_variants):
            raise ValueError(
                "Image-loading benchmark expected one measurement histogram per configured depth."
            )

        target_distribution = self._load_target_distribution(benchmark_case)
        per_depth_metrics: list[dict[str, Any]] = []

        for histogram, variant in zip(measurements, depth_variants, strict=True):
            depth = variant.get("depth")
            if not isinstance(depth, int):
                raise ValueError("Each depth variant must define an integer 'depth'.")

            measured_distribution, total_counts = self._measurement_distribution(
                histogram,
                benchmark_case.num_qubits,
            )
            per_depth_metrics.append(
                {
                    "depth": depth,
                    "mse": self._depth_mse(target_distribution, measured_distribution),
                    "shots_observed": int(round(total_counts)),
                }
            )

        best_depth_metric = min(per_depth_metrics, key=lambda item: (float(item["mse"]), int(item["depth"])))
        pass_threshold = benchmark_case.data.get("pass_threshold_mse", 0.1)
        if not isinstance(pass_threshold, (int, float)):
            raise ValueError("'pass_threshold_mse' must be numeric when provided.")

        return {
            "score": float(best_depth_metric["mse"]),
            "best_depth": int(best_depth_metric["depth"]),
            "pass_threshold_mse": float(pass_threshold),
            "passes_threshold": float(best_depth_metric["mse"]) < float(pass_threshold),
            "score_direction": benchmark_case.data.get("score_direction", "lower_is_better"),
            "score_aggregation": benchmark_case.data.get("score_aggregation", "min_mse_across_depths"),
            "construction_method": benchmark_case.data.get("construction_method", "mps"),
            "encoding": benchmark_case.data.get("encoding", "dense_amplitude"),
            "per_depth_metrics": per_depth_metrics,
        }
