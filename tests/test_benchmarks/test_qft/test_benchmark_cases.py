"""
Tests for QFT benchmark case loading.

Copyright (c) 2025 IonQ, Inc. All rights reserved.
Licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0
International (CC BY-NC-ND 4.0) License.
For details, visit: https://creativecommons.org/licenses/by-nc-nd/4.0/
"""

import json
from pathlib import Path

from apps_benchmark.primitives.benchmark_case import BenchmarkCase

REPO_ROOT = Path(__file__).resolve().parents[3]
QFT_CASES = REPO_ROOT / "apps_benchmark" / "benchmarks" / "qft" / "benchmark_cases"


class TestQftBenchmarkCases:
    """Tests for built-in QFT benchmark cases."""

    def test_qft_case_loads_with_final_public_algorithm_names(self) -> None:
        problem = BenchmarkCase.load_from_database(QFT_CASES / "qft_10_high_freq_challenge.json")

        assert problem.benchmark_category == "qft"
        assert problem.problem_type == "qft"
        assert problem.instance_name == "10_qubit_challenge"
        assert problem.num_qubits == 10
        assert problem.solution_algorithms == [
            "cosine_qft",
            "hidden_phase_qft",
        ]
        assert problem.data["frequency_index"] == 511
        assert problem.data["phase_index"] == 155
        assert problem.instance_id == "f75ae75f"

    def test_all_qft_cases_use_rewritten_category_and_algorithms(self) -> None:
        case_paths = sorted(QFT_CASES.glob("qft_*.json"))

        assert len(case_paths) == 25
        for path in case_paths:
            with open(path) as f:
                payload = json.load(f)

            assert payload["benchmark_category"] == "qft"
            assert payload["solution_algorithms"] == [
                "cosine_qft",
                "hidden_phase_qft",
            ]
            assert sorted(payload["data"]) == ["frequency_index", "phase_index"]
