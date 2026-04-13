"""
Tests for chemistry benchmark case loading.

This module tests loading and validation of chemistry benchmark cases.

Copyright (c) 2025 IonQ, Inc. All rights reserved.
Licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0
International (CC BY-NC-ND 4.0) License.
For details, visit: https://creativecommons.org/licenses/by-nc-nd/4.0/
"""

from pathlib import Path

import pytest

from apps_benchmark.backends.qiskit_aer_sim_backend import QiskitAerSimBackend
from apps_benchmark.benchmarks.chemistry.algorithms.vqe_puccd_runner import VqePuccdRunner
from apps_benchmark.primitives.benchmark_case import BenchmarkCase


class TestChemistryBenchmarkCases:
    """Tests for built-in chemistry benchmark cases."""

    def test_h002_chain_0_75_loads(self):
        """Test that H2 0.75A benchmark case loads correctly."""
        problem_path = (
            Path(__file__).parent.parent.parent.parent
            / "apps_benchmark"
            / "benchmarks"
            / "chemistry"
            / "benchmark_cases"
            / "h002_chain_0_75.json"
        )

        problem = BenchmarkCase.load_from_database(problem_path)

        assert problem.benchmark_category == "chemistry"
        assert problem.problem_type == "hydrogen_lattice_vqe"
        assert problem.instance_name == "h002_chain_0_75"
        assert problem.num_qubits == 2
        assert "vqe_puccd" in problem.solution_algorithms
        assert problem.instance_id == "610cfb55"

    def test_h002_chain_1_00_loads(self):
        """Test that H2 1.00A benchmark case loads correctly."""
        problem_path = (
            Path(__file__).parent.parent.parent.parent
            / "apps_benchmark"
            / "benchmarks"
            / "chemistry"
            / "benchmark_cases"
            / "h002_chain_1_00.json"
        )

        problem = BenchmarkCase.load_from_database(problem_path)

        assert problem.benchmark_category == "chemistry"
        assert problem.problem_type == "hydrogen_lattice_vqe"
        assert problem.instance_name == "h002_chain_1_00"
        assert problem.num_qubits == 2
        assert "vqe_puccd" in problem.solution_algorithms
        assert problem.instance_id == "1bbca30e"

    def test_benchmark_case_data_contains_hamiltonian(self):
        """Test that benchmark case data contains required Hamiltonian."""
        problem_path = (
            Path(__file__).parent.parent.parent.parent
            / "apps_benchmark"
            / "benchmarks"
            / "chemistry"
            / "benchmark_cases"
            / "h002_chain_0_75.json"
        )

        problem = BenchmarkCase.load_from_database(problem_path)

        assert "paired_hamiltonian_dict" in problem.data
        assert isinstance(problem.data["paired_hamiltonian_dict"], dict)
        assert "II" in problem.data["paired_hamiltonian_dict"]
        assert "ZZ" in problem.data["paired_hamiltonian_dict"]

    def test_benchmark_case_data_contains_reference_energy(self):
        """Test that benchmark case data contains reference FCI energy."""
        problem_path = (
            Path(__file__).parent.parent.parent.parent
            / "apps_benchmark"
            / "benchmarks"
            / "chemistry"
            / "benchmark_cases"
            / "h002_chain_0_75.json"
        )

        problem = BenchmarkCase.load_from_database(problem_path)

        assert "reference_energy_fci" in problem.data
        assert isinstance(problem.data["reference_energy_fci"], (int, float))
        assert problem.data["reference_energy_fci"] < 0  # Should be negative

    def test_benchmark_case_data_contains_num_alpha(self):
        """Test that benchmark case data contains number of alpha electrons."""
        problem_path = (
            Path(__file__).parent.parent.parent.parent
            / "apps_benchmark"
            / "benchmarks"
            / "chemistry"
            / "benchmark_cases"
            / "h002_chain_0_75.json"
        )

        problem = BenchmarkCase.load_from_database(problem_path)

        assert "num_alpha" in problem.data
        assert problem.data["num_alpha"] == 1  # H2 has 1 electron pair

    def test_benchmark_case_data_contains_optimizer_config(self):
        """Test that benchmark case data contains optimizer configuration."""
        problem_path = (
            Path(__file__).parent.parent.parent.parent
            / "apps_benchmark"
            / "benchmarks"
            / "chemistry"
            / "benchmark_cases"
            / "h002_chain_0_75.json"
        )

        problem = BenchmarkCase.load_from_database(problem_path)

        assert "optimizer_config" in problem.data
        assert "method" in problem.data["optimizer_config"]
        assert "options" in problem.data["optimizer_config"]

    def test_benchmark_cases_define_problem_specific_shot_defaults(self):
        """Chemistry cases should carry their own shot defaults."""
        for case_file in ALL_CASE_FILES:
            problem = BenchmarkCase.load_from_database(case_file)
            assert problem.data["recommended_minimum_shots_per_qc"] == 10_000

    def test_both_benchmark_cases_loadable(self):
        """Test that both H2 benchmark cases can be loaded."""
        base_path = (
            Path(__file__).parent.parent.parent.parent
            / "apps_benchmark"
            / "benchmarks"
            / "chemistry"
            / "benchmark_cases"
        )

        problem1 = BenchmarkCase.load_from_database(base_path / "h002_chain_0_75.json")
        problem2 = BenchmarkCase.load_from_database(base_path / "h002_chain_1_00.json")

        # Both should load without error
        assert problem1.instance_name == "h002_chain_0_75"
        assert problem2.instance_name == "h002_chain_1_00"

        # Should have different instance IDs
        assert problem1.instance_id != problem2.instance_id

    def test_benchmark_case_geometry_present(self):
        """Test that molecular geometry is present in data."""
        problem_path = (
            Path(__file__).parent.parent.parent.parent
            / "apps_benchmark"
            / "benchmarks"
            / "chemistry"
            / "benchmark_cases"
            / "h002_chain_0_75.json"
        )

        problem = BenchmarkCase.load_from_database(problem_path)

        assert "geometry" in problem.data
        assert isinstance(problem.data["geometry"], list)
        assert len(problem.data["geometry"]) == 2  # Two H atoms

    def test_benchmark_case_description_present(self):
        """Test that benchmark case description is present."""
        problem_path = (
            Path(__file__).parent.parent.parent.parent
            / "apps_benchmark"
            / "benchmarks"
            / "chemistry"
            / "benchmark_cases"
            / "h002_chain_0_75.json"
        )

        problem = BenchmarkCase.load_from_database(problem_path)

        assert "description" in problem.data
        assert "H2" in problem.data["description"] or "molecule" in problem.data["description"]

    def test_benchmark_case_dump_is_pretty_and_stable(self, tmp_path):
        """Test BenchmarkCase.dump writes sorted, indented JSON with a newline."""
        problem = BenchmarkCase(
            benchmark_category="chemistry",
            problem_type="dump_test",
            instance_name="example",
            instance_id="dump1234",
            num_qubits=2,
            solution_algorithms=["vqe_test"],
            data={"z_key": 1, "a_key": (1, 2)},
        )

        dump_path = tmp_path / "dump.json"
        problem.dump(dump_path, converter={"a_key": list})

        text = dump_path.read_text()
        assert text.endswith("\n")
        assert '  "benchmark_category": "chemistry"' in text
        assert '    "a_key": [' in text
        assert text.index('"data"') < text.index('"instance_id"')
        assert text.index('"a_key"') < text.index('"z_key"')

        payload = dump_path.read_text()
        assert '"instance_name": "example"' in payload
        assert '"solution_algorithms": [' in payload


BENCHMARK_CASES_DIR = (
    Path(__file__).parent.parent.parent.parent
    / "apps_benchmark"
    / "benchmarks"
    / "chemistry"
    / "benchmark_cases"
)

ALL_CASE_FILES = sorted(BENCHMARK_CASES_DIR.glob("h*.json"))


_runner = VqePuccdRunner()
_backend = QiskitAerSimBackend()


@pytest.mark.parametrize("case_file", ALL_CASE_FILES, ids=lambda f: f.stem)
def test_vqe_energy_exact_statevector(case_file: Path) -> None:
    """Verify each instance's optimal parameters reproduce its reported VQE energy."""
    case = BenchmarkCase.load_from_database(case_file)
    record = _runner.run_benchmark(case, _backend, shots=1, exact_simulation=True)

    expected = case.data["vqe_final_energy"]
    actual = record.problem_specific_data["final_energy"]
    assert abs(actual - expected) < 1e-8, (
        f"{case.instance_name}: energy mismatch {actual} vs expected {expected}"
    )
