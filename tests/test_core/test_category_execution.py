"""
Tests for category execution functionality.

This module tests the ability to run all benchmarks in a category,
including filtering, progress reporting, and result aggregation.

Copyright (c) 2025 IonQ, Inc. All rights reserved.
Licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0
International (CC BY-NC-ND 4.0) License.
For details, visit: https://creativecommons.org/licenses/by-nc-nd/4.0/
"""

import json
from pathlib import Path

from apps_benchmark.cli import main
from apps_benchmark.core.registry import list_builtin_benchmarks
from click.testing import CliRunner


def chemistry_case_counts(max_qubits: int | None = None) -> tuple[int, int]:
    """Return counts of runnable closed and open-only chemistry cases."""
    builtin = list_builtin_benchmarks()
    cases = builtin["chemistry"]["benchmark_cases"]
    closed = 0
    open_only = 0

    for case_info in cases:
        case_path = Path(case_info["file"])
        with open(case_path) as f:
            data = json.load(f)
        if max_qubits is not None and data["num_qubits"] > max_qubits:
            continue
        if case_info.get("all_solutions_open"):
            open_only += 1
        else:
            closed += 1

    return closed, open_only


class TestCategoryExecution:
    """Tests for running all benchmarks in a category."""

    def test_category_execution_success(self):
        """Test successful execution of all benchmarks in a category."""
        expected_runs, expected_open = chemistry_case_counts(max_qubits=10)
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "run",
                "--backend=mock_backend",
                "--category=chemistry",
                "--qbit-max=10",
                "--shots=100",
            ],
        )

        assert result.exit_code == 0
        assert "Running benchmarks in category 'chemistry'" in result.output
        assert f"Found {expected_runs} benchmark(s) to run" in result.output
        assert (
            f"Found {expected_open} open benchmark case(s) not to run (require user-provided solvers)"
            in result.output
        )
        assert f"[1/{expected_runs}]" in result.output
        assert f"[{expected_runs}/{expected_runs}]" in result.output
        assert "RESULTS SUMMARY" in result.output
        assert f"Total runs:      {expected_runs}" in result.output
        assert f"Completed:       {expected_runs}" in result.output

    def test_category_execution_with_qbit_filter(self):
        """Test that qbit_max filter works correctly."""
        runner = CliRunner()

        # Run with qbit_max=1 (should filter out all benchmarks with 2 qubits)
        result = runner.invoke(
            main,
            [
                "run",
                "--backend=mock_backend",
                "--category=chemistry",
                "--qbit-max=1",
            ],
        )

        assert result.exit_code == 1
        assert "No benchmark cases found with num_qubits <= 1" in result.output

    def test_category_execution_invalid_category(self):
        """Test error handling for non-existent category."""
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "run",
                "--backend=mock_backend",
                "--category=nonexistent",
            ],
        )

        assert result.exit_code == 1
        assert "Category 'nonexistent' not found" in result.output

    def test_category_execution_no_backend(self):
        """Test error handling when backend is not specified."""
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "run",
                "--category=chemistry",
            ],
        )

        assert result.exit_code == 1
        assert "--backend is required" in result.output

    def test_category_execution_shows_progress(self):
        """Test that progress is shown during execution."""
        expected_runs, _ = chemistry_case_counts(max_qubits=10)
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "run",
                "--backend=mock_backend",
                "--category=chemistry",
            ],
        )

        assert result.exit_code == 0
        # Check for progress indicators
        assert f"[1/{expected_runs}]" in result.output
        assert f"[{expected_runs}/{expected_runs}]" in result.output
        assert "done" in result.output

    def test_category_execution_detailed_results(self):
        """Test that detailed results are displayed."""
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "run",
                "--backend=mock_backend",
                "--category=chemistry",
            ],
        )
        assert result.exit_code == 0
        assert "DETAILED RESULTS" in result.output
        assert "Problem" in result.output
        assert "Algorithm" in result.output
        assert "Score" in result.output
        assert "Time (s)" in result.output
        assert "h002_chain_1_00" in result.output
        assert "vqe_puccd" in result.output


class TestCategoryExecutionFailFast:
    """Tests for fail-fast behavior during category execution."""

    def test_category_execution_stops_on_error(self):
        """Test that execution stops immediately on first error."""
        runner = CliRunner()

        # Use a backend that doesn't exist to trigger an error
        result = runner.invoke(
            main,
            [
                "run",
                "--backend=nonexistent_backend",
                "--category=chemistry",
            ],
        )

        assert result.exit_code == 1
        assert "Backend 'nonexistent_backend' not found" in result.output
        # Should not get to the benchmark execution stage
        assert "Running benchmarks in category" not in result.output


class TestCategoryExecutionWithDifferentBackends:
    """Tests for category execution with different backends."""

    def test_category_execution_with_qiskit_aer_sim_backend(self):
        """Test category execution with Qiskit backend."""
        expected_runs, _ = chemistry_case_counts(max_qubits=10)
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "run",
                "--backend=qiskit_aer_sim_backend",
                "--category=chemistry",
                "--shots=10",  # Use fewer shots for speed
            ],
        )

        assert result.exit_code == 0
        assert "Running benchmarks in category 'chemistry'" in result.output
        assert "Backend:         aer_simulator" in result.output
        assert f"Total runs:      {expected_runs}" in result.output


class TestResultAggregation:
    """Tests for result aggregation and summary statistics."""

    def test_summary_statistics_calculation(self):
        """Test that summary statistics are calculated correctly."""
        expected_runs, _ = chemistry_case_counts(max_qubits=10)
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "run",
                "--backend=mock_backend",
                "--category=chemistry",
                "--shots=100",
            ],
        )

        assert result.exit_code == 0
        assert "RESULTS SUMMARY" in result.output
        assert "Category:        chemistry" in result.output
        assert "Backend:         mock" in result.output
        assert f"Total runs:      {expected_runs}" in result.output
        assert f"Completed:       {expected_runs}" in result.output
        assert "Errored:         0" in result.output
        assert "Average score:" in result.output
        assert "Total time:" in result.output
        assert "Total shots:" in result.output

    def test_detailed_results_table_format(self):
        """Test that detailed results table is properly formatted."""
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "run",
                "--backend=mock_backend",
                "--category=chemistry",
            ],
        )

        assert result.exit_code == 0

        # Check table structure
        lines = result.output.split("\n")
        detailed_section = False
        for line in lines:
            if "DETAILED RESULTS" in line:
                detailed_section = True
            if detailed_section and "h002_chain" in line:
                # Verify the line has proper columns
                assert "vqe_puccd" in line
                break


class TestCategoryExecutionEdgeCases:
    """Tests for edge cases in category execution."""

    def test_category_execution_empty_category(self):
        """Test handling of category with no benchmark cases."""
        # This test would require creating a category with no cases
        # For now, we test the error path with qbit filter
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "run",
                "--backend=mock_backend",
                "--category=chemistry",
                "--qbit-max=1",  # Filter out everything
            ],
        )

        assert result.exit_code >= 1, f"Expected error exit for no cases found, output: {result.output}"
        assert "No benchmark cases found" in result.output

    def test_category_execution_with_high_qbit_max(self):
        """Test category execution with very high qbit_max."""
        expected_runs, expected_open = chemistry_case_counts(max_qubits=1000)
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "run",
                "--backend=mock_backend",
                "--category=chemistry",
                "--qbit-max=1000",
            ],
        )

        assert result.exit_code == 0
        assert f"Found {expected_runs} benchmark(s) to run" in result.output
        assert (
            f"Found {expected_open} open benchmark case(s) not to run (require user-provided solvers)"
            in result.output
        )


class TestCategoryListingIntegration:
    """Tests for integration between list and run commands."""

    def test_list_shows_available_categories(self):
        """Test that list command shows categories that can be run."""
        runner = CliRunner()
        list_result = runner.invoke(main, ["list"])

        assert list_result.exit_code == 0
        assert "chemistry" in list_result.output

        # Now verify we can run the listed category
        run_result = runner.invoke(
            main,
            [
                "run",
                "--backend=mock_backend",
                "--category=chemistry",
            ],
        )

        assert run_result.exit_code == 0
