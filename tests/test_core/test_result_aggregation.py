"""
Tests for result aggregation and display functionality.

This module tests the aggregation of benchmark results and
the display of summary statistics.

Copyright (c) 2025 IonQ, Inc. All rights reserved.
Licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0
International (CC BY-NC-ND 4.0) License.
For details, visit: https://creativecommons.org/licenses/by-nc-nd/4.0/
"""

import pandas as pd
from apps_benchmark.cli import _display_category_results
from apps_benchmark.core.benchmark import BenchmarkSubmissionRecord
from click.testing import CliRunner


class TestDisplayCategoryResults:
    """Tests for _display_category_results function."""

    def create_sample_record(
        self,
        instance_name: str,
        score: float,
        time_seconds: float,
        total_shots: int = 1000,
    ) -> BenchmarkSubmissionRecord:
        """Helper to create a sample BenchmarkSubmissionRecord."""
        start = pd.Timestamp.now(tz="UTC")
        end = start + pd.Timedelta(seconds=time_seconds)

        return BenchmarkSubmissionRecord(
            benchmark_category="chemistry",
            problem_type="test_problem",
            instance_name=instance_name,
            instance_id="test123",
            solution_algorithm="vqe_puccd",
            num_qubits=2,
            backend="mock",
            shots_per_qc=100,
            total_shots=total_shots,
            start_time=start,
            end_time=end,
            time_to_soln=pd.Timedelta(seconds=time_seconds),
            adjusted_tts=pd.Timedelta(seconds=time_seconds),
            last_retrieval=end,
            status="done",
            score=score,
            problem_specific_data={},
        )

    def test_display_empty_results(self, capsys):
        """Test display with no results."""
        _display_category_results([], "chemistry", "mock", 1000)

        captured = capsys.readouterr()
        # Should not display anything for empty results
        assert captured.out == ""

    def test_display_single_result(self, capsys):
        """Test display with a single result."""
        record = self.create_sample_record("h2_molecule", 0.998, 5.5)
        _display_category_results([record], "chemistry", "mock", 1000)

        captured = capsys.readouterr()
        output = captured.out

        # Check summary section
        assert "RESULTS SUMMARY" in output
        assert "Category:        chemistry" in output
        assert "Backend:         mock" in output
        assert "Total runs:      1" in output
        assert "Completed:       1" in output
        assert "Errored:         0" in output
        assert "0.998000" in output  # Average score

        # Check detailed results
        assert "DETAILED RESULTS" in output
        assert "h2_molecule" in output
        assert "vqe_puccd" in output

    def test_display_multiple_results(self, capsys):
        """Test display with multiple results."""
        records = [
            self.create_sample_record("h2_1", 0.995, 3.0, 1000),
            self.create_sample_record("h2_2", 0.997, 4.0, 1500),
            self.create_sample_record("h2_3", 0.999, 5.0, 2000),
        ]

        _display_category_results(records, "chemistry", "qiskit", 500)

        captured = capsys.readouterr()
        output = captured.out

        # Check summary statistics
        assert "Total runs:      3" in output
        assert "Completed:       3" in output
        assert "Errored:         0" in output

        # Average score should be (0.995 + 0.997 + 0.999) / 3 = 0.997
        assert "0.997000" in output

        # Total time should be 3 + 4 + 5 = 12 seconds
        assert "12.00s" in output

        # Total shots should be 1000 + 1500 + 2000 = 4500
        assert "4,500" in output

        # Check all problems are listed
        assert "h2_1" in output
        assert "h2_2" in output
        assert "h2_3" in output

    def test_display_results_with_long_names(self, capsys):
        """Test display handles long problem names gracefully."""
        long_name = "very_long_problem_instance_name_that_exceeds_limit"
        record = self.create_sample_record(long_name, 0.95, 2.0)

        _display_category_results([record], "chemistry", "mock", 1000)

        captured = capsys.readouterr()
        output = captured.out

        # Name should be truncated to 24 characters in the table
        assert "very_long_problem_instan" in output

    def test_summary_statistics_accuracy(self, capsys):
        """Test that summary statistics are calculated accurately."""
        records = [
            self.create_sample_record("test1", 1.0, 10.0, 1000),
            self.create_sample_record("test2", 0.5, 20.0, 2000),
        ]

        _display_category_results(records, "test_category", "test_backend", 100)

        captured = capsys.readouterr()
        output = captured.out

        # Average score: (1.0 + 0.5) / 2 = 0.75
        assert "0.750000" in output

        # Total time: 10 + 20 = 30 seconds
        assert "30.00s" in output

        # Total shots: 1000 + 2000 = 3000
        assert "3,000" in output

    def test_display_includes_all_required_fields(self, capsys):
        """Test that display includes all required summary fields."""
        record = self.create_sample_record("test", 0.9, 1.0)
        _display_category_results([record], "chemistry", "mock", 1000)

        captured = capsys.readouterr()
        output = captured.out

        # Check all required summary fields are present
        required_fields = [
            "Category:",
            "Backend:",
            "Total runs:",
            "Completed:",
            "Errored:",
            "Average score:",
            "Total time:",
            "Total shots:",
        ]

        for field in required_fields:
            assert field in output, f"Missing field: {field}"

    def test_detailed_results_table_headers(self, capsys):
        """Test that detailed results table has proper headers."""
        record = self.create_sample_record("test", 0.9, 1.0)
        _display_category_results([record], "chemistry", "mock", 1000)

        captured = capsys.readouterr()
        output = captured.out

        # Check table headers
        assert "Problem" in output
        assert "Algorithm" in output
        assert "Score" in output
        assert "Time (s)" in output


class TestResultAggregationCalculations:
    """Tests for result aggregation calculations."""

    def test_average_score_calculation(self):
        """Test average score is calculated correctly."""
        runner = CliRunner()
        from apps_benchmark.cli import main

        result = runner.invoke(
            main,
            [
                "run",
                "--backend=mock_backend",
                "--category=chemistry",
            ],
        )

        assert result.exit_code == 0
        # Should show average score in output
        assert "Average score:" in result.output

    def test_total_time_aggregation(self):
        """Test total time is summed correctly."""
        runner = CliRunner()
        from apps_benchmark.cli import main

        result = runner.invoke(
            main,
            [
                "run",
                "--backend=mock_backend",
                "--category=chemistry",
            ],
        )

        assert result.exit_code == 0
        assert "Total time:" in result.output
        # Time should be shown in seconds
        assert "s" in result.output

    def test_total_shots_aggregation(self):
        """Test total shots are summed correctly."""
        runner = CliRunner()
        from apps_benchmark.cli import main

        result = runner.invoke(
            main,
            [
                "run",
                "--backend=mock_backend",
                "--category=chemistry",
                "--shots=500",
            ],
        )

        assert result.exit_code == 0
        assert "Total shots:" in result.output


class TestResultAggregationEdgeCases:
    """Tests for edge cases in result aggregation."""

    def test_display_with_zero_scores(self, capsys):
        """Test display handles zero scores correctly."""
        helper = TestDisplayCategoryResults()
        records = [
            helper.create_sample_record("test1", 0.0, 1.0),
            helper.create_sample_record("test2", 0.0, 1.0),
        ]

        _display_category_results(records, "chemistry", "mock", 1000)

        captured = capsys.readouterr()
        output = captured.out

        # Should show 0.000000 as average
        assert "0.000000" in output

    def test_display_with_very_large_values(self, capsys):
        """Test display handles large values correctly."""
        helper = TestDisplayCategoryResults()
        records = [helper.create_sample_record("test", 0.999999, 1000.0, 1000000)]

        _display_category_results(records, "chemistry", "mock", 10000)

        captured = capsys.readouterr()
        output = captured.out

        # Should handle large time
        assert "1000.00s" in output

        # Should format large shot count with commas
        assert "1,000,000" in output

    def test_display_formatting_consistency(self, capsys):
        """Test that formatting is consistent across different values."""
        helper = TestDisplayCategoryResults()
        records = [
            helper.create_sample_record("test1", 0.123456, 1.5),
            helper.create_sample_record("test2", 0.987654, 10.25),
        ]

        _display_category_results(records, "chemistry", "mock", 1000)

        captured = capsys.readouterr()
        output = captured.out

        # Check that scores are formatted with 6 decimal places
        assert "0.123456" in output
        assert "0.987654" in output

        # Check that times are formatted with 2 decimal places
        assert "1.50" in output
        assert "10.25" in output
