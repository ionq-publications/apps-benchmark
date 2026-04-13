"""
Tests for abstract benchmark runner interface.

This module tests the AbstractAlgoRunner and BenchmarkSubmissionRecord classes.

Copyright (c) 2025 IonQ, Inc. All rights reserved.
Licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0
International (CC BY-NC-ND 4.0) License.
For details, visit: https://creativecommons.org/licenses/by-nc-nd/4.0/
"""

from typing import Any, Dict, Tuple

import pandas as pd
import pytest
from apps_benchmark.core.benchmark import AbstractAlgoRunner, BenchmarkSubmissionRecord
from apps_benchmark.primitives.benchmark_case import BenchmarkCase


class ConcreteRunner(AbstractAlgoRunner):
    """
    Minimal concrete benchmark runner for testing.
    """

    def name(self) -> str:
        return "test_runner"

    def setup_algo_inputs(self, benchmark_case: BenchmarkCase) -> Tuple[Any, ...]:
        # Simple setup - return some algorithm inputs
        num_qubits = benchmark_case.num_qubits
        test_data = benchmark_case.data.get("test_value", 42)
        return (num_qubits, test_data)

    def execute_benchmark_algo(
        self,
        algo_inputs: tuple[Any, ...],
        backend: Any,
        shots: int,
        **kwargs: Any,
    ) -> Any:
        # Simple execution - return mock results
        num_qubits, test_data = algo_inputs
        return {
            "energy": -1.5,
            "iterations": 10,
            "num_qubits": num_qubits,
            "test_data": test_data,
        }

    def compute_merit_figures(
        self, algo_output: Any, benchmark_case: BenchmarkCase
    ) -> Dict[str, Any]:
        # Simple merit computation
        return {
            "total_shots": 10000,
            "score": 0.95,
            "final_energy": algo_output["energy"],
            "iterations": algo_output["iterations"],
        }


class MockBackend:
    """Mock backend for testing."""

    def name(self) -> str:
        return "mock_backend"


class TestBenchmarkSubmissionRecord:
    """Tests for BenchmarkSubmissionRecord dataclass."""

    def test_record_creation(self):
        """Test creating a benchmark submission record."""
        start = pd.Timestamp.now(tz="UTC")
        end = start + pd.Timedelta(seconds=30)

        record = BenchmarkSubmissionRecord(
            benchmark_category="chemistry",
            problem_type="vqe",
            instance_name="h2_molecule",
            instance_id="abc123",
            solution_algorithm="vqe_puccd",
            num_qubits=4,
            backend="test_backend",
            shots_per_qc=1000,
            total_shots=10000,
            start_time=start,
            end_time=end,
            time_to_soln=end - start,
            adjusted_tts=end - start,
            last_retrieval=end,
            status="done",
            score=0.98,
            problem_specific_data={"energy": -1.5},
        )

        assert record.benchmark_category == "chemistry"
        assert record.problem_type == "vqe"
        assert record.instance_name == "h2_molecule"
        assert record.instance_id == "abc123"
        assert record.solution_algorithm == "vqe_puccd"
        assert record.num_qubits == 4
        assert record.backend == "test_backend"
        assert record.shots_per_qc == 1000
        assert record.total_shots == 10000
        assert record.status == "done"
        assert record.score == 0.98
        assert record.adjusted_tts == end - start
        assert record.last_retrieval == end
        assert record.problem_specific_data == {"energy": -1.5}

    def test_record_time_calculations(self):
        """Test that time calculations work correctly."""
        start = pd.Timestamp("2026-01-01 10:00:00", tz="UTC")
        end = pd.Timestamp("2026-01-01 10:05:00", tz="UTC")

        record = BenchmarkSubmissionRecord(
            benchmark_category="chemistry",
            problem_type="vqe",
            instance_name="test",
            instance_id="test123",
            solution_algorithm="vqe",
            num_qubits=2,
            backend="test",
            shots_per_qc=1000,
            total_shots=1000,
            start_time=start,
            end_time=end,
            time_to_soln=end - start,
            adjusted_tts=end - start,
            last_retrieval=end,
            status="done",
            score=1.0,
        )

        assert record.time_to_soln == pd.Timedelta(minutes=5)
        assert record.time_to_soln.total_seconds() == 300

    def test_record_default_problem_specific_data(self):
        """Test that problem_specific_data defaults to empty dict."""
        start = pd.Timestamp.now(tz="UTC")
        end = start + pd.Timedelta(seconds=1)

        record = BenchmarkSubmissionRecord(
            benchmark_category="test",
            problem_type="test",
            instance_name="test",
            instance_id="test",
            solution_algorithm="test",
            num_qubits=2,
            backend="test",
            shots_per_qc=1000,
            total_shots=1000,
            start_time=start,
            end_time=end,
            time_to_soln=end - start,
            adjusted_tts=end - start,
            last_retrieval=end,
            status="done",
            score=1.0,
        )

        assert record.problem_specific_data == {}
        assert isinstance(record.problem_specific_data, dict)


class TestAbstractAlgoRunner:
    """Tests for AbstractAlgoRunner class."""

    def test_concrete_runner_instantiation(self):
        """Test that a concrete runner can be instantiated."""
        runner = ConcreteRunner()
        assert isinstance(runner, AbstractAlgoRunner)

    def test_runner_name(self):
        """Test runner name method."""
        runner = ConcreteRunner()
        assert runner.name() == "test_runner"

    def test_setup_algo_inputs(self):
        """Test setup_algo_inputs method."""
        runner = ConcreteRunner()
        problem = BenchmarkCase(
            benchmark_category="chemistry",
            problem_type="vqe",
            instance_name="test",
            num_qubits=4,
            solution_algorithms=["test_runner"],
            data={"test_value": 100},
        )

        inputs = runner.setup_algo_inputs(problem)

        assert isinstance(inputs, tuple)
        assert len(inputs) == 2
        assert inputs[0] == 4  # num_qubits
        assert inputs[1] == 100  # test_value

    def test_execute_benchmark_algo(self):
        """Test execute_benchmark_algo method."""
        runner = ConcreteRunner()
        backend = MockBackend()
        algo_inputs = (4, 100)

        result = runner.execute_benchmark_algo(algo_inputs, backend, 1000)

        assert isinstance(result, dict)
        assert "energy" in result
        assert result["energy"] == -1.5

    def test_compute_merit_figures(self):
        """Test compute_merit_figures method."""
        runner = ConcreteRunner()
        problem = BenchmarkCase(
            benchmark_category="chemistry",
            problem_type="vqe",
            instance_name="test",
            num_qubits=4,
            solution_algorithms=["test_runner"],
            data={},
        )

        algo_output = {"energy": -1.5, "iterations": 10}
        merit = runner.compute_merit_figures(algo_output, problem)

        assert isinstance(merit, dict)
        assert "total_shots" in merit
        assert "score" in merit
        assert merit["total_shots"] == 10000
        assert merit["score"] == 0.95

    def test_benchmark_category_property(self):
        """Test that benchmark_category is auto-derived from module path."""
        runner = ConcreteRunner()

        # The runner is in tests.test_core.test_benchmark_interface
        # So benchmark_category should extract the third-to-last part
        category = runner.benchmark_category

        # In test context, this will be "test_core" based on the module path
        assert isinstance(category, str)

    def test_run_benchmark_end_to_end(self):
        """Test full run_benchmark orchestration."""
        runner = ConcreteRunner()
        backend = MockBackend()
        problem = BenchmarkCase(
            benchmark_category="chemistry",
            problem_type="vqe",
            instance_name="h2_test",
            num_qubits=4,
            solution_algorithms=["test_runner"],
            data={"test_value": 42},
            instance_id="test_id_123",
        )

        record = runner.run_benchmark(problem, backend, shots=1000)

        # Check record structure
        assert isinstance(record, BenchmarkSubmissionRecord)
        assert record.problem_type == "vqe"
        assert record.instance_name == "h2_test"
        assert record.instance_id == "test_id_123"
        assert record.solution_algorithm == "test_runner"
        assert record.num_qubits == 4
        assert record.backend == "mock_backend"
        assert record.shots_per_qc == 1000
        assert record.total_shots == 10000
        assert record.status == "done"
        assert record.score == 0.95

        # Check timing
        assert isinstance(record.start_time, pd.Timestamp)
        assert isinstance(record.end_time, pd.Timestamp)
        assert isinstance(record.time_to_soln, pd.Timedelta)
        assert isinstance(record.adjusted_tts, pd.Timedelta)
        assert isinstance(record.last_retrieval, pd.Timestamp)
        assert record.end_time >= record.start_time
        assert record.time_to_soln >= pd.Timedelta(0)
        assert record.adjusted_tts == record.time_to_soln
        assert record.last_retrieval == record.end_time

        # Check problem-specific data
        assert "final_energy" in record.problem_specific_data
        assert "iterations" in record.problem_specific_data
        assert record.problem_specific_data["final_energy"] == -1.5
        assert record.problem_specific_data["iterations"] == 10

        # Ensure total_shots and score not in problem_specific_data
        assert "total_shots" not in record.problem_specific_data
        assert "score" not in record.problem_specific_data

    def test_run_benchmark_with_kwargs(self):
        """Test run_benchmark with additional kwargs."""

        class KwargsRunner(AbstractAlgoRunner):
            """Runner that uses kwargs."""

            def name(self) -> str:
                return "kwargs_runner"

            def setup_algo_inputs(self, benchmark_case):
                return (benchmark_case.num_qubits,)

            def execute_benchmark_algo(self, algo_inputs, backend, shots, **kwargs):
                # Use kwargs
                optimizer = kwargs.get("optimizer", "COBYLA")
                max_iter = kwargs.get("max_iter", 100)
                return {"optimizer": optimizer, "max_iter": max_iter}

            def compute_merit_figures(self, algo_output, benchmark_case):
                return {
                    "total_shots": 1000,
                    "score": 1.0,
                    "optimizer_used": algo_output["optimizer"],
                }

        runner = KwargsRunner()
        backend = MockBackend()
        problem = BenchmarkCase(
            benchmark_category="test",
            problem_type="test",
            instance_name="test",
            num_qubits=2,
            solution_algorithms=["kwargs_runner"],
            data={},
        )

        record = runner.run_benchmark(problem, backend, shots=1000, optimizer="SPSA", max_iter=50)

        assert record.problem_specific_data["optimizer_used"] == "SPSA"

    def test_cannot_instantiate_abstract_runner(self):
        """Test that AbstractAlgoRunner cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            AbstractAlgoRunner()

    def test_runner_missing_abstract_methods(self):
        """Test that runner with missing methods cannot be instantiated."""

        class IncompleteRunner(AbstractAlgoRunner):
            """Runner missing some required methods."""

            def name(self) -> str:
                return "incomplete"

            def setup_algo_inputs(self, benchmark_case):
                return ()

            # Missing: execute_benchmark_algo and compute_merit_figures

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteRunner()


class TestRunBenchmarkOrchestration:
    """Tests for run_benchmark orchestration logic."""

    def test_orchestration_calls_methods_in_order(self):
        """Test that run_benchmark calls methods in correct order."""
        call_order = []

        class OrderedRunner(AbstractAlgoRunner):
            """Runner that tracks method call order."""

            def name(self) -> str:
                return "ordered"

            def setup_algo_inputs(self, benchmark_case):
                call_order.append("setup")
                return (benchmark_case.num_qubits,)

            def execute_benchmark_algo(self, algo_inputs, backend, shots, **kwargs):
                call_order.append("execute")
                return {"result": 42}

            def compute_merit_figures(self, algo_output, benchmark_case):
                call_order.append("compute")
                return {"total_shots": 1000, "score": 1.0}

        runner = OrderedRunner()
        backend = MockBackend()
        problem = BenchmarkCase(
            benchmark_category="test",
            problem_type="test",
            instance_name="test",
            num_qubits=2,
            solution_algorithms=["ordered"],
            data={},
        )

        runner.run_benchmark(problem, backend, shots=1000)

        assert call_order == ["setup", "execute", "compute"]

    def test_orchestration_timing_is_accurate(self):
        """Test that timing captures actual execution duration."""
        import time

        class SlowRunner(AbstractAlgoRunner):
            """Runner that takes measurable time."""

            def name(self) -> str:
                return "slow"

            def setup_algo_inputs(self, benchmark_case):
                return ()

            def execute_benchmark_algo(self, algo_inputs, backend, shots, **kwargs):
                time.sleep(0.1)  # Sleep 100ms
                return {"result": 1}

            def compute_merit_figures(self, algo_output, benchmark_case):
                return {"total_shots": 1000, "score": 1.0}

        runner = SlowRunner()
        backend = MockBackend()
        problem = BenchmarkCase(
            benchmark_category="test",
            problem_type="test",
            instance_name="test",
            num_qubits=2,
            solution_algorithms=["slow"],
            data={},
        )

        record = runner.run_benchmark(problem, backend, shots=1000)

        # Should take at least 100ms
        assert record.time_to_soln >= pd.Timedelta(milliseconds=100)

    def test_orchestration_separates_problem_specific_data(self):
        """Test that problem_specific_data excludes total_shots and score."""
        runner = ConcreteRunner()
        backend = MockBackend()
        problem = BenchmarkCase(
            benchmark_category="test",
            problem_type="test",
            instance_name="test",
            num_qubits=2,
            solution_algorithms=["test_runner"],
            data={},
        )

        record = runner.run_benchmark(problem, backend, shots=1000)

        # Check that required fields are in record
        assert record.total_shots == 10000
        assert record.score == 0.95

        # Check that they're NOT in problem_specific_data
        assert "total_shots" not in record.problem_specific_data
        assert "score" not in record.problem_specific_data

        # Check that other fields ARE in problem_specific_data
        assert "final_energy" in record.problem_specific_data
        assert "iterations" in record.problem_specific_data
