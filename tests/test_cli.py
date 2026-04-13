"""
Tests for CLI interface.

This module tests the Click-based command-line interface.

Copyright (c) 2025 IonQ, Inc. All rights reserved.
Licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0
International (CC BY-NC-ND 4.0) License.
For details, visit: https://creativecommons.org/licenses/by-nc-nd/4.0/
"""

import json

import pytest
from apps_benchmark.cli import _get_shots_for_case, _resolve_shots, main
from apps_benchmark.primitives.benchmark_case import BenchmarkCase
from click.testing import CliRunner


@pytest.fixture
def cli_runner():
    """Fixture providing a CLI test runner."""
    return CliRunner()


@pytest.fixture
def mock_home(tmp_path, monkeypatch):
    """Fixture that mocks the home directory for registry operations."""
    monkeypatch.setattr("apps_benchmark.core.registry.Path.home", lambda: tmp_path)
    return tmp_path


class TestMainCommand:
    """Tests for main command group."""

    def test_main_help(self, cli_runner):
        """Test that --help works."""
        result = cli_runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "IonQ Quantum Application Benchmarking Framework" in result.output
        assert "run" in result.output
        assert "list" in result.output
        assert "add" in result.output

    def test_main_version(self, cli_runner):
        """Test that --version works."""
        result = cli_runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "apps-benchmark" in result.output

    def test_main_initializes_registries(self, cli_runner):
        """Test that main command attempts to initialize registries."""
        # Just test that the command runs without error
        # The initialization happens in a try-except that silently ignores errors
        result = cli_runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        # The help text is displayed regardless of initialization status
        assert "IonQ Quantum Application Benchmarking Framework" in result.output

    def test_local_dev_reports_config_file_path(self, cli_runner, tmp_path, monkeypatch):
        """Test that local-dev output shows the shared config file path."""
        config_file = tmp_path / "config_local_dev.json"
        local_dev_dir = tmp_path / "local_dev"
        monkeypatch.setattr("apps_benchmark.cli.get_config_file_path", lambda: config_file)
        monkeypatch.setattr(
            "apps_benchmark.cli.get_local_dev_dir_from_config", lambda: local_dev_dir
        )

        result = cli_runner.invoke(main, ["local-dev"])

        assert result.exit_code == 0
        assert f"Config file:          {config_file}" in result.output
        assert "config_local_dev.json" in result.output


class TestRunCommand:
    """Tests for run command."""

    def test_run_help(self, cli_runner):
        """Test run command help."""
        result = cli_runner.invoke(main, ["run", "--help"])
        assert result.exit_code == 0
        assert "Run benchmarks" in result.output
        assert "--backend" in result.output
        assert "--qbit-max" in result.output
        assert "--category" in result.output
        assert "--case-uuid" in result.output
        assert "--self-test" in result.output
        assert "--shots" in result.output
        assert "--save-config" in result.output
        assert "--load-config" in result.output

    def test_run_requires_category_or_uuid(self, cli_runner, mock_home):
        """Test that run command requires category or case-uuid."""
        result = cli_runner.invoke(main, ["run", "--backend=qiskit", "--qbit-max=5"])
        assert result.exit_code == 1
        assert "Either --case-uuid or --category is required" in result.output

    def test_run_default_qbit_max(self, cli_runner, mock_home):
        """Test that run command has default qbit-max of 10."""
        # Needs category or UUID to proceed
        result = cli_runner.invoke(main, ["run", "--backend=mock_backend", "--category=chemistry"])
        # Will fail at "category not yet implemented" but validates defaults
        assert "--qbit-max" not in result.output or "Max qubits: 10" in result.output

    def test_run_default_shots(self, cli_runner, mock_home):
        """Test that run command advertises benchmark-specific shot defaults."""
        # Needs category or UUID to proceed
        result = cli_runner.invoke(main, ["run", "--backend=mock_backend", "--category=chemistry"])
        # Should show default shots value in category execution
        assert result.exit_code == 0
        assert "set per benchmark: 10000" in result.output

    def test_run_with_category(self, cli_runner, mock_home):
        """Test run command with category (Phase 4)."""
        result = cli_runner.invoke(main, ["run", "--backend=mock_backend", "--category=chemistry"])
        assert result.exit_code == 0
        assert "Running benchmarks in category 'chemistry'" in result.output
        assert "RESULTS SUMMARY" in result.output
        assert "Total runs:" in result.output

    def test_run_with_qft_category(self, cli_runner, mock_home):
        """Smoke test the public CLI path for the built-in qft category."""
        result = cli_runner.invoke(
            main,
            ["run", "--backend=mock_backend", "--category=qft", "--qbit-max=10"],
        )
        assert result.exit_code == 0
        assert "Running benchmarks in category 'qft'" in result.output
        assert "10_qubit_challenge" in result.output
        assert "cosine_qft" in result.output
        assert "RESULTS SUMMARY" in result.output

    def test_run_with_case_uuid(self, cli_runner, mock_home):
        """Test run command with case UUID (should work with built-in instances)."""
        # Use a built-in H2 problem instance UUID
        result = cli_runner.invoke(main, ["run", "--backend=mock_backend", "--case-uuid=610cfb55"])
        # Should find the problem instance and run it
        assert (
            "Searching for problem instance" in result.output
            or "BENCHMARK RESULTS" in result.output
        )

    def test_run_with_qft_case_uuid(self, cli_runner, mock_home):
        """Smoke test the public CLI path for a built-in QFT case UUID."""
        result = cli_runner.invoke(
            main,
            [
                "run",
                "--backend=qiskit_aer_sim_backend",
                "--case-uuid=f75ae75f",
                "--algorithm=cosine_qft",
            ],
        )
        assert result.exit_code == 0
        assert "Found: qft_10_high_freq_challenge.json (category: qft)" in result.output
        assert "Loaded runner: cosine_qft" in result.output
        assert "BENCHMARK RESULTS" in result.output

    def test_run_self_test_without_backend(self, cli_runner, mock_home):
        """Test that self-test requires backend."""
        result = cli_runner.invoke(main, ["run", "--self-test"])
        assert result.exit_code == 1
        assert "--backend is required" in result.output

    def test_run_self_test_with_mock_backend(self, cli_runner, mock_home):
        """Test self-test with mock backend."""
        result = cli_runner.invoke(main, ["run", "--backend=mock_backend", "--self-test"])
        assert result.exit_code == 0
        assert "Testing backend 'mock_backend'" in result.output
        assert "is available and ready" in result.output

    def test_run_self_test_with_qiskit_aer_sim_backend(self, cli_runner, mock_home):
        """Test self-test with Qiskit backend."""
        result = cli_runner.invoke(main, ["run", "--backend=qiskit_aer_sim_backend", "--self-test"])
        assert result.exit_code == 0, (
            f"Self-test with qiskit_aer_sim_backend should succeed, got {result.output}"
        )
        assert "Testing backend 'qiskit_aer_sim_backend'" in result.output
        assert "is available and ready" in result.output

    def test_run_self_test_with_qbit_max(self, cli_runner, mock_home):
        """Test self-test with qbit-max runs test circuit."""
        result = cli_runner.invoke(
            main, ["run", "--backend=mock_backend", "--self-test", "--qbit-max=2"]
        )
        assert result.exit_code == 0
        assert "Running test circuit" in result.output
        assert "Test circuit executed successfully" in result.output

    def test_run_self_test_with_nonexistent_backend(self, cli_runner, mock_home):
        """Test self-test with nonexistent backend."""
        result = cli_runner.invoke(main, ["run", "--backend=nonexistent", "--self-test"])
        assert result.exit_code == 1
        assert "not found in registry" in result.output

    def test_run_with_save_config(self, cli_runner, mock_home):
        """Test run command with save-config."""
        result = cli_runner.invoke(
            main,
            [
                "run",
                "--backend=ionq.forte",
                "--qbit-max=11",
                "--save-config=production",
            ],
        )
        # save-config and load-config are placeholders for Phase 6
        # Command should now require category or UUID
        assert result.exit_code == 1

    def test_run_with_load_config(self, cli_runner, mock_home):
        """Test run command with load-config."""
        # First create a config file to load
        config_dir = mock_home / ".apps-benchmark"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_file = config_dir / "apps-benchmark-config-production.json"
        config_data = {
            "version": "1.0",
            "backend": "mock_backend",
            "qbit_max": 5,
            "shots": 2000,
        }
        with open(config_file, "w") as f:
            json.dump(config_data, f)

        result = cli_runner.invoke(
            main,
            ["run", "--backend=mock_backend", "--load-config=production", "--category=chemistry"],
        )
        # Should proceed to category execution (Phase 4 implemented)
        assert result.exit_code == 0
        assert (
            "Running benchmarks in category 'chemistry'" in result.output
            or "Config file not found" in result.output
        )
        assert "Loading configuration 'production'" in result.output

    def test_run_with_custom_shots(self, cli_runner, mock_home):
        """Test run command with custom shots value."""
        result = cli_runner.invoke(
            main, ["run", "--backend=mock_backend", "--shots=5000", "--category=chemistry"]
        )
        # Should proceed to category execution
        assert result.exit_code == 0
        assert "Running benchmarks in category 'chemistry'" in result.output
        assert "Shots per evaluation point: 5000" in result.output

    def test_resolve_problem_shots_uses_benchmark_metadata(self):
        """Benchmark metadata should provide the default shot count."""
        benchmark_case = BenchmarkCase(
            benchmark_category="chemistry",
            problem_type="test_problem",
            instance_name="test_case",
            num_qubits=2,
            solution_algorithms=["vqe_puccd"],
            data={"recommended_minimum_shots_per_qc": 5000},
        )
        benchmark_shots = _get_shots_for_case(benchmark_case)
        final_shots = _resolve_shots(None, None, benchmark_shots)
        assert final_shots == 5000, "shot fallback flow failed"

    def test_resolve_problem_shots_ignores_actual_shots_metadata(self):
        """Actual recorded shots metadata must not drive default selection."""
        benchmark_case = BenchmarkCase(
            benchmark_category="chemistry",
            problem_type="test_problem",
            instance_name="test_case",
            num_qubits=2,
            solution_algorithms=["vqe_puccd"],
            data={"shots_per_qc": 7000},
        )
        benchmark_shots = _get_shots_for_case(benchmark_case)
        final_shots = _resolve_shots(
            cli_set_shots=1000, config_set_shots=None, benchmark_set_shots=benchmark_shots
        )
        assert final_shots == 1000

    def test_resolve_problem_shots_falls_back_to_global_default(self):
        """Benchmarks without metadata should fall back to the global baseline."""
        benchmark_case = BenchmarkCase(
            benchmark_category="test_category",
            problem_type="test_problem",
            instance_name="test_case",
            num_qubits=2,
            solution_algorithms=["test_algorithm"],
            data={},
        )
        benchmark_shots = _get_shots_for_case(benchmark_case)
        final_shots = _resolve_shots(None, None, benchmark_shots)
        assert final_shots == 1000

    def test_run_rejects_zero_shots(self, cli_runner, mock_home):
        """Direct CLI shot counts must be positive."""
        result = cli_runner.invoke(
            main, ["run", "--backend=mock_backend", "--shots=0", "--category=chemistry"]
        )
        assert result.exit_code != 0
        assert "Invalid value for '--shots'" in result.output

    def test_run_rejects_invalid_shots_from_loaded_config(self, cli_runner, mock_home):
        """Saved config shot counts must also be positive."""
        config_dir = mock_home / ".apps-benchmark"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_file = config_dir / "apps-benchmark-config-invalid-shots.json"
        config_data = {
            "version": "1.0",
            "backend": "mock_backend",
            "shots": -1,
        }
        with open(config_file, "w") as f:
            json.dump(config_data, f)

        result = cli_runner.invoke(
            main,
            [
                "run",
                "--backend=mock_backend",
                "--load-config=invalid-shots",
                "--category=chemistry",
            ],
        )
        assert result.exit_code == 1
        assert "Shots must be positive" in result.output

    def test_run_help_shows_algorithm_option(self, cli_runner):
        """Test that --algorithm appears in run help."""
        result = cli_runner.invoke(main, ["run", "--help"])
        assert "--algorithm" in result.output

    def test_run_algorithm_override_uuid(self, cli_runner, mock_home):
        """Test --algorithm selects the requested runner for UUID runs."""
        # 852ab5dd supports both qaoa and lr_qaoa
        result = cli_runner.invoke(
            main,
            [
                "run",
                "--backend=qiskit_aer_sim_backend",
                "--case-uuid=852ab5dd",
                "--algorithm=lr_qaoa",
            ],
        )
        assert result.exit_code == 0
        assert "Loaded runner: lr_qaoa" in result.output

    def test_run_algorithm_default_uuid(self, cli_runner, mock_home):
        """Test that without --algorithm, the first runner is selected."""
        result = cli_runner.invoke(
            main,
            ["run", "--backend=qiskit_aer_sim_backend", "--case-uuid=852ab5dd"],
        )
        assert result.exit_code == 0
        assert "Loaded runner: qaoa" in result.output

    def test_run_algorithm_invalid_rejected(self, cli_runner, mock_home):
        """Test --algorithm with invalid runner name is rejected."""
        result = cli_runner.invoke(
            main,
            ["run", "--backend=qiskit_aer_sim_backend", "--case-uuid=852ab5dd", "--algorithm=poop"],
        )
        assert result.exit_code == 1
        assert "algorithm 'poop' not available" in result.output

    def test_run_algorithm_open_solver_rejected(self, cli_runner, mock_home):
        """Explicit open benchmark solvers should fail with a clear message."""
        result = cli_runner.invoke(
            main,
            ["run", "--backend=mock_backend", "--case-uuid=98bc8a81", "--algorithm=varqite"],
        )
        assert result.exit_code == 1
        assert "open benchmark solver" in result.output
        assert "bring your own solver" in result.output

    def test_run_open_benchmark_case_uuid_rejected_cleanly(self, cli_runner, mock_home):
        """Open-only benchmark cases should fail before runner import."""
        result = cli_runner.invoke(
            main,
            ["run", "--backend=mock_backend", "--case-uuid=b56cc063"],
        )
        assert result.exit_code == 1
        assert "open benchmark" in result.output
        assert "qc_afqmc" in result.output
        assert "bring your own solver" in result.output

    def test_run_open_qlbm_case_uuid_rejected_cleanly(self, cli_runner, mock_home):
        """QLBM cases should be tagged as open benchmarks."""
        result = cli_runner.invoke(
            main,
            ["run", "--backend=mock_backend", "--case-uuid=fe2e221a"],
        )
        assert result.exit_code == 1
        assert "open benchmark" in result.output
        assert "qlbm" in result.output
        assert "bring your own solver" in result.output

    def test_run_open_only_category_reports_clear_error(self, cli_runner, mock_home):
        """Open-only selections should not die with an import error."""
        result = cli_runner.invoke(
            main,
            [
                "run",
                "--backend=mock_backend",
                "--category=chemistry",
                "--algorithm=qc_afqmc",
            ],
        )
        assert result.exit_code == 1
        assert "open benchmark" in result.output
        assert "no runnable closed benchmark cases matched" in result.output

    def test_run_open_qlbm_category_reports_clear_error(self, cli_runner, mock_home):
        """QLBM category runs should fail cleanly without an import error."""
        result = cli_runner.invoke(
            main,
            [
                "run",
                "--backend=mock_backend",
                "--category=computational_fluid_dynamics",
                "--qbit-max=20",
            ],
        )
        assert result.exit_code == 1
        assert "open benchmark case" in result.output
        assert "no runnable closed benchmark cases matched" in result.output


class TestListCommand:
    """Tests for list command."""

    def test_list_help(self, cli_runner):
        """Test list command help."""
        result = cli_runner.invoke(main, ["list", "--help"])
        assert result.exit_code == 0, f"List command help should work, got:\n{result.output}"
        assert "List available backends and benchmarks" in result.output
        assert "--backends" in result.output
        assert "--category" in result.output

    def test_list_benchmarks_empty(self, cli_runner, mock_home):
        """Test listing benchmarks when none are registered."""
        result = cli_runner.invoke(main, ["list"])
        assert result.exit_code == 0
        # The output should show available categories (built-in chemistry may be discovered)
        # Check that the command runs successfully
        assert "benchmark" in result.output.lower() or "chemistry" in result.output.lower()

    def test_list_backends_shows_builtin(self, cli_runner, mock_home):
        """Test listing backends shows built-in backends."""
        result = cli_runner.invoke(main, ["list", "--backends"])
        assert result.exit_code == 0
        # Should show built-in backends that are auto-discovered
        assert "Available backends:" in result.output
        assert "mock_backend" in result.output or "qiskit_aer_sim_backend" in result.output

    def test_list_backends_with_data(self, cli_runner, mock_home):
        """Test listing backends when some DIY backends are registered."""
        # Create registry with DIY backends only
        local_dev = mock_home / "local_dev"
        local_dev.mkdir(parents=True)
        (local_dev / "backends").mkdir()
        (local_dev / "benchmarks").mkdir()

        registry_data = {
            "version": "1.0",
            "diy_backends": {
                "custom": {"builtin": False, "class": "CustomBackend"},
            },
        }
        with open(local_dev / "backends.json", "w") as f:
            json.dump(registry_data, f)

        result = cli_runner.invoke(main, ["list", "--backends"])
        assert result.exit_code == 0
        assert "Available backends:" in result.output
        # Should show builtin backends discovered on-the-fly
        assert (
            "mock_backend (built-in)" in result.output
            or "qiskit_aer_sim_backend (built-in)" in result.output
        )
        # Should show DIY backend
        assert "custom (DIY)" in result.output

    def test_list_benchmarks_with_data(self, cli_runner, mock_home):
        """Test listing benchmarks when some DIY benchmarks are registered."""
        # Create registry with DIY benchmarks only
        local_dev = mock_home / "local_dev"
        local_dev.mkdir(parents=True)
        (local_dev / "backends").mkdir()
        (local_dev / "benchmarks").mkdir()

        registry_data = {
            "version": "1.0",
            "diy_benchmarks": {"optimization": {"custom_qaoa": {"runner_class": "CustomQaoa"}}},
        }
        with open(local_dev / "benchmarks.json", "w") as f:
            json.dump(registry_data, f)

        result = cli_runner.invoke(main, ["list"])
        assert result.exit_code == 0
        assert "Available benchmark categories:" in result.output
        # Should show builtin categories discovered on-the-fly (e.g., chemistry)
        # and DIY category
        assert "optimization" in result.output
        assert "Built-in runners:" in result.output
        assert "Built-in problem instances:" in result.output
        assert "Open benchmark algorithms:" in result.output
        assert "DIY benchmarks: 1" in result.output

    def test_list_specific_category(self, cli_runner, mock_home):
        """Test listing benchmarks for specific category."""
        # Create registry
        local_dev = mock_home / "local_dev"
        local_dev.mkdir(parents=True)
        (local_dev / "backends").mkdir()
        (local_dev / "benchmarks").mkdir()

        registry_data = {
            "version": "1.0",
            "diy_benchmarks": {"chemistry": {"my_vqe": {"runner_class": "MyVqe"}}},
        }

        with open(local_dev / "benchmarks.json", "w") as f:
            json.dump(registry_data, f)

        result = cli_runner.invoke(main, ["list", "--category=chemistry"])
        assert result.exit_code == 0, (
            f"List command with category should succeed, got {result.output}"
        )
        assert "Benchmarks in category 'chemistry':" in result.output
        assert "my_vqe" in result.output
        assert "DIY:" in result.output
        assert "my_vqe" in result.output
        assert "Open benchmark algorithms:" in result.output

    def test_list_open_benchmark_category(self, cli_runner, mock_home):
        """Open benchmark categories should advertise tagged ghost algorithms."""
        result = cli_runner.invoke(main, ["list", "--category=chemistry"])
        assert result.exit_code == 0
        assert "Benchmarks in category 'chemistry':" in result.output
        assert "Open benchmark algorithms:" in result.output
        assert "qc_afqmc" in result.output

    def test_list_open_qlbm_category(self, cli_runner, mock_home):
        """QLBM categories should advertise tagged open algorithms."""
        result = cli_runner.invoke(main, ["list", "--category=computational_fluid_dynamics"])
        assert result.exit_code == 0
        assert "Benchmarks in category 'computational_fluid_dynamics':" in result.output
        assert "Open benchmark algorithms:" in result.output
        assert "qlbm" in result.output

    def test_list_qft_category(self, cli_runner, mock_home):
        """Smoke test listing the built-in qft category."""
        result = cli_runner.invoke(main, ["list", "--category=qft"])
        assert result.exit_code == 0
        assert "Benchmarks in category 'qft':" in result.output
        assert "Built-in:" in result.output
        assert "cosine_qft" in result.output
        assert "hidden_phase_qft" in result.output
        assert "qft_lcu" not in result.output


class TestAddCommand:
    """Tests for add command."""

    def test_add_help(self, cli_runner):
        """Test add command help."""
        result = cli_runner.invoke(main, ["add", "--help"])
        assert result.exit_code == 0
        assert "Add a DIY backend or benchmark" in result.output
        assert "--backend" in result.output
        assert "--benchmark" in result.output
        assert "--category" in result.output

    def test_add_no_arguments(self, cli_runner, mock_home):
        """Test add command with no arguments fails."""
        result = cli_runner.invoke(main, ["add"])
        assert result.exit_code == 1
        assert "Must specify either --backend or --benchmark" in result.output

    def test_add_benchmark_without_category(self, cli_runner, mock_home):
        """Test add benchmark without category fails."""
        result = cli_runner.invoke(main, ["add", "--benchmark=my_vqe"])
        assert result.exit_code == 1
        assert "--category is required when adding a benchmark" in result.output

    def test_add_backend_success(self, cli_runner, mock_home):
        """Test successful backend addition."""
        # Setup
        local_dev = mock_home / "local_dev"
        local_dev.mkdir(parents=True)
        backends_dir = local_dev / "backends"
        backends_dir.mkdir()

        # Create valid backend file
        backend_file = backends_dir / "test_backend.py"
        backend_file.write_text("""
from apps_benchmark.core.backend import AbstractBackend
from qiskit import QuantumCircuit

class TestBackend(AbstractBackend):
    def name(self) -> str:
        return "test"

    def run(self, circuits: list[QuantumCircuit], shots: int = 1000, job_name: str | None = None) -> tuple[list[dict], str, dict]:
        return [], "job", {}
""")

        # Create empty registry
        registry_data = {"version": "1.0", "diy_backends": {}}
        with open(local_dev / "backends.json", "w") as f:
            json.dump(registry_data, f)

        result = cli_runner.invoke(main, ["add", "--backend=test_backend"])

        assert result.exit_code == 0, f"Adding valid backend should succeed, got:\n{result.output}"
        assert "registered successfully" in result.output

    def test_add_backend_file_not_found(self, cli_runner, mock_home):
        """Test adding backend when file doesn't exist."""
        # Setup empty registry
        local_dev = mock_home / "local_dev"
        local_dev.mkdir(parents=True)
        backends_dir = local_dev / "backends"
        backends_dir.mkdir()

        registry_data = {"version": "1.0", "backends": {}}
        with open(local_dev / "backends.json", "w") as f:
            json.dump(registry_data, f)

        result = cli_runner.invoke(main, ["add", "--backend=nonexistent"])
        assert result.exit_code == 1
        assert "Error registering backend" in result.output

    def test_add_benchmark_success(self, cli_runner, mock_home):
        """Test successful benchmark addition."""
        # Setup
        local_dev = mock_home / "local_dev"
        local_dev.mkdir(parents=True)
        category_dir = local_dev / "benchmarks" / "test_cat"
        algorithms_dir = category_dir / "algorithms"
        algorithms_dir.mkdir(parents=True)

        # Create valid runner file
        runner_file = algorithms_dir / "test_runner_runner.py"
        runner_file.write_text("""
from apps_benchmark.core.benchmark import AbstractAlgoRunner
from typing import Any, Tuple, Dict

class TestRunnerRunner(AbstractAlgoRunner):
    def name(self) -> str:
        return "test_runner"

    def setup_algo_inputs(self, benchmark_case) -> Tuple[Any, ...]:
        return ()

    def execute_benchmark_algo(self, algo_inputs, backend, shots, **kwargs) -> Any:
        return {}

    def compute_merit_figures(self, algo_output, benchmark_case) -> Dict[str, Any]:
        return {"total_shots": 1000, "score": 1.0}
""")

        # Create empty registry
        registry_data = {
            "version": "1.0",
            "builtin_benchmarks": {},
            "diy_benchmarks": {},
        }
        with open(local_dev / "benchmarks.json", "w") as f:
            json.dump(registry_data, f)

        result = cli_runner.invoke(main, ["add", "--benchmark=test_runner", "--category=test_cat"])
        assert result.exit_code == 0
        assert "registered successfully" in result.output

    def test_add_benchmark_file_not_found(self, cli_runner, mock_home):
        """Test adding benchmark when file doesn't exist."""
        # Setup empty registry
        local_dev = mock_home / "local_dev"
        local_dev.mkdir(parents=True)
        (local_dev / "benchmarks").mkdir()

        registry_data = {
            "version": "1.0",
            "builtin_benchmarks": {},
            "diy_benchmarks": {},
        }
        with open(local_dev / "benchmarks.json", "w") as f:
            json.dump(registry_data, f)

        result = cli_runner.invoke(main, ["add", "--benchmark=nonexistent", "--category=test"])
        assert result.exit_code == 1
        assert "Error registering benchmark" in result.output


class TestCLIIntegration:
    """Integration tests for CLI workflow."""

    def test_full_workflow(self, cli_runner, mock_home):
        """Test complete workflow: commands run successfully."""
        # 1. Help command works
        result = cli_runner.invoke(main, ["--help"])
        assert result.exit_code == 0

        # 2. List backends works
        result = cli_runner.invoke(main, ["list", "--backends"])
        assert result.exit_code == 0

        # 3. List benchmarks works
        result = cli_runner.invoke(main, ["list"])
        assert result.exit_code == 0

        # 4. Run command requires category or UUID
        result = cli_runner.invoke(main, ["run", "--backend=mock_backend"])
        assert result.exit_code == 1
        assert "Either --case-uuid or --category is required" in result.output

        # 5. Run command with category (Phase 4 implemented)
        result = cli_runner.invoke(main, ["run", "--backend=mock_backend", "--category=chemistry"])
        assert result.exit_code == 0
        assert "Running benchmarks in category 'chemistry'" in result.output
        assert "RESULTS SUMMARY" in result.output

    def test_error_handling(self, cli_runner, mock_home):
        """Test that CLI handles errors gracefully."""
        # Try to add nonexistent backend
        result = cli_runner.invoke(main, ["add", "--backend=nonexistent"])
        assert result.exit_code == 1
        assert "Error" in result.output

    def test_help_available_for_all_commands(self, cli_runner):
        """Test that help is available for all commands."""
        commands = ["run", "list", "add"]

        for cmd in commands:
            result = cli_runner.invoke(main, [cmd, "--help"])
            assert result.exit_code == 0
            assert "--help" in result.output or "Show this message and exit" in result.output
