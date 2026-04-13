"""
Tests for registry system.

This module tests registry initialization, backend/benchmark registration,
and discovery mechanisms.

Copyright (c) 2025 IonQ, Inc. All rights reserved.
Licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0
International (CC BY-NC-ND 4.0) License.
For details, visit: https://creativecommons.org/licenses/by-nc-nd/4.0/
"""

import json
from pathlib import Path

import pytest
from apps_benchmark.core.registry import (
    _discover_builtin_backends,
    _discover_builtin_benchmarks,
    ensure_local_dev_dir,
    get_local_dev_dir,
    initialize_empty_registry,
    initialize_registries,
    list_builtin_backends,
    list_builtin_benchmarks,
    list_diy_backends,
    list_diy_benchmarks,
    register_diy_backend,
    register_diy_benchmark,
)
from apps_benchmark.errors import (
    BackendValidationError,
    BenchmarkValidationError,
    RegistryError,
)
from apps_benchmark.utils.validation import (
    check_duplicate_backend,
    check_duplicate_benchmark,
    check_duplicate_benchmark_case_ids,
)


class TestGetLocalDevDir:
    """Tests for local_dev directory management."""

    def test_get_local_dev_dir_returns_path(self):
        """Test that get_local_dev_dir returns a Path object."""
        result = get_local_dev_dir()
        assert isinstance(result, Path)
        assert result.name == "local_dev"
        assert result.parent == Path.home()

    def test_ensure_local_dev_dir_creates_structure(self, tmp_path, monkeypatch):
        """Test that ensure_local_dev_dir creates directory structure."""
        # Mock home directory
        monkeypatch.setattr("apps_benchmark.core.registry.Path.home", lambda: tmp_path)

        result = ensure_local_dev_dir()

        assert result.exists()
        assert result.is_dir()
        assert (result / "backends").exists()
        assert (result / "backends").is_dir()
        assert (result / "benchmarks").exists()
        assert (result / "benchmarks").is_dir()

    def test_ensure_local_dev_dir_idempotent(self, tmp_path, monkeypatch):
        """Test that ensure_local_dev_dir can be called multiple times."""
        monkeypatch.setattr("apps_benchmark.core.registry.Path.home", lambda: tmp_path)

        # Call twice
        result1 = ensure_local_dev_dir()
        result2 = ensure_local_dev_dir()

        assert result1 == result2
        assert result1.exists()


class TestInitializeEmptyRegistry:
    """Tests for empty registry initialization."""

    def test_initialize_backend_registry(self, tmp_path):
        """Test initializing empty backend registry."""
        registry_path = tmp_path / "backends.json"

        initialize_empty_registry(registry_path)

        assert registry_path.exists()
        with open(registry_path) as f:
            data = json.load(f)

        assert data["version"] == "1.0"
        assert data["diy_backends"] == {}

    def test_initialize_benchmark_registry(self, tmp_path):
        """Test initializing empty benchmark registry."""
        registry_path = tmp_path / "benchmarks.json"

        initialize_empty_registry(registry_path)

        assert registry_path.exists()
        with open(registry_path) as f:
            data = json.load(f)

        assert data["version"] == "1.0"
        assert data["diy_benchmarks"] == {}
        # builtin_benchmarks no longer stored in registry


class TestDiscoverBuiltinBackends:
    """Tests for backend discovery."""

    def test_discover_builtin_backends_returns_dict(self):
        """Test that discover_builtin_backends returns a dict."""
        result = _discover_builtin_backends()
        assert isinstance(result, dict)

    def test_discover_builtin_backends_structure(self):
        """Test that discovered backends have correct structure."""
        backends = _discover_builtin_backends()

        for backend_name, info in backends.items():
            assert isinstance(backend_name, str)
            assert isinstance(info, dict)
            assert "module" in info
            assert "class" in info
            assert "builtin" in info
            assert info["builtin"] is True
            assert "location" in info
            assert "registered_at" in info


class TestDiscoverBuiltinBenchmarks:
    """Tests for benchmark discovery."""

    def test_discover_builtin_benchmarks_returns_dict(self):
        """Test that discover_builtin_benchmarks returns a dict."""
        result = _discover_builtin_benchmarks()
        assert isinstance(result, dict)

    def test_discover_builtin_benchmarks_structure(self):
        """Test that discovered benchmarks have correct structure."""
        benchmarks = _discover_builtin_benchmarks()

        for category_name, info in benchmarks.items():
            assert isinstance(category_name, str)
            assert isinstance(info, dict)
            assert "location" in info
            assert "runners" in info
            assert "benchmark_cases" in info
            assert isinstance(info["runners"], list)
            assert isinstance(info["benchmark_cases"], list)

    def test_discover_builtin_benchmarks_includes_open_metadata(self):
        """Open benchmark case metadata should be exposed in registry discovery."""
        benchmarks = _discover_builtin_benchmarks()

        chemistry_cases = benchmarks["chemistry"]["benchmark_cases"]
        qcafqmc_case = next(
            entry for entry in chemistry_cases if entry["problem_type"] == "qc_afqmc"
        )
        assert qcafqmc_case["open_solution_algorithms"] == ["qc_afqmc"]
        assert qcafqmc_case["all_solutions_open"] is True

    def test_discover_builtin_benchmarks_includes_qlbm_open_metadata(self):
        """QLBM discovery should surface open benchmark metadata."""
        benchmarks = _discover_builtin_benchmarks()

        cfd_case = benchmarks["computational_fluid_dynamics"]["benchmark_cases"][0]
        assert cfd_case["open_solution_algorithms"] == ["qlbm"]
        assert cfd_case["all_solutions_open"] is True

    def test_discover_builtin_benchmarks_finds_nested_open_cases(self):
        """Nested benchmark_cases directories should be discovered for built-ins."""
        benchmarks = _discover_builtin_benchmarks()

        chemistry_cases = benchmarks["chemistry"]["benchmark_cases"]
        qcafqmc_cases = [entry for entry in chemistry_cases if entry["problem_type"] == "qc_afqmc"]

        assert len(qcafqmc_cases) == 2
        assert all(entry["open_solution_algorithms"] == ["qc_afqmc"] for entry in qcafqmc_cases)
        assert all(entry["all_solutions_open"] is True for entry in qcafqmc_cases)

    def test_discover_builtin_benchmarks_fails_on_missing_instance_id(self, tmp_path, monkeypatch):
        """Test corrupted builtin problem instances fail discovery."""
        package_root = tmp_path / "mock_benchmarks"
        package_root.mkdir()
        (package_root / "__init__.py").write_text("")
        category_dir = package_root / "chemistry"
        (category_dir / "algorithms").mkdir(parents=True)
        instances_dir = category_dir / "benchmark_cases"
        instances_dir.mkdir()
        (instances_dir / "broken.json").write_text(
            json.dumps(
                {
                    "benchmark_category": "chemistry",
                    "problem_type": "vqe",
                    "instance_name": "broken",
                    "num_qubits": 2,
                    "solution_algorithms": ["vqe"],
                    "data": {},
                }
            )
        )

        import apps_benchmark.benchmarks

        monkeypatch.setattr(
            apps_benchmark.benchmarks,
            "__file__",
            str(package_root / "__init__.py"),
        )

        with pytest.raises(RegistryError, match="missing required 'instance_id'"):
            _discover_builtin_benchmarks()


class TestInitializeRegistries:
    """Tests for full registry initialization."""

    def test_initialize_registries_creates_structure(self, tmp_path, monkeypatch):
        """Test that initialize_registries creates full structure."""
        monkeypatch.setattr("apps_benchmark.core.registry.Path.home", lambda: tmp_path)

        initialize_registries()

        local_dev = tmp_path / "local_dev"
        assert local_dev.exists()
        assert (local_dev / "backends").exists()
        assert (local_dev / "benchmarks").exists()
        assert (local_dev / "backends.json").exists()
        assert (local_dev / "benchmarks.json").exists()

    def test_initialize_registries_idempotent(self, tmp_path, monkeypatch):
        """Test that initialize_registries can be called multiple times."""
        monkeypatch.setattr("apps_benchmark.core.registry.Path.home", lambda: tmp_path)

        # Call twice
        initialize_registries()
        initialize_registries()

        local_dev = tmp_path / "local_dev"
        assert local_dev.exists()


class TestRegisterDIYBackend:
    """Tests for DIY backend registration."""

    def test_register_backend_success(self, tmp_path, monkeypatch):
        """Test successful backend registration."""
        monkeypatch.setattr("apps_benchmark.core.registry.Path.home", lambda: tmp_path)

        # Setup
        local_dev = ensure_local_dev_dir()
        initialize_empty_registry(local_dev / "backends.json")

        # Create a valid backend file
        backend_file = local_dev / "backends" / "my_test_backend.py"
        backend_file.write_text("""
from apps_benchmark.core.backend import AbstractBackend
from qiskit import QuantumCircuit

class MyTestBackend(AbstractBackend):
    def name(self) -> str:
        return "my_test_backend"

    def run(self, circuits: list[QuantumCircuit], shots: int = 1000, job_name: str | None = None) -> tuple[list[dict], str, dict]:
        results = [{"00": shots} for _ in circuits]
        return results, "job_123", {}
""")

        # Register
        register_diy_backend("my_test_backend")

        # Verify registration
        with open(local_dev / "backends.json") as f:
            registry = json.load(f)

        assert "my_test_backend" in registry["diy_backends"]
        backend_info = registry["diy_backends"]["my_test_backend"]
        assert backend_info["class"] == "MyTestBackend"
        assert backend_info["builtin"] is False
        assert backend_info["validated"] is True

    def test_register_backend_file_not_found(self, tmp_path, monkeypatch):
        """Test registration fails when file doesn't exist."""
        monkeypatch.setattr("apps_benchmark.core.registry.Path.home", lambda: tmp_path)

        ensure_local_dev_dir()
        local_dev = tmp_path / "local_dev"
        initialize_empty_registry(local_dev / "backends.json")

        with pytest.raises(FileNotFoundError, match="Backend file not found"):
            register_diy_backend("nonexistent_backend")

    def test_register_backend_wrong_class_name(self, tmp_path, monkeypatch):
        """Test registration fails when class name doesn't match convention."""
        monkeypatch.setattr("apps_benchmark.core.registry.Path.home", lambda: tmp_path)

        local_dev = ensure_local_dev_dir()
        initialize_empty_registry(local_dev / "backends.json")

        # Create backend with wrong class name
        backend_file = local_dev / "backends" / "my_backend.py"
        backend_file.write_text("""
from apps_benchmark.core.backend import AbstractBackend

class WrongClassName(AbstractBackend):
    def name(self) -> str:
        return "my_backend"

    def run(self, circuits, shots=1000, job_name=None):
        return [], "job", {}
""")

        with pytest.raises(BackendValidationError, match="expected class 'MyBackend'"):
            register_diy_backend("my_backend")

    def test_register_backend_missing_methods(self, tmp_path, monkeypatch):
        """Test registration fails when backend has missing methods."""
        monkeypatch.setattr("apps_benchmark.core.registry.Path.home", lambda: tmp_path)

        local_dev = ensure_local_dev_dir()
        initialize_empty_registry(local_dev / "backends.json")

        # Create incomplete backend
        backend_file = local_dev / "backends" / "incomplete_backend.py"
        backend_file.write_text("""
from apps_benchmark.core.backend import AbstractBackend

class IncompleteBackend(AbstractBackend):
    def name(self) -> str:
        return "incomplete"
    # Missing run() method
""")

        with pytest.raises(BackendValidationError, match="unimplemented abstract methods"):
            register_diy_backend("incomplete_backend")

    def test_register_backend_duplicate(self, tmp_path, monkeypatch):
        """Test registration fails for duplicate backend name."""
        monkeypatch.setattr("apps_benchmark.core.registry.Path.home", lambda: tmp_path)

        local_dev = ensure_local_dev_dir()
        initialize_empty_registry(local_dev / "backends.json")

        # Create backend
        backend_file = local_dev / "backends" / "test_backend.py"
        backend_file.write_text("""
from apps_benchmark.core.backend import AbstractBackend
from qiskit import QuantumCircuit

class TestBackend(AbstractBackend):
    def name(self) -> str:
        return "test"

    def run(self, circuits: list[QuantumCircuit], shots: int = 1000, job_name: str | None = None) -> tuple[list[dict], str, dict]:
        return [], "job", {}
""")

        # Register once
        register_diy_backend("test_backend")

        # Try to register again
        with pytest.raises(BackendValidationError, match="already registered"):
            register_diy_backend("test_backend")


class TestRegisterDIYBenchmark:
    """Tests for DIY benchmark registration."""

    def test_register_benchmark_success(self, tmp_path, monkeypatch):
        """Test successful benchmark registration."""
        monkeypatch.setattr("apps_benchmark.core.registry.Path.home", lambda: tmp_path)

        local_dev = ensure_local_dev_dir()
        initialize_empty_registry(local_dev / "benchmarks.json")

        # Create category and algorithms directory
        category_dir = local_dev / "benchmarks" / "test_category"
        algorithms_dir = category_dir / "algorithms"
        algorithms_dir.mkdir(parents=True)

        # Create runner file
        runner_file = algorithms_dir / "my_test_runner.py"
        runner_file.write_text("""
from apps_benchmark.core.benchmark import AbstractAlgoRunner
from typing import Any, Dict, Tuple

class MyTestRunner(AbstractAlgoRunner):
    def name(self) -> str:
        return "my_test"

    def setup_algo_inputs(self, benchmark_case) -> Tuple[Any, ...]:
        return ()

    def execute_benchmark_algo(self, algo_inputs, backend, shots, **kwargs) -> Any:
        return {"result": 1}

    def compute_merit_figures(self, algo_output, benchmark_case) -> Dict[str, Any]:
        return {"total_shots": 1000, "score": 1.0}
""")

        # Register
        register_diy_benchmark("my_test", "test_category")

        # Verify registration
        with open(local_dev / "benchmarks.json") as f:
            registry = json.load(f)

        assert "test_category" in registry["diy_benchmarks"]
        assert "my_test" in registry["diy_benchmarks"]["test_category"]
        benchmark_info = registry["diy_benchmarks"]["test_category"]["my_test"]
        assert benchmark_info["runner_class"] == "MyTestRunner"

    def test_register_benchmark_with_benchmark_cases(self, tmp_path, monkeypatch):
        """Test benchmark registration includes problem instances."""
        monkeypatch.setattr("apps_benchmark.core.registry.Path.home", lambda: tmp_path)

        local_dev = ensure_local_dev_dir()
        initialize_empty_registry(local_dev / "benchmarks.json")

        # Create structure
        category_dir = local_dev / "benchmarks" / "chemistry"
        algorithms_dir = category_dir / "algorithms"
        instances_dir = category_dir / "benchmark_cases"
        algorithms_dir.mkdir(parents=True)
        instances_dir.mkdir(parents=True)

        # Create runner
        runner_file = algorithms_dir / "vqe_test_runner.py"
        runner_file.write_text("""
from apps_benchmark.core.benchmark import AbstractAlgoRunner
from typing import Any, Dict, Tuple

class VqeTestRunner(AbstractAlgoRunner):
    def name(self) -> str:
        return "vqe_test"

    def setup_algo_inputs(self, benchmark_case) -> Tuple[Any, ...]:
        return ()

    def execute_benchmark_algo(self, algo_inputs, backend, shots, **kwargs) -> Any:
        return {}

    def compute_merit_figures(self, algo_output, benchmark_case) -> Dict[str, Any]:
        return {"total_shots": 1000, "score": 1.0}
""")

        # Create benchmark case
        problem_file = instances_dir / "h2.json"
        benchmark_case_data = {
            "benchmark_category": "chemistry",
            "problem_type": "vqe",
            "instance_name": "h2",
            "instance_id": "h2_001",
            "num_qubits": 2,
            "solution_algorithms": ["vqe_test"],
            "data": {},
        }
        with open(problem_file, "w") as f:
            json.dump(benchmark_case_data, f)

        # Register
        register_diy_benchmark("vqe_test", "chemistry")

        # Verify benchmark cases were discovered
        with open(local_dev / "benchmarks.json") as f:
            registry = json.load(f)

        benchmark_info = registry["diy_benchmarks"]["chemistry"]["vqe_test"]
        assert len(benchmark_info["benchmark_cases"]) == 1
        assert benchmark_info["benchmark_cases"][0]["uuid"] == "h2_001"
        assert benchmark_info["benchmark_cases"][0]["name"] == "h2"

    def test_register_benchmark_fails_on_duplicate_case_id(self, tmp_path, monkeypatch):
        """Test benchmark registration rejects duplicate benchmark case IDs."""
        monkeypatch.setattr("apps_benchmark.core.registry.Path.home", lambda: tmp_path)

        local_dev = ensure_local_dev_dir()
        registry_path = local_dev / "benchmarks.json"
        registry_data = {
            "version": "1.0",
            "builtin_benchmarks": {
                "chemistry": {
                    "runners": ["existing"],
                    "benchmark_cases": [{"uuid": "h2_001"}],
                }
            },
            "diy_benchmarks": {},
        }
        with open(registry_path, "w") as f:
            json.dump(registry_data, f)

        category_dir = local_dev / "benchmarks" / "chemistry"
        algorithms_dir = category_dir / "algorithms"
        instances_dir = category_dir / "benchmark_cases"
        algorithms_dir.mkdir(parents=True)
        instances_dir.mkdir(parents=True)

        runner_file = algorithms_dir / "vqe_test_runner.py"
        runner_file.write_text("""
from apps_benchmark.core.benchmark import AbstractAlgoRunner
from typing import Any, Dict, Tuple

class VqeTestRunner(AbstractAlgoRunner):
    def name(self) -> str:
        return "vqe_test"

    def setup_algo_inputs(self, benchmark_case) -> Tuple[Any, ...]:
        return ()

    def execute_benchmark_algo(self, algo_inputs, backend, shots, **kwargs) -> Any:
        return {}

    def compute_merit_figures(self, algo_output, benchmark_case) -> Dict[str, Any]:
        return {"total_shots": 1000, "score": 1.0}
""")

        problem_file = instances_dir / "h2.json"
        with open(problem_file, "w") as f:
            json.dump(
                {
                    "benchmark_category": "chemistry",
                    "problem_type": "vqe",
                    "instance_name": "h2",
                    "instance_id": "h2_001",
                    "num_qubits": 2,
                    "solution_algorithms": ["vqe_test"],
                    "data": {},
                },
                f,
            )

        with pytest.raises(BenchmarkValidationError, match="already registered"):
            register_diy_benchmark("vqe_test", "chemistry")

    def test_register_benchmark_file_not_found(self, tmp_path, monkeypatch):
        """Test registration fails when runner file doesn't exist."""
        monkeypatch.setattr("apps_benchmark.core.registry.Path.home", lambda: tmp_path)

        ensure_local_dev_dir()
        local_dev = tmp_path / "local_dev"
        initialize_empty_registry(local_dev / "benchmarks.json")

        with pytest.raises(FileNotFoundError, match="Runner file not found"):
            register_diy_benchmark("nonexistent", "chemistry")

    def test_register_benchmark_wrong_class_name(self, tmp_path, monkeypatch):
        """Test registration fails when class name doesn't match convention."""
        monkeypatch.setattr("apps_benchmark.core.registry.Path.home", lambda: tmp_path)

        local_dev = ensure_local_dev_dir()
        initialize_empty_registry(local_dev / "benchmarks.json")

        # Create structure
        category_dir = local_dev / "benchmarks" / "test_cat"
        algorithms_dir = category_dir / "algorithms"
        algorithms_dir.mkdir(parents=True)

        # Create runner with wrong name
        runner_file = algorithms_dir / "my_runner_runner.py"
        runner_file.write_text("""
from apps_benchmark.core.benchmark import AbstractAlgoRunner

class WrongName(AbstractAlgoRunner):
    def name(self) -> str:
        return "my_runner"

    def setup_algo_inputs(self, benchmark_case):
        return ()

    def execute_benchmark_algo(self, algo_inputs, backend, shots, **kwargs):
        return {}

    def compute_merit_figures(self, algo_output, benchmark_case):
        return {"total_shots": 1000, "score": 1.0}
""")

        with pytest.raises(BenchmarkValidationError, match="expected class 'MyRunnerRunner'"):
            register_diy_benchmark("my_runner", "test_cat")

    def test_register_benchmark_fails_on_missing_instance_id(self, tmp_path, monkeypatch):
        """Test DIY benchmark registration fails for corrupted problem instances."""
        monkeypatch.setattr("apps_benchmark.core.registry.Path.home", lambda: tmp_path)

        local_dev = ensure_local_dev_dir()
        initialize_empty_registry(local_dev / "benchmarks.json")

        category_dir = local_dev / "benchmarks" / "chemistry"
        algorithms_dir = category_dir / "algorithms"
        instances_dir = category_dir / "benchmark_cases"
        algorithms_dir.mkdir(parents=True)
        instances_dir.mkdir(parents=True)

        runner_file = algorithms_dir / "vqe_test_runner.py"
        runner_file.write_text("""
from apps_benchmark.core.benchmark import AbstractAlgoRunner
from typing import Any, Tuple

class VqeTestRunner(AbstractAlgoRunner):
    def name(self) -> str:
        return "vqe_test"

    def setup_algo_inputs(self, problem_instance) -> tuple[Any, ...]:
        return ()

    def execute_benchmark_algo(self, algo_inputs, backend, shots, **kwargs) -> Any:
        return {}

    def compute_merit_figures(self, algo_output, problem_instance) -> dict[str, Any]:
        return {"total_shots": 1000, "score": 1.0}
""")

        problem_file = instances_dir / "broken.json"
        problem_file.write_text(
            json.dumps(
                {
                    "benchmark_category": "chemistry",
                    "problem_type": "vqe",
                    "instance_name": "broken",
                    "num_qubits": 2,
                    "solution_algorithms": ["vqe_test"],
                    "data": {},
                }
            )
        )

        with pytest.raises(BenchmarkValidationError, match="missing required 'instance_id'"):
            register_diy_benchmark("vqe_test", "chemistry")


class TestListBackends:
    """Tests for listing backends."""

    def test_list_backends_empty(self, tmp_path, monkeypatch):
        """Test listing backends when no DIY backends are registered."""
        monkeypatch.setattr("apps_benchmark.core.registry.Path.home", lambda: tmp_path)

        local_dev = ensure_local_dev_dir()
        initialize_empty_registry(local_dev / "backends.json")

        builtin_backends = list_builtin_backends()
        diy_backends = list_diy_backends()

        # Builtin backends should be discovered
        assert isinstance(builtin_backends, dict)
        assert len(builtin_backends) >= 0
        # DIY backends should be empty
        assert diy_backends == {}

    def test_list_backends_no_registry(self, tmp_path, monkeypatch):
        """Test listing backends when registry doesn't exist."""
        monkeypatch.setattr("apps_benchmark.core.registry.Path.home", lambda: tmp_path)

        builtin_backends = list_builtin_backends()
        diy_backends = list_diy_backends()

        # Builtin backends should still be discovered
        assert isinstance(builtin_backends, dict)
        assert len(builtin_backends) >= 0
        # DIY backends should be empty (no registry file)
        assert diy_backends == {}

    def test_list_backends_with_backends(self, tmp_path, monkeypatch):
        """Test listing backends when some DIY backends are registered."""
        monkeypatch.setattr("apps_benchmark.core.registry.Path.home", lambda: tmp_path)

        local_dev = ensure_local_dev_dir()
        registry_path = local_dev / "backends.json"

        # Create registry with DIY backends only
        registry_data = {
            "version": "1.0",
            "diy_backends": {
                "backend1": {"builtin": False, "class": "Backend1"},
                "backend2": {"builtin": False, "class": "Backend2"},
            },
        }
        with open(registry_path, "w") as f:
            json.dump(registry_data, f)

        builtin_backends = list_builtin_backends()
        diy_backends = list_diy_backends()

        # Check builtin backends are discovered
        assert isinstance(builtin_backends, dict)

        # Check DIY backends
        assert len(diy_backends) == 2
        assert "backend1" in diy_backends
        assert "backend2" in diy_backends


class TestListBenchmarks:
    """Tests for listing benchmarks."""

    def test_list_benchmarks_empty(self, tmp_path, monkeypatch):
        """Test listing benchmarks when no DIY benchmarks are registered."""
        monkeypatch.setattr("apps_benchmark.core.registry.Path.home", lambda: tmp_path)

        local_dev = ensure_local_dev_dir()
        initialize_empty_registry(local_dev / "benchmarks.json")

        builtin_benchmarks = list_builtin_benchmarks()
        diy_benchmarks = list_diy_benchmarks()

        # Builtin benchmarks should be discovered
        assert isinstance(builtin_benchmarks, dict)
        # DIY benchmarks should be empty
        assert diy_benchmarks == {}

    def test_list_benchmarks_no_registry(self, tmp_path, monkeypatch):
        """Test listing benchmarks when registry doesn't exist."""
        monkeypatch.setattr("apps_benchmark.core.registry.Path.home", lambda: tmp_path)

        builtin_benchmarks = list_builtin_benchmarks()
        diy_benchmarks = list_diy_benchmarks()

        # Builtin benchmarks should still be discovered
        assert isinstance(builtin_benchmarks, dict)
        # DIY benchmarks should be empty (no registry file)
        assert diy_benchmarks == {}

    def test_list_benchmarks_with_benchmarks(self, tmp_path, monkeypatch):
        """Test listing benchmarks when some DIY benchmarks are registered."""
        monkeypatch.setattr("apps_benchmark.core.registry.Path.home", lambda: tmp_path)

        local_dev = ensure_local_dev_dir()
        registry_path = local_dev / "benchmarks.json"

        # Create registry with DIY benchmarks only
        registry_data = {
            "version": "1.0",
            "diy_benchmarks": {"optimization": {"qaoa": {"runner_class": "QaoaRunner"}}},
        }
        with open(registry_path, "w") as f:
            json.dump(registry_data, f)

        builtin_benchmarks = list_builtin_benchmarks()
        diy_benchmarks = list_diy_benchmarks()

        # Builtin benchmarks should be discovered on-the-fly
        assert isinstance(builtin_benchmarks, dict)

        # Check DIY benchmarks
        assert "optimization" in diy_benchmarks
        assert "qaoa" in diy_benchmarks["optimization"]


class TestValidationHelpers:
    """Tests for validation helper functions."""

    def test_check_duplicate_backend_no_duplicate(self):
        """Test check_duplicate_backend when no duplicate exists."""
        registry = {"diy_backends": {"backend1": {}}}

        # Should not raise
        check_duplicate_backend("backend2", registry)

    def test_check_duplicate_backend_with_duplicate(self):
        """Test check_duplicate_backend when DIY duplicate exists."""
        registry = {"diy_backends": {"backend1": {"builtin": False}}}

        with pytest.raises(BackendValidationError, match="already registered"):
            check_duplicate_backend("backend1", registry)

    def test_check_duplicate_benchmark_no_duplicate(self):
        """Test check_duplicate_benchmark when no duplicate exists."""
        registry = {
            "diy_benchmarks": {"chemistry": {"vqe": {}}},
        }

        # Should not raise (qaoa is different from vqe)
        check_duplicate_benchmark("qaoa", "chemistry", registry)

    def test_check_duplicate_benchmark_builtin_duplicate(self):
        """Test check_duplicate_benchmark when builtin duplicate exists."""
        # Note: This test checks against actual builtin benchmarks discovered on-the-fly
        # We need to use a real builtin benchmark name to test this
        registry = {
            "diy_benchmarks": {},
        }

        # Discover actual builtin benchmarks to find a real runner name
        builtin_benchmarks = _discover_builtin_benchmarks()

        # Skip test if no builtin benchmarks found
        if not builtin_benchmarks:
            pytest.skip("No builtin benchmarks found")

        # Get first category and first runner
        category = list(builtin_benchmarks.keys())[0]
        runners = builtin_benchmarks[category].get("runners", [])

        if not runners:
            pytest.skip("No builtin runners found")

        runner_name = runners[0]

        # Should raise because it conflicts with builtin
        with pytest.raises(BenchmarkValidationError, match="already exists as a built-in"):
            check_duplicate_benchmark(runner_name, category, registry)

    def test_check_duplicate_benchmark_diy_duplicate(self):
        """Test check_duplicate_benchmark when DIY duplicate exists."""
        registry = {
            "diy_benchmarks": {"optimization": {"custom_qaoa": {}}},
        }

        with pytest.raises(BenchmarkValidationError, match="already registered"):
            check_duplicate_benchmark("custom_qaoa", "optimization", registry)

    def test_check_duplicate_benchmark_case_ids_no_duplicate(self):
        """Test duplicate case ID validation passes for unique IDs."""
        registry = {
            "builtin_benchmarks": {
                "chemistry": {
                    "runners": ["vqe"],
                    "benchmark_cases": [{"uuid": "built_in_case"}],
                }
            },
            "diy_benchmarks": {
                "optimization": {"qaoa": {"benchmark_cases": [{"uuid": "diy_case"}]}}
            },
        }

        check_duplicate_benchmark_case_ids(
            [{"uuid": "new_case_1"}, {"uuid": "new_case_2"}],
            registry,
        )

    def test_check_duplicate_benchmark_case_ids_rejects_duplicate_payload_id(self):
        """Test duplicate case ID validation rejects repeated incoming IDs."""
        registry = {"builtin_benchmarks": {}, "diy_benchmarks": {}}

        with pytest.raises(BenchmarkValidationError, match="Duplicate benchmark case ID"):
            check_duplicate_benchmark_case_ids(
                [{"uuid": "dup_case"}, {"uuid": "dup_case"}],
                registry,
            )

    def test_check_duplicate_benchmark_case_ids_rejects_existing_registry_id(self):
        """Test duplicate case ID validation rejects collisions with registry."""
        registry = {
            "builtin_benchmarks": {
                "chemistry": {
                    "runners": ["vqe"],
                    "benchmark_cases": [{"uuid": "existing_case"}],
                }
            },
            "diy_benchmarks": {},
        }

        with pytest.raises(BenchmarkValidationError, match="already registered"):
            check_duplicate_benchmark_case_ids(
                [{"uuid": "existing_case"}],
                registry,
            )
