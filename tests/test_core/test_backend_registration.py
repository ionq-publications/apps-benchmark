"""
Tests for backend registration and discovery.

This module tests that built-in backends are discovered correctly
and can be loaded and validated.

Copyright (c) 2025 IonQ, Inc. All rights reserved.
Licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0
International (CC BY-NC-ND 4.0) License.
For details, visit: https://creativecommons.org/licenses/by-nc-nd/4.0/
"""

from apps_benchmark.backends.mock_backend import MockBackend
from apps_benchmark.backends.qiskit_aer_sim_backend import QiskitAerSimBackend
from apps_benchmark.core.registry import (
    _discover_builtin_backends,
    initialize_registries,
    list_builtin_backends,
    list_diy_backends,
)


class TestBuiltinBackendDiscovery:
    """Tests for discovering built-in backends."""

    def test_discover_builtin_backends_finds_mock(self):
        """Test that mock backend is discovered."""
        backend_builtin = _discover_builtin_backends()

        assert "mock_backend" in backend_builtin
        backend_info = backend_builtin["mock_backend"]
        assert backend_info["builtin"] is True
        assert backend_info["class"] == "MockBackend"
        assert "apps_benchmark.backends.mock_backend" in backend_info["module"]

    def test_discover_builtin_backends_finds_qiskit(self):
        """Test that Qiskit backend is discovered."""
        backend_builtin = _discover_builtin_backends()

        assert "qiskit_aer_sim_backend" in backend_builtin
        backend_info = backend_builtin["qiskit_aer_sim_backend"]
        assert backend_info["builtin"] is True
        assert backend_info["class"] == "QiskitAerSimBackend"
        assert "apps_benchmark.backends.qiskit_aer_sim_backend" in backend_info["module"]

    def test_discovered_backends_have_required_fields(self):
        """Test that discovered backends have all required fields."""
        backends_builtin = _discover_builtin_backends()

        required_fields = ["module", "class", "builtin", "location", "registered_at"]

        for backend_name, backend_info in backends_builtin.items():
            for field in required_fields:
                assert field in backend_info, f"Backend '{backend_name}' missing field '{field}'"

    def test_discovered_backends_are_marked_builtin(self):
        """Test that all discovered backends are marked as built-in."""
        backends = _discover_builtin_backends()

        for _backend_name, backend_info in backends.items():
            assert backend_info["builtin"] is True

    def test_discovered_backends_have_timestamps(self):
        """Test that discovered backends have registration timestamps."""
        backends = _discover_builtin_backends()

        for _backend_name, backend_info in backends.items():
            assert "registered_at" in backend_info
            # Timestamp should be ISO format string
            assert isinstance(backend_info["registered_at"], str)
            assert len(backend_info["registered_at"]) > 10  # ISO timestamp is longer


class TestBackendRegistryInitialization:
    """Tests for backend registry initialization."""

    def test_initialize_registries_discovers_backends(self, tmp_path, monkeypatch):
        """Test that initialize_registries creates empty DIY registry (builtins discovered on-the-fly)."""
        monkeypatch.setattr("apps_benchmark.core.registry.Path.home", lambda: tmp_path)

        initialize_registries()

        local_dev = tmp_path / "local_dev"
        backends_registry = local_dev / "backends.json"

        assert backends_registry.exists()

        # Load and check registry
        import json

        with open(backends_registry) as f:
            registry = json.load(f)

        # Registry should only contain DIY backends (empty on first init)
        assert "diy_backends" in registry
        assert registry["diy_backends"] == {}

        # Builtin backends should be discovered
        builtin_backends = list_builtin_backends()
        diy_backends = list_diy_backends()

        assert len(builtin_backends) >= 2  # At least mock and qiskit
        assert "mock_backend" in builtin_backends
        assert "qiskit_aer_sim_backend" in builtin_backends
        assert diy_backends == {}  # No DIY backends registered yet

    def test_list_backends_returns_builtin_backends(self, tmp_path, monkeypatch):
        """Test that list_backends returns built-in backends."""
        monkeypatch.setattr("apps_benchmark.core.registry.Path.home", lambda: tmp_path)

        initialize_registries()

        builtin_backends = list_builtin_backends()
        diy_backends = list_diy_backends()

        # Check builtin backends
        assert len(builtin_backends) >= 2
        assert "mock_backend" in builtin_backends
        assert "qiskit_aer_sim_backend" in builtin_backends

        # Check DIY backends (should be empty on init)
        assert diy_backends == {}


class TestBackendLoading:
    """Tests for loading backends from registry."""

    def test_load_mock_backend_from_module(self):
        """Test loading mock backend directly."""

        backend = MockBackend()
        assert backend.name() == "mock"
        assert backend.validate_connection() is True

    def test_load_qiskit_backend_from_module(self):
        """Test loading Qiskit backend directly."""

        backend = QiskitAerSimBackend()
        assert backend.name() == backend.simulator.name
        assert backend.name() == "aer_simulator"
        assert backend.validate_connection() is True

    def test_backends_have_correct_interfaces(self):
        """Test that backends implement required interfaces."""
        from apps_benchmark.backends.qiskit_aer_sim_backend import QiskitAerSimBackend
        from apps_benchmark.core.backend import AbstractBackend

        # Check mock backend
        mock_backend = MockBackend()
        assert isinstance(mock_backend, AbstractBackend)
        assert hasattr(mock_backend, "name")
        assert hasattr(mock_backend, "run")
        assert hasattr(mock_backend, "validate_connection")

        # Check Qiskit backend
        qiskit_backend = QiskitAerSimBackend()
        assert isinstance(qiskit_backend, AbstractBackend)
        assert hasattr(qiskit_backend, "name")
        assert hasattr(qiskit_backend, "run")
        assert hasattr(qiskit_backend, "validate_connection")


class TestBackendValidation:
    """Tests for backend validation."""

    def test_mock_backend_passes_validation(self):
        """Test that mock backend passes interface validation."""
        from apps_benchmark.utils.validation import validate_backend_interface

        # Should not raise
        validate_backend_interface(MockBackend)

    def test_qiskit_backend_passes_validation(self):
        """Test that Qiskit backend passes interface validation."""
        from apps_benchmark.utils.validation import validate_backend_interface

        # Should not raise
        validate_backend_interface(QiskitAerSimBackend)

    def test_backends_can_be_instantiated(self):
        """Test that backends can be instantiated without errors."""
        from apps_benchmark.backends.qiskit_aer_sim_backend import QiskitAerSimBackend

        # Should not raise
        mock = MockBackend()
        qiskit = QiskitAerSimBackend()

        assert mock is not None
        assert qiskit is not None


class TestBackendFunctionality:
    """Tests for basic backend functionality."""

    def test_mock_backend_basic_execution(self):
        """Test that mock backend can execute circuits."""
        from qiskit import QuantumCircuit

        backend = MockBackend()
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        results, job_id, job_data = backend.run([qc], shots=1000)

        assert len(results) == 1
        assert isinstance(results[0], dict)
        assert sum(results[0].values()) == 1000
        assert isinstance(job_id, str)
        assert isinstance(job_data, dict)

    def test_qiskit_backend_basic_execution(self):
        """Test that Qiskit backend can execute circuits."""
        from qiskit import QuantumCircuit

        backend = QiskitAerSimBackend()
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        results, job_id, job_data = backend.run([qc], shots=1000)

        assert len(results) == 1
        assert isinstance(results[0], dict)
        assert sum(results[0].values()) == 1000
        assert isinstance(job_id, str)
        assert isinstance(job_data, dict)

    def test_both_backends_validate_successfully(self):
        """Test that both backends validate successfully."""
        from apps_benchmark.backends.qiskit_aer_sim_backend import QiskitAerSimBackend

        mock_backend = MockBackend()
        qiskit_backend = QiskitAerSimBackend()

        assert mock_backend.validate_connection() is True
        assert qiskit_backend.validate_connection() is True


class TestBackendRegistryPersistence:
    """Tests for registry persistence."""

    def test_registry_persists_after_initialization(self, tmp_path, monkeypatch):
        """Test that registry file persists."""
        monkeypatch.setattr("apps_benchmark.core.registry.Path.home", lambda: tmp_path)

        # Initialize
        initialize_registries()

        # Check file exists
        local_dev = tmp_path / "local_dev"
        backends_registry = local_dev / "backends.json"
        assert backends_registry.exists()

        # Re-initialize (should not recreate)
        initialize_registries()
        assert backends_registry.exists()

    def test_registry_can_be_read_multiple_times(self, tmp_path, monkeypatch):
        """Test that registry can be read multiple times."""
        monkeypatch.setattr("apps_benchmark.core.registry.Path.home", lambda: tmp_path)

        initialize_registries()

        builtin1 = list_builtin_backends()
        builtin2 = list_builtin_backends()
        builtin3 = list_builtin_backends()

        diy1 = list_diy_backends()
        diy2 = list_diy_backends()
        diy3 = list_diy_backends()

        ## NOTE: there are slight timestamp differences on repeated
        # writes, so we need to clean those before comparing
        def clean_timestamps(registry):
            for backend_info in registry.values():
                if "registered_at" in backend_info:
                    backend_info["registered_at"] = "TIMESTAMP"
            return registry

        builtin1 = clean_timestamps(builtin1)
        builtin2 = clean_timestamps(builtin2)
        builtin3 = clean_timestamps(builtin3)

        diy1 = clean_timestamps(diy1)
        diy2 = clean_timestamps(diy2)
        diy3 = clean_timestamps(diy3)

        assert builtin1 == builtin2, (
            f"Builtin backends differ on repeated reads: \n{builtin1} vs \n{builtin2}"
        )
        assert builtin2 == builtin3
        assert diy1 == diy2
        assert diy2 == diy3


class TestBackendDiscoveryRobustness:
    """Tests for robustness of backend discovery."""

    def test_discover_ignores_non_backend_files(self):
        """Test that discovery ignores files that don't contain backends."""
        backends = _discover_builtin_backends()

        # Should not include __init__.py
        assert "__init__" not in backends

    def test_discover_only_finds_abstractbackend_subclasses(self):
        """Test that discovery only finds AbstractBackend subclasses."""
        backends = _discover_builtin_backends()

        # All discovered backends should be backend classes
        for _backend_name, backend_info in backends.items():
            assert backend_info["class"] in [
                "MockBackend",
                "QiskitAerSimBackend",
                "IonqCloudBackend",
            ]

    def test_discover_handles_multiple_backends(self):
        """Test that discovery handles multiple backends correctly."""
        backends = _discover_builtin_backends()

        # Should have at least 2 backends
        assert len(backends) >= 2

        # Each backend should have unique class name
        class_names = [info["class"] for info in backends.values()]
        assert len(class_names) == len(set(class_names))


class TestBackendIntegration:
    """Integration tests for backend system."""

    def test_full_backend_workflow(self, tmp_path, monkeypatch):
        """Test complete backend workflow."""
        monkeypatch.setattr("apps_benchmark.core.registry.Path.home", lambda: tmp_path)

        # 1. Initialize registries
        initialize_registries()

        # 2. List backends
        builtin_backends = list_builtin_backends()
        # list_diy_backends()
        assert len(builtin_backends) >= 2

        # 3. Load and use mock backend
        from qiskit import QuantumCircuit

        backend = MockBackend()
        qc = QuantumCircuit(2)
        results, job_id, _ = backend.run([qc], shots=100)
        assert len(results) == 1

        # 4. Load and use Qiskit backend
        from apps_benchmark.backends.qiskit_aer_sim_backend import QiskitAerSimBackend

        backend = QiskitAerSimBackend()
        qc = QuantumCircuit(2)
        qc.h(0)
        results, job_id, _ = backend.run([qc], shots=100)
        assert len(results) == 1

    def test_backends_work_independently(self):
        """Test that multiple backends can be used independently."""
        from apps_benchmark.backends.qiskit_aer_sim_backend import QiskitAerSimBackend
        from qiskit import QuantumCircuit

        mock = MockBackend()
        qiskit = QiskitAerSimBackend()

        qc = QuantumCircuit(2)
        qc.h(0)

        # Run on both backends
        results_mock, _, _ = mock.run([qc], shots=100)
        results_qiskit, _, _ = qiskit.run([qc], shots=100)

        # Both should succeed
        assert len(results_mock) == 1
        assert len(results_qiskit) == 1
        assert sum(results_mock[0].values()) == 100
        assert sum(results_qiskit[0].values()) == 100
