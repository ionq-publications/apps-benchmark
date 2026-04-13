"""
Tests for Qiskit Aer backend.

This module tests the QiskitAerSimBackend implementation.

Copyright (c) 2025 IonQ, Inc. All rights reserved.
Licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0
International (CC BY-NC-ND 4.0) License.
For details, visit: https://creativecommons.org/licenses/by-nc-nd/4.0/
"""

import pytest
from apps_benchmark.backends.qiskit_aer_sim_backend import QiskitAerSimBackend
from apps_benchmark.core.backend import AbstractBackend
from apps_benchmark.errors import BackendError
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


class TestQiskitAerSimBackendBasics:
    """Tests for basic Qiskit backend functionality."""

    def test_qiskit_backend_is_abstract_backend(self):
        """Test that QiskitAerSimBackend inherits from AbstractBackend."""
        backend = QiskitAerSimBackend()
        assert isinstance(backend, AbstractBackend)

    def test_qiskit_backend_name(self):
        """Test backend name."""
        backend = QiskitAerSimBackend()
        assert backend.name() == backend.simulator.name
        assert backend.name() == "aer_simulator"

    def test_qiskit_backend_instantiation_defaults(self):
        """Test default instantiation."""
        backend = QiskitAerSimBackend()
        assert backend.method == "automatic"
        assert backend.optimization_level == 1
        assert isinstance(backend.simulator, AerSimulator)

    def test_qiskit_backend_instantiation_custom_method(self):
        """Test instantiation with custom method."""
        backend = QiskitAerSimBackend(method="statevector")
        assert backend.method == "statevector"

    def test_qiskit_backend_instantiation_custom_optimization(self):
        """Test instantiation with custom optimization level."""
        backend = QiskitAerSimBackend(optimization_level=3)
        assert backend.optimization_level == 3

    def test_qiskit_backend_instantiation_with_options(self):
        """Test instantiation with additional simulator options."""
        backend = QiskitAerSimBackend(method="automatic", seed_simulator=12345)
        assert "seed_simulator" in backend.simulator_options

    def test_validate_connection(self):
        """Test validate_connection returns True for initialized simulator."""
        backend = QiskitAerSimBackend()
        assert backend.validate_connection() is True

    def test_get_simulator_info(self):
        """Test getting simulator info."""
        backend = QiskitAerSimBackend()
        info = backend.get_simulator_info()

        assert "backend_name" in info
        assert "backend_version" in info
        assert "simulator" in info
        assert info["simulator"] is True
        assert info["local"] is True
        assert info["method"] == "automatic"
        assert info["optimization_level"] == 1


class TestQiskitAerSimBackendExecution:
    """Tests for Qiskit backend circuit execution."""

    def test_run_rejects_circuit_with_measurements(self):
        """Test that pre-measured circuits are rejected."""
        backend = QiskitAerSimBackend()
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure([0, 1], [0, 1])

        with pytest.raises(BackendError, match=r"Measurement gates are unsupported.*circuit 0"):
            backend.run([qc], shots=1000)

    def test_run_circuit_without_measurements(self):
        """Test that backend auto-adds measurements if missing."""
        backend = QiskitAerSimBackend()
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        # No explicit measurement

        results, job_id, job_data = backend.run([qc], shots=1000)

        # Should still get results (measurements added automatically)
        assert len(results) == 1
        assert isinstance(results[0], dict)
        assert sum(results[0].values()) == 1000
        assert all(inst.operation.name != "measure" for inst in qc.data)

    def test_run_multiple_circuits(self):
        """Test running multiple circuits."""
        backend = QiskitAerSimBackend()

        qc1 = QuantumCircuit(1)
        qc1.x(0)  # Fixed: was x(1) on 1-qubit circuit

        qc2 = QuantumCircuit(2)
        qc2.h(0)
        qc2.cx(0, 1)

        qc3 = QuantumCircuit(3)
        qc3.h(0)
        qc3.h(1)
        qc3.h(2)

        results, job_id, job_data = backend.run([qc1, qc2, qc3], shots=500)

        assert len(results) == 3
        assert all(isinstance(r, dict) for r in results)
        assert all(sum(r.values()) == 500 for r in results)

    def test_bell_state_results(self):
        """Test that Bell state produces expected distribution."""
        backend = QiskitAerSimBackend()
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        results, _, _ = backend.run([qc], shots=10000)

        # Should get roughly 50/50 split between |00> and |11>
        result = results[0]
        assert "00" in result
        assert "11" in result

        # Check distribution (allow some statistical variation)
        assert 4500 <= result.get("00", 0) <= 5500
        assert 4500 <= result.get("11", 0) <= 5500

        # Should have minimal |01> and |10>
        assert result.get("01", 0) < 100
        assert result.get("10", 0) < 100

    def test_ghz_state_results(self):
        """Test GHZ state on 3 qubits."""
        backend = QiskitAerSimBackend()
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(0, 2)

        results, _, _ = backend.run([qc], shots=10000)

        result = results[0]
        # Should get roughly equal split between |000> and |111>
        assert "000" in result
        assert "111" in result
        assert 4500 <= result.get("000", 0) <= 5500
        assert 4500 <= result.get("111", 0) <= 5500

    def test_different_shot_counts(self):
        """Test that shot counts are respected."""
        backend = QiskitAerSimBackend()
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        results_100, _, _ = backend.run([qc], shots=100)
        results_1000, _, _ = backend.run([qc], shots=1000)
        results_5000, _, _ = backend.run([qc], shots=5000)

        assert sum(results_100[0].values()) == 100
        assert sum(results_1000[0].values()) == 1000
        assert sum(results_5000[0].values()) == 5000

    def test_job_data_serialization(self):
        """Test that job data is properly serialized."""
        backend = QiskitAerSimBackend()
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        results, job_id, job_data = backend.run([qc], shots=1000, job_name="test_job")

        assert "circuits" in job_data
        assert "shots" in job_data
        assert "job_name" in job_data
        assert job_data["shots"] == 1000
        assert job_data["job_name"] == "test_job"

    def test_job_data_round_trip(self):
        """Test job data serialization round trip."""
        backend = QiskitAerSimBackend()
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.h(1)
        qc.cx(0, 2)

        _, _, job_data = backend.run([qc], shots=2000, job_name="round_trip")

        # Deserialize
        hydrated = backend.de_serialize_job_data(job_data)

        assert hydrated["shots"] == 2000
        assert hydrated["job_name"] == "round_trip"
        assert len(hydrated["circuits"]) == 1
        assert isinstance(hydrated["circuits"][0], QuantumCircuit)


class TestQiskitAerSimBackendSimulationMethods:
    """Tests for different simulation methods."""

    def test_statevector_method(self):
        """Test statevector simulation method."""
        backend = QiskitAerSimBackend(method="statevector")
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)

        results, _, _ = backend.run([qc], shots=1000)

        assert len(results) == 1
        assert sum(results[0].values()) == 1000

    def test_automatic_method(self):
        """Test automatic simulation method."""
        backend = QiskitAerSimBackend(method="automatic")
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        results, _, _ = backend.run([qc], shots=1000)

        assert len(results) == 1
        assert sum(results[0].values()) == 1000


class TestQiskitAerSimBackendOptimizationLevels:
    """Tests for different optimization levels."""

    def test_optimization_level_0(self):
        """Test optimization level 0 (no optimization)."""
        backend = QiskitAerSimBackend(optimization_level=0)
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        results, _, _ = backend.run([qc], shots=1000)
        assert sum(results[0].values()) == 1000

    def test_optimization_level_3(self):
        """Test optimization level 3 (max optimization)."""
        backend = QiskitAerSimBackend(optimization_level=3)
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        results, _, _ = backend.run([qc], shots=1000)
        assert sum(results[0].values()) == 1000


class TestQiskitAerSimBackendVariousCircuits:
    """Tests for various circuit types."""

    def test_single_qubit_circuit(self):
        """Test 1-qubit circuit."""
        backend = QiskitAerSimBackend()
        qc = QuantumCircuit(1)
        qc.h(0)

        results, _, _ = backend.run([qc], shots=1000)

        # Should get roughly 50/50 split
        assert "0" in results[0]
        assert "1" in results[0]
        assert 400 <= results[0]["0"] <= 600
        assert 400 <= results[0]["1"] <= 600

    def test_identity_circuit(self):
        """Test circuit with no gates (identity)."""
        backend = QiskitAerSimBackend()
        qc = QuantumCircuit(2)
        # No gates

        results, _, _ = backend.run([qc], shots=1000)

        # Should get all |00>
        assert results[0].get("00", 0) == 1000

    def test_x_gate_circuit(self):
        """Test circuit with X gates."""
        backend = QiskitAerSimBackend()
        qc = QuantumCircuit(2)
        qc.x(0)
        qc.x(1)

        results, _, _ = backend.run([qc], shots=1000)

        # Should get all |11>
        assert results[0].get("11", 0) == 1000

    def test_circuit_with_all_single_qubit_gates(self):
        """Test circuit with various single-qubit gates."""
        backend = QiskitAerSimBackend()
        qc = QuantumCircuit(4)
        qc.h(0)
        qc.x(1)
        qc.y(2)
        qc.z(3)

        results, _, _ = backend.run([qc], shots=1000)

        # Should execute without error
        assert sum(results[0].values()) == 1000

    def test_circuit_with_multi_qubit_gates(self):
        """Test circuit with multi-qubit gates."""
        backend = QiskitAerSimBackend()
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.ccx(0, 1, 2)

        results, _, _ = backend.run([qc], shots=1000)

        # Should execute without error
        assert sum(results[0].values()) == 1000

    def test_large_circuit(self):
        """Test circuit with many qubits."""
        backend = QiskitAerSimBackend()
        qc = QuantumCircuit(10)
        qc.h(0)
        for i in range(9):
            qc.cx(i, i + 1)

        results, _, _ = backend.run([qc], shots=1000)

        # Should execute without error
        assert sum(results[0].values()) == 1000


class TestQiskitAerSimBackendErrorHandling:
    """Tests for error handling."""

    def test_invalid_simulation_method_raises_error(self):
        """Test that invalid simulation method raises error."""
        with pytest.raises(BackendError, match="Failed to initialize"):
            QiskitAerSimBackend(method="nonexistent_method")


class TestQiskitAerSimBackendIntegration:
    """Integration tests for Qiskit backend."""

    def test_qiskit_backend_usable_in_benchmarking(self):
        """Test that Qiskit backend works with benchmarking workflow."""
        backend = QiskitAerSimBackend()

        # Create VQE-like circuit
        qc = QuantumCircuit(4)
        qc.h(0)
        qc.h(1)
        qc.cx(0, 2)
        qc.cx(1, 3)
        qc.rz(0.5, 2)
        qc.rz(0.5, 3)

        results, job_id, job_data = backend.run([qc], shots=1000)

        # Results should be usable
        assert len(results) == 1
        assert sum(results[0].values()) == 1000
        assert len(job_id) > 0

    def test_multiple_sequential_runs(self):
        """Test multiple sequential executions."""
        backend = QiskitAerSimBackend()
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        results1, job_id1, _ = backend.run([qc], shots=500)
        results2, job_id2, _ = backend.run([qc], shots=500)
        results3, job_id3, _ = backend.run([qc], shots=500)

        # All should succeed
        assert sum(results1[0].values()) == 500
        assert sum(results2[0].values()) == 500
        assert sum(results3[0].values()) == 500

        # Job IDs should be different
        assert job_id1 != job_id2
        assert job_id2 != job_id3

    def test_qiskit_backend_determinism_with_seed(self):
        """Test that results are deterministic with seed."""
        backend1 = QiskitAerSimBackend(seed_simulator=42)
        backend2 = QiskitAerSimBackend(seed_simulator=42)

        qc = QuantumCircuit(3)
        qc.h(0)
        qc.h(1)
        qc.h(2)

        results1, _, _ = backend1.run([qc], shots=1000)
        results2, _, _ = backend2.run([qc], shots=1000)

        # With same seed, results should be identical
        assert results1[0] == results2[0]
