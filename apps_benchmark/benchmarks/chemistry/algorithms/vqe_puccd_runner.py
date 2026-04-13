"""
VQE with paired-UCC-Doubles ansatz for chemistry benchmarks.

Copyright (c) 2025 IonQ, Inc. All rights reserved.
Licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0
International (CC BY-NC-ND 4.0) License.
For details, visit: https://creativecommons.org/licenses/by-nc-nd/4.0/
"""
import time
import warnings
from numbers import Integral
from typing import Any

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

# Imports needed for gate_counts
from qiskit.compiler import transpile
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGCircuitError
from qiskit.quantum_info import SparsePauliOp
from qiskit.result import sampled_expectation_value
from scipy.optimize import minimize

# Use explicit V1/V2 primitives to match current Qiskit API
try:
    # Prefer explicit V1 if available in this Qiskit version
    from qiskit.primitives import EstimatorV1
except Exception:  # pragma: no cover - environment may not have V1
    EstimatorV1 = None

from qiskit.primitives import StatevectorEstimator as StatevectorEstimatorV2

from apps_benchmark.core.backend import (
    AbstractBackend,
    MeasurementBatch,
    MeasurementHistogram,
)
from apps_benchmark.core.qc_benchmark_runner import CircuitBenchmarkRunner
from apps_benchmark.primitives.benchmark_case import BenchmarkCase


class VqePuccdRunner(CircuitBenchmarkRunner):
    """
    VQE implementation using paired-UCC-Doubles ansatz.
    Handles optimization and inference modes.
    Supports both shot-based sampling and exact statevector simulation.
    """

    _current_hamiltonian: SparsePauliOp | None = None
    _current_ansatz: QuantumCircuit | None = None
    _current_num_qubits: int | None = None
    _current_backend: AbstractBackend | None = None  # Used only for shot-based
    _current_shots: int | None = None  # Used only for shot-based
    _iteration_count: int = 0
    _num_circuits_per_eval: int = 1  # Relevant for shot-based
    _warned_space: bool = False  # Relevant for shot-based counts parsing
    _optimizer_config: dict[str, Any] = {
        "method": "COBYLA",
        "options": {"maxiter": 10_000, "tol": 1e-6},
    }
    _precomputed_parameters: list[float] | None = None
    _energy_history: list[float] = []
    _param_history: list[np.ndarray] = []
    _eval_times: list[float] = []

    # flags for exact expectation values
    _exact_simulation: bool = False
    _qiskit_primitive_version: str = "v1"  # Default requested version; may fall back to V2
    _current_exact_estimator: Any | None = None

    def name(self) -> str:
        return "vqe_puccd"

    def _require_current_hamiltonian(self) -> SparsePauliOp:
        if self._current_hamiltonian is None:
            raise RuntimeError("Current Hamiltonian is not initialized.")
        return self._current_hamiltonian

    def _require_current_ansatz(self) -> QuantumCircuit:
        if self._current_ansatz is None:
            raise RuntimeError("Current ansatz is not initialized.")
        return self._current_ansatz

    def _require_current_num_qubits(self) -> int:
        if self._current_num_qubits is None:
            raise RuntimeError("Current qubit count is not initialized.")
        return self._current_num_qubits

    @staticmethod
    def _coerce_optional_float(value: Any) -> float | None:
        if isinstance(value, (int, float, np.integer, np.floating)):
            value_float = float(value)
            if np.isnan(value_float):
                return None
            return value_float
        return None

    def _build_ansatz(self, num_qubits: int, num_occ_pairs: int) -> QuantumCircuit:
        excitation_pairs = []
        for i in range(num_occ_pairs):
            for a in range(num_occ_pairs, num_qubits):
                excitation_pairs.append([i, a])

        circuit = QuantumCircuit(num_qubits, name="pUCCD Ansatz")

        # Initialize Hartree-Fock state
        for occ in range(num_occ_pairs):
            circuit.x(occ)
        circuit.barrier()

        # Apply parameterized double excitations
        parameter_vector = ParameterVector("θ", length=len(excitation_pairs))
        for idx, pair in enumerate(excitation_pairs):
            theta = parameter_vector[idx]
            i, a = pair[0], pair[1]
            circuit.s(i)
            circuit.s(a)
            circuit.h(a)
            circuit.cx(a, i)
            circuit.ry(theta, i)
            circuit.ry(theta, a)
            circuit.cx(a, i)
            circuit.h(a)
            circuit.sdg(a)
            circuit.sdg(i)
            circuit.barrier()

        return circuit

    def _prepare_circuits(
        self, base_circuit: QuantumCircuit, operator: SparsePauliOp
    ) -> tuple[list[QuantumCircuit], list[SparsePauliOp]]:
        # Note this is only for shot-based simulation but is bypassed in exact mode
        basis_change_map = {"X": ["h"], "Y": ["sdg", "h"], "Z": [], "I": []}
        commuting_ops = operator.group_commuting(qubit_wise=True)

        qc_list = []
        formatted_obs = []
        num_qubits = base_circuit.num_qubits

        for comm_op in commuting_ops:
            if comm_op.num_qubits != num_qubits:
                raise ValueError(
                    f"Commuting op qubits {comm_op.num_qubits} != circuit qubits {num_qubits}"
                )

            pauli_labels = comm_op.paulis.to_labels(array=True)
            if pauli_labels.ndim == 0:
                pauli_labels = np.array([str(pauli_labels)])
            if pauli_labels.size == 0:
                pauli_labels = np.array(["I" * num_qubits])

            first_label_str = str(pauli_labels[0]) if pauli_labels.size > 0 else "I" * num_qubits
            if len(first_label_str) != num_qubits:
                first_label_str = first_label_str.ljust(num_qubits, "I")

            basis = ""
            for qubit_idx in range(num_qubits):
                current_qubit_basis = "Z"
                ops_on_qubit = {
                    p[qubit_idx] for p in pauli_labels if len(p) > qubit_idx and p[qubit_idx] != "I"
                }
                if "X" in ops_on_qubit:
                    current_qubit_basis = "X"
                elif "Y" in ops_on_qubit:
                    current_qubit_basis = "Y"
                basis += current_qubit_basis

            term_coeff_list = comm_op.to_list()
            if term_coeff_list:
                terms, coeffs = zip(*term_coeff_list, strict=True)
                coefficient_values = list(coeffs)
            else:
                terms = ()
                coefficient_values = []
            new_terms = []
            for term in terms:
                term_str = term.ljust(num_qubits, "I")
                new_term = "".join(["Z" if c in "XY" else c for c in term_str])
                new_terms.append(new_term)

            if not new_terms:
                new_terms = ["I" * num_qubits]
                coefficient_values = [coefficient_values[0] if coefficient_values else 1.0]

            new_op = SparsePauliOp(new_terms, coeffs=coefficient_values)
            if new_op.num_qubits != num_qubits:
                try:
                    padded_labels = [
                        label.ljust(num_qubits, "I") for label in new_op.paulis.to_labels()
                    ]
                    new_op = SparsePauliOp(padded_labels, coeffs=new_op.coeffs)
                    if new_op.num_qubits != num_qubits:
                        print(
                            f"Warning: Padded obs qubits {new_op.num_qubits} != {num_qubits} even after manual pad."
                        )
                except Exception as pad_err:
                    print(
                        f"Warning: Obs qubits {new_op.num_qubits} != {num_qubits}. Padding failed: {pad_err}"
                    )

            formatted_obs.append(new_op)

            basis_circuit = QuantumCircuit(num_qubits, name=f"BasisRot:{basis}")
            basis_circuit.barrier()
            for idx, pauli_char in enumerate(reversed(basis)):
                if pauli_char != "I":
                    for gate_name in basis_change_map[pauli_char]:
                        getattr(basis_circuit, gate_name)(idx)

            composed_qc = base_circuit.compose(basis_circuit)
            qc_list.append(composed_qc)

        return qc_list, formatted_obs

    def _normalize_counts(
        self,
        counts: MeasurementHistogram,
        num_qubits: int,
    ) -> dict[str, float]:
        """
        Normalize measurement counts into probabilities. (ONLY for shot-based)
        Handles the specific "bitstring bitstring" format from some backends.
        """
        shots = float(sum(counts.values()))
        if shots == 0:
            return {}
        probabilities = {}
        duplicate_format = False
        if counts and " " in next(iter(counts.keys())):
            sample_key = next(iter(counts.keys()))
            parts = sample_key.split(" ")
            if len(parts) == 2 and parts[0] == parts[1] and len(parts[0]) == num_qubits:
                duplicate_format = True
                if not self._warned_space:
                    print("      Note: Detected 'bitstring bitstring' format. Using first part.")
                    self._warned_space = True
        for bitstring, count in counts.items():
            if duplicate_format and " " in bitstring:
                processed_bitstring = bitstring.split(" ")[0]
            else:
                processed_bitstring = bitstring
                if " " in processed_bitstring:  # for cases like "00 0" -> "00"
                    processed_bitstring = processed_bitstring.split(" ")[0]
                    if not self._warned_space:
                        print(
                            f"      Note: Found space-separated counts keys. Using first part: '{bitstring}' -> '{processed_bitstring}'."
                        )
                        self._warned_space = True
            if (
                not all(c in "01" for c in processed_bitstring)
                or len(processed_bitstring) != num_qubits
            ):
                # print(f"      Warning: Skipping invalid bitstring '{processed_bitstring}' (original: '{bitstring}') for num_qubits {num_qubits}.")
                continue
            probabilities[processed_bitstring] = count / shots
        return probabilities

    def _compute_energy_from_results(
        self,
        results: MeasurementBatch,
        observables: list[SparsePauliOp],
        num_qubits: int,
        shots: int,
    ) -> tuple[float, float]:
        """Computes total energy and its standard error from backend counts results. (ONLY for shot-based)"""
        total_energy_val = 0.0
        sum_of_individual_se_sq = 0.0
        se_is_defined = (shots >= 2) #Standard Error rate is/is-not defined

        if len(results) != len(observables):
            raise ValueError(
                f"Results count {len(results)} != Observables count {len(observables)}"
            )

        if shots > 0 and not se_is_defined:
            warnings.warn(
                "Standard error is undefined for shot-based runs with fewer than 2 shots; reporting NaN.",
                stacklevel=2,
            )

        for i, op_group_Z_basis in enumerate(observables):
            counts_for_group = results[i] if i < len(results) else {}
            if not counts_for_group:
                continue

            probabilities_for_group = self._normalize_counts(counts_for_group, num_qubits)
            if not probabilities_for_group:
                continue

            mu_group = sampled_expectation_value(probabilities_for_group, op_group_Z_basis)
            exp_val_sq_group = 0.0
            for bitstring, prob in probabilities_for_group.items():
                value_for_bitstring = 0.0
                for pauli_label_str, coeff in op_group_Z_basis.to_list():
                    term_eigenvalue = 1.0
                    for char_idx in range(num_qubits):
                        pauli_char = pauli_label_str[char_idx]
                        bit_val = int(bitstring[char_idx])
                        if pauli_char == "Z" and bit_val == 1:
                            term_eigenvalue *= -1.0
                    value_for_bitstring += coeff * term_eigenvalue
                exp_val_sq_group += prob * (value_for_bitstring**2)

            var_single_shot_group = exp_val_sq_group - (mu_group**2)
            current_se_sq = np.nan

            real_variance = (
                var_single_shot_group.real
                if isinstance(var_single_shot_group, complex)
                else var_single_shot_group
            )

            if real_variance < 0:
                if not np.isclose(real_variance, 0, atol=1e-9):
                    print(
                        f"    VqePuccdRunner Warning: Computed real variance is negative ({real_variance:.2e}) for group {i}. Clamping to 0 for SE."
                    )
                real_variance = (
                    0.0  # variance cannot be negative so force zero (shouldn't happen tho)
                )

            if se_is_defined:
                current_se_sq = real_variance / shots

            total_energy_val += mu_group
            if not np.isnan(current_se_sq):
                sum_of_individual_se_sq += current_se_sq
            else:
                sum_of_individual_se_sq = np.nan

        final_total_se = np.nan
        if not np.isnan(sum_of_individual_se_sq):
            if sum_of_individual_se_sq >= 0:
                final_total_se = np.sqrt(sum_of_individual_se_sq)

        return total_energy_val, final_total_se

    def setup_algo_inputs(self, problem_instance: BenchmarkCase) -> tuple[Any, ...]:
        print(f"Setting up VQE for instance: {problem_instance.instance_name}")
        self._precomputed_parameters = None
        self._energy_history = []
        self._param_history = []
        self._eval_times = []
        try:
            hamiltonian_dict = problem_instance.data["paired_hamiltonian_dict"]
            hamiltonian = SparsePauliOp.from_list(list(hamiltonian_dict.items()))
            num_qubits = hamiltonian.num_qubits
            self._current_num_qubits = num_qubits
        except KeyError as exc:
            raise ValueError(
                "BenchmarkCase 'data' must contain 'paired_hamiltonian_dict'."
            ) from exc
        except Exception as exc:
            raise ValueError(f"Failed to load Hamiltonian: {exc}") from exc
        try:
            num_occ_pairs = int(problem_instance.data["num_alpha"])
        except KeyError as exc:
            raise ValueError("BenchmarkCase 'data' must contain 'num_alpha'.") from exc

        ansatz_circuit = self._build_ansatz(num_qubits, num_occ_pairs)
        print(f"  Hamiltonian loaded ({len(hamiltonian)} terms, {num_qubits} qubits)")
        print(f"  Ansatz built ({ansatz_circuit.num_parameters} parameters)")
        self._current_hamiltonian = hamiltonian
        self._current_ansatz = ansatz_circuit

        self._num_circuits_per_eval = 1
        try:
            if ansatz_circuit.num_parameters > 0:
                dummy_params = np.zeros(ansatz_circuit.num_parameters)
                dummy_bound_ansatz = ansatz_circuit.assign_parameters(dummy_params)
            else:
                dummy_bound_ansatz = ansatz_circuit

            qc_list_template, _ = self._prepare_circuits(dummy_bound_ansatz, hamiltonian)
            self._num_circuits_per_eval = len(qc_list_template)
            print(
                f"  Determined {self._num_circuits_per_eval} circuits needed per evaluation (for shot-based mode)."
            )
        except Exception as e:
            print(f"  Warning: Could not determine num_circuits_per_eval: {e}. Using 1.")

        loaded_params = problem_instance.data.get("optimal_parameters")
        if loaded_params is not None:
            print("  Found 'optimal_parameters' in instance data (will use for inference).")
            if (
                ansatz_circuit.num_parameters == 0
                and isinstance(loaded_params, list)
                and len(loaded_params) == 0
            ):
                self._precomputed_parameters = []
            elif (
                isinstance(loaded_params, list)
                and len(loaded_params) == ansatz_circuit.num_parameters
            ):
                self._precomputed_parameters = loaded_params
            elif ansatz_circuit.num_parameters > 0:
                print(
                    f"    Warning: optimal_parameters format/length mismatch (expected {ansatz_circuit.num_parameters}, got {len(loaded_params) if isinstance(loaded_params, list) else type(loaded_params)}). Cannot use for inference."
                )
                self._precomputed_parameters = None  # set to None if we fail

        optimizer_config = problem_instance.data.get("optimizer_config", {})
        if optimizer_config:  # e.g. if not empty dict
            self._optimizer_config = optimizer_config
            method = optimizer_config.get("method", "COBYLA")  # default to COBYLA if undefined
            options = optimizer_config.get("options", {})
            tol = options.get("tol", "default")
            maxiter = options.get("maxiter", 10_000)  # Default if undefined
            print(f"  Using optimizer {method} with tolerance {tol}, max iterations {maxiter}")
        else:  # just set some defaults in case
            method = self._optimizer_config.get("method", "COBYLA")
            options = self._optimizer_config.get("options", {"maxiter": 10_000, "tol": 1e-6})
            tol = options.get("tol", "default")
            maxiter = options.get("maxiter", 10_000)
            print(
                f"  Using default optimizer {method} with tolerance {tol}, max iterations {maxiter}"
            )

        return (hamiltonian, ansatz_circuit)

    def _objective_function(self, parameters: np.ndarray) -> float:
        self._iteration_count += 1
        start_time = time.time()
        energy = float("nan")

        try:
            if self._exact_simulation:
                estimator = self._current_exact_estimator
                if estimator is None:
                    raise RuntimeError("Exact estimator was not initialized.")
                observable = self._require_current_hamiltonian()
                ansatz = self._require_current_ansatz()

                # Route based on available estimator implementation
                if isinstance(estimator, StatevectorEstimatorV2):
                    estimator_v2: StatevectorEstimatorV2 = estimator
                    pub = (ansatz, observable, parameters)
                    job = estimator_v2.run(pubs=[pub])
                    result = job.result()
                    raw_evs = result[0].data.evs
                    energy = float(np.squeeze(raw_evs))
                else:
                    job = estimator.run(
                        circuits=[ansatz],
                        observables=[observable],
                        parameter_values=[parameters],
                    )
                    result = job.result()
                    energy = result.values[0]
            else:
                if self._current_backend is None or self._current_shots is None:
                    raise RuntimeError("Backend or shots not set for shot-based simulation.")
                bound_ansatz = self._require_current_ansatz().assign_parameters(parameters)
                qc_list, formatted_observables = self._prepare_circuits(
                    bound_ansatz,
                    self._require_current_hamiltonian(),
                )
                if len(qc_list) != self._num_circuits_per_eval:
                    print(
                        f"    Warning: Circuit count mismatch in iteration {self._iteration_count}. Expected {self._num_circuits_per_eval}, got {len(qc_list)}"
                    )
                results_list, _, _ = self._current_backend.run(qc_list, shots=self._current_shots)
                energy, _ = self._compute_energy_from_results(
                    results_list,
                    formatted_observables,
                    self._require_current_num_qubits(),
                    self._current_shots,
                )

            end_time = time.time()
            eval_time = end_time - start_time
            self._energy_history.append(float(energy))
            self._param_history.append(parameters.copy())
            self._eval_times.append(eval_time)

            if self._iteration_count % 10 == 0 or self._iteration_count <= 5:
                mode_str = "Exact" if self._exact_simulation else f"Shots ({self._current_shots})"
                print(
                    f"    Iter {self._iteration_count:4d}: E = {energy:.8f}, time = {eval_time:.3f}s ({mode_str} - V{self._qiskit_primitive_version if self._exact_simulation else 'N/A'})"
                )
            return energy
        except Exception as e:
            print(f"    ERROR in objective function (iteration {self._iteration_count}): {e}")
            end_time = time.time()
            self._eval_times.append(end_time - start_time)
            self._energy_history.append(float("nan"))
            self._param_history.append(parameters.copy())
            return float("nan")

    def execute_benchmark_algo(
        self,
        algo_inputs: tuple[Any, ...],
        backend: AbstractBackend,
        shots: int,
        **kwargs: Any,
    ) -> dict[str, Any]:
        hamiltonian, ansatz = algo_inputs
        exact_simulation = bool(kwargs.get("exact_simulation", False))
        qiskit_primitive_version = str(kwargs.get("qiskit_primitive_version", "v1"))

        self._current_hamiltonian = hamiltonian
        self._current_ansatz = ansatz
        self._current_num_qubits = ansatz.num_qubits
        self._iteration_count = 0
        self._warned_space = False
        self._energy_history = []
        self._param_history = []
        self._eval_times = []

        self._exact_simulation = exact_simulation
        self._qiskit_primitive_version = qiskit_primitive_version
        self._current_exact_estimator = None
        self._current_backend = None
        self._current_shots = None
        self._benchmark_qc = []
        self._latest_counts = []
        self._job_id = "local"
        self._job_data = {}

        final_energy: float = float("nan")
        final_energy_se: float | None = None
        optimal_parameters: np.ndarray | None = None
        nfev: int = 0
        optimizer_success: bool = False
        optimizer_message: str = ""
        shots_per_eval: int = 0  # This is specific to shot-based mode
        num_circuits_per_eval_used: int = 1  # Default to 1 for exact, updated for shot-based
        total_circuits_run: int = 0
        optimization_metrics: dict[str, Any] = {}
        final_circuits_to_save: list[QuantumCircuit] | None = None

        simulation_mode_str = "Exact Statevector" if exact_simulation else "Shot-Based Sampling"
        print(f"Executing VQE in {simulation_mode_str} mode.")

        if exact_simulation:
            print(f"  Using Qiskit Primitive Version: {qiskit_primitive_version}")
            try:
                if qiskit_primitive_version == "v1":
                    if EstimatorV1 is not None:
                        self._current_exact_estimator = EstimatorV1()
                        print("  Initialized V1 Estimator.")
                    else:
                        # Fall back to V2 statevector estimator when V1 is unavailable
                        self._current_exact_estimator = StatevectorEstimatorV2()
                        print("  V1 Estimator unavailable; using V2 StatevectorEstimator fallback.")
                elif qiskit_primitive_version == "v2":
                    self._current_exact_estimator = StatevectorEstimatorV2()
                    print("  Initialized V2 StatevectorEstimator.")
                else:
                    raise ValueError(f"Unsupported primitive version: {qiskit_primitive_version}")
            except Exception as exc:
                print(
                    f"  ERROR: Failed to initialize exact estimator (Version {qiskit_primitive_version}): {exc}"
                )
                raise RuntimeError(f"Failed to initialize exact estimator: {exc}") from exc
            shots_per_eval = 0  # No shots in exact mode
            final_energy_se = 0.0  # SE is 0 for exact energy
            num_circuits_per_eval_used = 1  # Only one circuit in exact (no basis rotations)
        else:
            print(f"  Using Backend: {backend.name()}, Shots per evaluation point: {shots}")
            if shots <= 0:
                raise ValueError("Shots must be positive for shot-based simulation.")
            self._current_backend = backend
            self._current_shots = shots
            shots_per_eval = shots
            num_circuits_per_eval_used = self._num_circuits_per_eval

        if self._precomputed_parameters is not None:
            print("Running in INFERENCE mode (using parameters from BenchmarkCase).")
            optimal_parameters = np.array(self._precomputed_parameters)
            if optimal_parameters is None:
                raise ValueError(
                    "Optimal parameters are None in inference mode despite _precomputed_parameters being set."
                )

            bound_ansatz = ansatz.assign_parameters(optimal_parameters)
            eval_time = 0.0

            try:
                print("  Evaluating energy with loaded parameters...")
                eval_start_time = time.time()

                if self._exact_simulation:
                    if self._current_exact_estimator is None:
                        raise RuntimeError("Estimator not init.")
                    final_circuits_to_save = [bound_ansatz]
                    if isinstance(self._current_exact_estimator, StatevectorEstimatorV2):
                        estimator_v2: StatevectorEstimatorV2 = self._current_exact_estimator
                        pub = (ansatz, hamiltonian, optimal_parameters)
                        job = estimator_v2.run(pubs=[pub])
                        raw_evs = job.result()[0].data.evs
                        final_energy = float(np.squeeze(raw_evs))
                    else:
                        estimator = self._current_exact_estimator
                        if estimator is None:
                            raise RuntimeError("Estimator not init.")
                        job = estimator.run([ansatz], [hamiltonian], [optimal_parameters])
                        final_energy = job.result().values[0]
                    total_circuits_run = 1
                    final_energy_se = 0.0  # Exact
                    self._benchmark_qc = final_circuits_to_save
                else:
                    final_circuits_to_save, formatted_observables = self._prepare_circuits(
                        bound_ansatz, hamiltonian
                    )
                    if len(final_circuits_to_save) != num_circuits_per_eval_used:
                        print(
                            f"    Warning: Circuit count mismatch in inference. Expected {num_circuits_per_eval_used}, got {len(final_circuits_to_save)}"
                        )
                    results_list, job_id, job_data = backend.run(
                        final_circuits_to_save, shots=shots
                    )
                    final_energy, final_energy_se = self._compute_energy_from_results(
                        results_list,
                        formatted_observables,
                        self._require_current_num_qubits(),
                        shots,
                    )
                    total_circuits_run = len(final_circuits_to_save)
                    self._benchmark_qc = final_circuits_to_save
                    self._latest_counts = results_list
                    self._job_id = job_id
                    self._job_data = job_data

                eval_end_time = time.time()
                eval_time = eval_end_time - eval_start_time
                print(f"  Inference complete in {eval_time:.2f}s. Energy = {final_energy:.8f}")
                self._energy_history = [final_energy]
                self._param_history = [optimal_parameters]
                self._eval_times = [eval_time]
                nfev = 1  # One function evaluation for inference
                optimizer_success = True  # No optimization, but evaluation is successful
                optimizer_message = "Inference mode: Parameters loaded from instance data."
                optimization_metrics = {
                    "iterations": 1,
                    "converged": True,
                    "function_evaluations": 1,
                    "evaluation_time_stats": {
                        "min": eval_time,
                        "max": eval_time,
                        "mean": eval_time,
                        "total": eval_time,
                    },
                    "energy_convergence": {
                        "initial": final_energy,
                        "final": final_energy,
                        "improvement": 0.0,
                    },
                }
            except Exception as e:
                print(f"    ERROR during inference evaluation: {e}")
                final_energy = float("nan")
                final_energy_se = None
                nfev = 0
                optimizer_success = False
                optimizer_message = f"Inference mode failed: {e}"
                total_circuits_run = 0
                optimization_metrics = {"error": str(e)}
                final_circuits_to_save = None
        else:
            print("Running in OPTIMIZATION mode ('optimal_parameters' not found).")
            initial_parameters = np.random.default_rng(42).normal(
                loc=0.0, scale=1e-2, size=ansatz.num_parameters
            )
            method = self._optimizer_config.get("method", "COBYLA")
            options = self._optimizer_config.get("options", {"maxiter": 10_000})
            print(f"Starting optimizer ({method}) with {len(initial_parameters)} parameters...")
            print(f"  Optimizer options: {options}")

            opt_start_time = time.time()
            optimizer_result = minimize(
                self._objective_function,
                x0=initial_parameters,
                method=method,
                options=options,
            )
            opt_end_time = time.time()
            opt_duration = opt_end_time - opt_start_time
            print(f"Optimizer finished in {opt_duration:.4f}s")

            final_energy = (
                optimizer_result.fun
                if hasattr(optimizer_result, "fun") and not np.isnan(optimizer_result.fun)
                else float("nan")
            )
            nfev = (
                optimizer_result.nfev
                if hasattr(optimizer_result, "nfev")
                else self._iteration_count
            )  # Fallback if nfev not present
            optimizer_success = (
                optimizer_result.success if hasattr(optimizer_result, "success") else False
            )
            optimizer_success = optimizer_success and not np.isnan(
                final_energy
            )  # Must have valid energy
            optimizer_message = (
                optimizer_result.message
                if hasattr(optimizer_result, "message")
                else "Optimizer message not available."
            )

            if optimizer_success:
                optimal_parameters = optimizer_result.x
                print(f"VQE Result: Success=True, Energy={final_energy:.8f}, NFEV={nfev}")
                bound_ansatz_opt = ansatz.assign_parameters(optimal_parameters)
                if self._exact_simulation:
                    final_circuits_to_save = [bound_ansatz_opt]
                    final_energy_se = 0.0  # Exact
                    total_circuits_run = nfev * num_circuits_per_eval_used  # nfev * 1 for exact
                    self._benchmark_qc = final_circuits_to_save
                else:
                    print("  Re-evaluating at optimal parameters for final SE (shot-based)...")
                    final_circuits_to_save, formatted_observables_opt = self._prepare_circuits(
                        bound_ansatz_opt, hamiltonian
                    )
                    if self._current_backend is None or self._current_shots is None:
                        raise RuntimeError(
                            "Backend or shots not set for final SE calculation in shot-based optimization."
                        )
                    results_list_opt, job_id_opt, job_data_opt = self._current_backend.run(
                        final_circuits_to_save, shots=self._current_shots
                    )
                    # Recalculate energy and SE at this point
                    final_energy_at_opt, final_energy_se_at_opt = self._compute_energy_from_results(
                        results_list_opt,
                        formatted_observables_opt,
                        self._require_current_num_qubits(),
                        self._current_shots,
                    )
                    # We'll use the energy from this final evaluation for consistency
                    print(
                        f"    Optimizer energy: {final_energy:.8f}, Final eval energy: {final_energy_at_opt:.8f} +/- {final_energy_se_at_opt if final_energy_se_at_opt is not None else 'NaN'}"
                    )
                    final_energy = final_energy_at_opt  # Update final_energy
                    final_energy_se = final_energy_se_at_opt
                    total_circuits_run = (nfev * num_circuits_per_eval_used) + len(
                        final_circuits_to_save
                    )
                    self._benchmark_qc = final_circuits_to_save
                    self._latest_counts = results_list_opt
                    self._job_id = job_id_opt
                    self._job_data = job_data_opt
            else:
                print(f"VQE Result: Success=False, Message={optimizer_message}, NFEV={nfev}")
                optimal_parameters = np.full(ansatz.num_parameters, float("nan"))
                final_energy_se = None
                total_circuits_run = nfev * num_circuits_per_eval_used

            initial_energy_hist = None
            if len(self._energy_history) > 0 and not np.isnan(self._energy_history[0]):
                initial_energy_hist = self._energy_history[0]

            eval_times_arr = np.array(self._eval_times) if self._eval_times else np.array([])
            time_stats = {
                "min": np.min(eval_times_arr).item() if eval_times_arr.size > 0 else None,
                "max": np.max(eval_times_arr).item() if eval_times_arr.size > 0 else None,
                "mean": np.mean(eval_times_arr).item() if eval_times_arr.size > 0 else None,
                "total": np.sum(eval_times_arr).item() if eval_times_arr.size > 0 else None,
            }
            optimization_metrics = {
                "iterations": self._iteration_count,
                "converged": optimizer_success,
                "function_evaluations": nfev,
                "optimizer_details": {
                    "method": method,
                    "options": options,
                    "message": optimizer_message,
                    "success": optimizer_success,
                },
                "evaluation_time_stats": time_stats,
                "energy_convergence": {
                    "initial": initial_energy_hist,
                    "final": final_energy,
                    "improvement": (
                        float(initial_energy_hist - final_energy)
                        if initial_energy_hist is not None and not np.isnan(final_energy)
                        else None
                    ),
                    "total_optimization_time": opt_duration,
                },
            }

        algo_output = {
            "final_energy": final_energy,
            "optimal_parameters": optimal_parameters.tolist()
            if optimal_parameters is not None
            else None,
            "nfev": nfev,
            "optimizer_success": optimizer_success,
            "optimizer_message": optimizer_message,
            "final_energy_se": final_energy_se,
            "shots_per_eval": shots_per_eval,
            "num_circuits_per_eval": num_circuits_per_eval_used,
            "mode": "inference" if self._precomputed_parameters is not None else "optimization",
            "simulation_mode": simulation_mode_str.lower().replace(" ", "_"),
            "primitive_version": self._qiskit_primitive_version if self._exact_simulation else None,
            "total_circuits": total_circuits_run,
            "total_shots": total_circuits_run * shots_per_eval,
            "optimization_metrics": optimization_metrics,
            "energy_history": self._energy_history,
            "evaluation_times": self._eval_times,
            "final_circuits": final_circuits_to_save,
        }

        self._current_backend = None
        self._current_shots = None
        self._precomputed_parameters = None
        self._current_exact_estimator = None
        return algo_output

    def compute_merit_figures(
        self, algo_output: dict[str, Any], problem_instance: BenchmarkCase
    ) -> dict[str, Any]:
        def _format_float_or_nan(value: float | None, digits: int) -> str:
            if value is None or np.isnan(value):
                return "NaN"
            return f"{value:.{digits}f}"

        final_energy = self._coerce_optional_float(algo_output.get("final_energy"))
        final_energy_se = self._coerce_optional_float(algo_output.get("final_energy_se"))

        nfev = int(algo_output.get("nfev", 1))
        shots_per_eval = int(algo_output.get("shots_per_eval", 0))
        num_circuits_per_eval = int(algo_output.get("num_circuits_per_eval", 1))
        total_circuits = int(algo_output.get("total_circuits", 0))
        total_shots = int(algo_output.get("total_shots", 0))
        mode = str(algo_output.get("mode", "unknown"))
        sim_mode = str(algo_output.get("simulation_mode", "unknown"))
        prim_ver = algo_output.get("primitive_version")

        print(f"  Calculating metrics for mode: {mode} ({sim_mode})")

        ref_doci = self._coerce_optional_float(problem_instance.data.get("reference_energy_doci"))
        ref_fci = self._coerce_optional_float(problem_instance.data.get("reference_energy_fci"))
        hf_energy = self._coerce_optional_float(problem_instance.data.get("hf_energy"))

        accuracy_wrt_doci = float("nan")
        accuracy_wrt_doci_se = float("nan")
        fraction_correlation_energy = float("nan")
        fraction_correlation_energy_se = float("nan")
        doci_correlation_energy = float("nan")
        vqe_correlation_energy = float("nan")

        #  1. Accuracy with respect to DOCI
        if final_energy is not None and ref_doci is not None:
            accuracy_wrt_doci = abs(final_energy - ref_doci)
            accuracy_wrt_doci_se = final_energy_se if final_energy_se is not None else float("nan")
            print(
                f"    Accuracy (vs DOCI): {accuracy_wrt_doci:.8f} ± {_format_float_or_nan(accuracy_wrt_doci_se, 8)}"
            )
        elif ref_doci is None:
            print(
                "    Warning: Cannot calculate accuracy vs DOCI (reference_energy_doci missing/invalid)."
            )
        elif final_energy is None:
            print("    Warning: Cannot calculate accuracy vs DOCI (final_energy is NaN).")

        #  2. Fraction Correlation Energy
        correlation_tolerance = 1e-9  # Tolerance for checking if correlation energy is zero

        if final_energy is not None and ref_doci is not None and hf_energy is not None:
            doci_correlation_energy = ref_doci - hf_energy
            vqe_correlation_energy = final_energy - hf_energy

            print(f"    HF Energy: {hf_energy:.8f}")
            print(f"    DOCI Correlation Energy (Ec_DOCI): {doci_correlation_energy:.8f}")
            print(
                f"    VQE Correlation Energy (Ec_VQE): {vqe_correlation_energy:.8f} ± {_format_float_or_nan(final_energy_se, 8)}"
            )

            # In shot-based mode this is a noisy ratio estimator rather than a
            # physically bounded fraction, so values outside [0, 1] are possible.
            if abs(doci_correlation_energy) > correlation_tolerance:
                fraction_correlation_energy = (
                    vqe_correlation_energy / doci_correlation_energy
                )  # leaving as 0-1
                if final_energy_se is not None:
                    fraction_correlation_energy_se = (
                        abs(1.0 / doci_correlation_energy) * final_energy_se
                    )
                else:
                    fraction_correlation_energy_se = float("nan")
            else:  # DOCI correlation energy is effectively zero
                if abs(vqe_correlation_energy) <= correlation_tolerance:
                    fraction_correlation_energy = 1.0  # Captured 100% of (near) zero
                    # If both VQE_corr and DOCI_corr are ~0, SE of fraction is 0 if VQE_SE is also ~0
                    fraction_correlation_energy_se = (
                        0.0
                        if (final_energy_se is None or abs(final_energy_se) < correlation_tolerance)
                        else float("nan")
                    )
                else:  # DOCI correlation is zero, VQE is non-zero
                    fraction_correlation_energy = (
                        0.0  # VQE found correlation where DOCI reference had none.
                    )
                    fraction_correlation_energy_se = float("nan")  # Division by zero in SE formula

            fraction_label = "Fraction Correlation Energy Captured"
            fraction_suffix = ""
            if sim_mode == "shot-based_sampling":
                fraction_label = "Correlation Energy Ratio (Ec_VQE / Ec_DOCI)"
                fraction_suffix = " [unbounded under shot noise]"
                
            se_percentage = (
                f"{fraction_correlation_energy_se * 100.0:.2f}"
                if not np.isnan(fraction_correlation_energy_se)
                else "NaN"
            )
            print(f"    {fraction_label}: {fraction_correlation_energy * 100.0:.2f}% ± {se_percentage}%{fraction_suffix}")  

        elif final_energy is None:
            print(
                "    Warning: Cannot calculate Fraction Correlation Energy (final_energy is NaN)."
            )
        elif ref_doci is None:
            print(
                "    Warning: Cannot calculate Fraction Correlation Energy (reference_energy_doci missing/invalid)."
            )
        elif hf_energy is None:
            print(
                "    Warning: Cannot calculate Fraction Correlation Energy (hf_energy missing/invalid)."
            )

        optimization_metrics = algo_output.get("optimization_metrics", {})

        merit_figures = {
            "score": fraction_correlation_energy,
            "score_se": fraction_correlation_energy_se,
            "total_shots": total_shots,
            "total_circuits": total_circuits,
            "final_energy": final_energy if final_energy is not None else float("nan"),
            "final_energy_se": final_energy_se if final_energy_se is not None else float("nan"),
            "reference_doci": ref_doci,
            "reference_fci": ref_fci,
            "hf_energy": hf_energy,
            "doci_correlation_energy": doci_correlation_energy,
            "vqe_correlation_energy": vqe_correlation_energy,
            "accuracy_vs_doci": accuracy_wrt_doci,  # Old score value
            "accuracy_vs_doci_se": accuracy_wrt_doci_se,  # Old score_se value
            "nfev": nfev,
            "num_circuits_per_eval": num_circuits_per_eval,
            "shots_per_eval": shots_per_eval,
            "optimizer_success": algo_output.get("optimizer_success"),
            "optimizer_message": algo_output.get("optimizer_message", ""),
            "mode": mode,
            "simulation_mode": sim_mode,
            "primitive_version": prim_ver,
            "optimization_details": optimization_metrics,
            # Provide QC-runner fields expected by CircuitBenchmarkRunner
            "benchmark_qc_hist": getattr(self, "_latest_counts", []),
            "job_id": getattr(self, "_job_id", "local"),
            "job_data": getattr(self, "_job_data", {}),
        }

        opt_params = algo_output.get("optimal_parameters")
        if opt_params is not None and not (
            isinstance(opt_params, list) and any(np.isnan(p) for p in opt_params)
        ):
            merit_figures["optimal_parameters"] = opt_params
        if "final_circuits" in algo_output:
            merit_figures["final_circuits"] = algo_output["final_circuits"]
        return merit_figures

    def gate_counts(
        self, final_circuits: list[QuantumCircuit] | None, **kwargs: Any
    ) -> tuple[int, int, int, int] | None:
        if not final_circuits or not isinstance(final_circuits, list) or len(final_circuits) == 0:
            print(" [!] Invalid or empty circuit list provided to gate_counts.")
            return None
        circuit = final_circuits[0]
        if not isinstance(circuit, QuantumCircuit):
            print(
                f" [!] First item in final_circuits is not a valid Qiskit QuantumCircuit (type: {type(circuit)})."
            )
            return None

        basis_gates = ["rx", "ry", "rz", "cx"]
        try:
            transpiled_circuit = transpile(circuit, basis_gates=basis_gates, **kwargs)
            dag = circuit_to_dag(transpiled_circuit)
        except DAGCircuitError:
            print(" [!] DAGCircuitError encountered, attempting decomposition first.")
            try:
                transpiled_circuit = transpile(
                    circuit.decompose(), basis_gates=basis_gates, **kwargs
                )
                dag = circuit_to_dag(transpiled_circuit)
            except Exception as e:
                print(f" [!] Error during transpilation/DAG conversion even after decompose: {e}")
                return None
        except Exception as e:
            print(f" [!] Error during transpilation/DAG conversion: {e}")
            return None

        gate_counts_dict = dag.count_ops()
        num_qubits_transpiled = dag.num_qubits()
        one_qubit_gates = 0
        two_qubit_gates = 0
        for gate, count in gate_counts_dict.items():
            if gate in ["rx", "ry", "rz"]:
                one_qubit_gates += count
            elif gate == "cx":
                two_qubit_gates += count

        cx_depth = 0
        if two_qubit_gates > 0:
            try:
                ops_to_remove = [
                    op.name
                    for op in dag.op_nodes()
                    if op.name
                    in [
                        "rx",
                        "ry",
                        "rz",
                        "barrier",
                        "id",
                        "u",
                        "u1",
                        "u2",
                        "u3",
                        "rzx",
                        "rxx",
                        "ryy",
                        "rzz",
                        "s",
                        "sdg",
                        "t",
                        "tdg",
                        "h",
                        "p",
                        "x",
                        "y",
                        "z",
                    ]
                ]
                for op_name in ops_to_remove:
                    dag.remove_all_ops_named(op_name)
                cx_depth = dag.depth()
            except Exception as e:
                print(f" [!] Error calculating CX depth after removing 1Q gates: {e}")
                cx_depth = -1
        print(f" [+] Num qubits:   {num_qubits_transpiled}")
        print(f" [+] Num 1Q gates: {one_qubit_gates} (Transpiled: rx,ry,rz)")
        print(f" [+] Num 2Q gates: {two_qubit_gates} (Transpiled: cx)")
        print(f" [+] CX depth:     {cx_depth} (After 1Q gate removal)")
        return (
            num_qubits_transpiled,
            one_qubit_gates,
            two_qubit_gates,
            cx_depth,
        )

    def get_benchmark_circuits(self, problem_instance: BenchmarkCase) -> list[QuantumCircuit]:
        """
        Get the list of quantum circuits needed for VQE INFERENCE.
        This is required for CircuitBenchmarkRunner's async compatibility.
        """
        self.setup_algo_inputs(problem_instance)

        if self._precomputed_parameters is None:
            raise ValueError(
                "get_benchmark_circuits for VQE is only supported in inference mode "
                "(requires 'optimal_parameters' in the BenchmarkCase data)."
            )

        bound_ansatz = self._require_current_ansatz().assign_parameters(
            self._precomputed_parameters
        )
        qc_list, _ = self._prepare_circuits(bound_ansatz, self._require_current_hamiltonian())
        return qc_list

    def merit_figures_from_measurements(
        self,
        measurements: MeasurementBatch,
        problem_instance: BenchmarkCase,
    ) -> dict[str, Any]:
        """
        Compute merit figures from retrieved measurement results after an async job.
        Mirrors the synchronous path by reusing `compute_merit_figures`.
        """
        print(
            f"  Calculating metrics from async results for instance: {problem_instance.instance_name}"
        )

        # Hydrate Hamiltonian/Ansatz and get precomputed params
        self.setup_algo_inputs(problem_instance)
        if self._precomputed_parameters is None:
            raise ValueError(
                "Async result processing for VQE requires 'optimal_parameters' in the BenchmarkCase data."
            )

        # Build formatted observables for the energy estimator
        bound_ansatz = self._require_current_ansatz().assign_parameters(
            self._precomputed_parameters
        )
        _, formatted_observables = self._prepare_circuits(
            bound_ansatz,
            self._require_current_hamiltonian(),
        )

        # Counts-based backends return integer histograms, while some async paths
        # can return normalized probabilities. Detect and handle both cases.
        _DEFAULT_SHOTS_PER_QC = 10_000
        first_hist = measurements[0] if (measurements and measurements[0]) else {}
        inferred_total = sum(float(v) for v in first_hist.values())
        has_float_values = any(isinstance(v, (float, np.floating)) for v in first_hist.values())
        has_non_integral_values = any(
            not isinstance(v, Integral) and not float(v).is_integer() for v in first_hist.values()
        )
        looks_normalized = np.isclose(inferred_total, 1.0, atol=0.01)

        if has_non_integral_values:
            if not looks_normalized:
                raise ValueError("Non-integer async histograms must be normalized probabilities.")
            shots = int(problem_instance.data.get("shots_per_qc", _DEFAULT_SHOTS_PER_QC))
            warnings.warn(
                f"Detected normalized probability distributions; using shots_per_qc={shots}.",
                stacklevel=2,
            )
        elif first_hist and looks_normalized and has_float_values:
            shots = int(problem_instance.data.get("shots_per_qc", _DEFAULT_SHOTS_PER_QC))
            warnings.warn(
                "Ambiguous async histogram payload with unit total and float-valued integral "
                f"entries; interpreting as normalized probabilities using shots_per_qc={shots}. "
                "Pass integer counts to avoid ambiguity.",
                stacklevel=2,
            )
        else:
            shots = int(inferred_total)

        if shots > 0:
            final_energy, final_energy_se = self._compute_energy_from_results(
                measurements,
                formatted_observables,
                self._require_current_num_qubits(),
                shots,
            )
        else:
            final_energy, final_energy_se = float("nan"), float("nan")

        # Reuse the synchronous metrics logic
        mock_algo_output = {
            "final_energy": final_energy,
            "final_energy_se": final_energy_se,
            "total_shots": shots * len(measurements),
            "total_circuits": len(measurements),
            "mode": "inference_async",
            "simulation_mode": "shot-based_sampling",
        }
        return self.compute_merit_figures(mock_algo_output, problem_instance)
