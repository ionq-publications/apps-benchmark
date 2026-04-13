"""
QCBM copula runner for optimization benchmarks.

Copyright (c) 2025 IonQ, Inc. All rights reserved.
Licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0
International (CC BY-NC-ND 4.0) License.
For details, visit: https://creativecommons.org/licenses/by-nc-nd/4.0/
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from scipy.stats import t

from apps_benchmark.core.backend import MeasurementBatch
from apps_benchmark.core.qc_benchmark_runner import CircuitBenchmarkRunner
from apps_benchmark.primitives.benchmark_case import BenchmarkCase

_ANSATZ_1 = "ansatz_1"
_ANSATZ_2 = "ansatz_2"
_ALPHA = 0.95
_COPULA_DECODER_SEED = 0


class QcbmCopulaRunner(CircuitBenchmarkRunner):
    """
    Sample from a trained QCBM copula circuit and score its VaR agreement.

    The original legacy implementation re-downloaded market data at scoring
    time. The benchmark cases in this repo already store the reference VaR, so
    this port keeps scoring deterministic and offline.
    """

    def name(self) -> str:
        """Return the algorithm name."""
        return "qcbm_copula"

    def get_benchmark_circuits(self, benchmark_case: BenchmarkCase) -> list[QuantumCircuit]:
        """Construct and bind the trained QCBM circuit for the given case."""
        case_data = benchmark_case.data
        num_qubits = benchmark_case.num_qubits
        num_layers = _get_num_layers(case_data)
        num_variables = _get_num_variables(case_data)
        theta = np.asarray(_get_parameters(case_data), dtype=float)
        ansatz_family = _get_ansatz_family(benchmark_case)

        if ansatz_family == _ANSATZ_2:
            bits_per_variable = _get_bits_per_variable(benchmark_case)
            circuit = build_ansatz_2_circuit(num_variables, bits_per_variable, num_layers)
        else:
            circuit = build_ansatz_1_circuit(num_qubits, num_layers)

        circuit.assign_parameters(theta, inplace=True)
        return [circuit]

    def merit_figures_from_measurements(
        self,
        measurements: MeasurementBatch,
        benchmark_case: BenchmarkCase,
    ) -> dict[str, Any]:
        """
        Compute the VaR ratio between generated samples and stored reference VaR.
        """
        case_data = benchmark_case.data
        ansatz_family = _get_ansatz_family(benchmark_case)
        counts = _histogram_to_counts(measurements[0], self._shots)
        num_qubits = benchmark_case.num_qubits
        num_variables = _get_num_variables(case_data)
        bits_per_variable = _get_bits_per_variable(benchmark_case)
        alpha = _get_var_confidence_level(case_data)

        copula_samples = get_copula_samples(
            counts,
            num_qubits,
            bits_per_variable,
            seed=_COPULA_DECODER_SEED,
        )
        generated_samples = convert_to_real_space(
            copula_samples,
            np.asarray(_get_t_fit_parameters(case_data), dtype=float),
            np.asarray(_get_means(case_data), dtype=float),
            np.asarray(_get_standard_deviations(case_data), dtype=float),
        )

        generated_losses = -np.sum(generated_samples, axis=1) / num_variables
        generated_var = float(np.quantile(generated_losses, alpha))
        reference_var = _get_reference_var(case_data)

        if reference_var <= 0.0 or generated_var <= 0.0:
            raise ValueError(
                f"VaR values must be positive for copula scoring, got "
                f"reference={reference_var} generated={generated_var}."
            )

        var_ratio = generated_var / reference_var
        score = min(var_ratio, 1.0 / var_ratio)
        return {
            "score": score,
            "VaR_ratio": var_ratio,
            "alpha": alpha,
            "ansatz_family": ansatz_family,
            "reference_VaR": reference_var,
            "reference_VaR_95": reference_var,
            "generated_VaR": generated_var,
            "generated_VaR_95": generated_var,
        }


def build_ansatz_1_circuit(num_qubits: int, num_layers: int) -> QuantumCircuit:
    """
    Build ansatz_1: layers of Rx, Rz, and all-to-all Rxx entanglers.
    """
    circuit = QuantumCircuit(num_qubits)
    num_parameters = 2 * num_qubits * (num_layers + 1)
    num_parameters += num_layers * num_qubits * (num_qubits - 1) // 2
    theta = ParameterVector("theta", num_parameters)

    for qubit in range(num_qubits):
        circuit.rx(theta[qubit], qubit)
    for qubit in range(num_qubits):
        circuit.rz(theta[num_qubits + qubit], qubit)
    circuit.barrier()

    parameter_index = 2 * num_qubits
    for _ in range(num_layers):
        for control in range(num_qubits - 1):
            for target in range(control + 1, num_qubits):
                circuit.rxx(theta[parameter_index], control, target)
                parameter_index += 1
        circuit.barrier()
        for qubit in range(num_qubits):
            circuit.rz(theta[parameter_index], qubit)
            parameter_index += 1
        for qubit in range(num_qubits):
            circuit.rx(theta[parameter_index], qubit)
            parameter_index += 1
        circuit.barrier()

    return circuit


def build_ansatz_2_circuit(
    num_variables: int,
    bits_per_variable: int,
    num_layers: int,
) -> QuantumCircuit:
    """
    Build ansatz_2 from the reference implementation.

    The structured copula ansatz starts from a GHZ-style register entanglement
    pattern, then applies layers of Rz, Rx, and within-register adjacent Rzz
    couplings. The adjacent Rzz pattern matches the legacy source exactly.
    """
    num_qubits = bits_per_variable * num_variables
    circuit = QuantumCircuit(num_qubits)
    num_parameters = num_layers * num_variables * (3 * bits_per_variable - 1)
    theta = ParameterVector("theta", num_parameters)

    for qubit in range(bits_per_variable):
        circuit.h(qubit)

    for variable in range(num_variables - 1):
        for qubit in range(bits_per_variable):
            circuit.cx(qubit, qubit + (variable + 1) * bits_per_variable)

    circuit.barrier()
    for layer in range(num_layers):
        offset = layer * num_variables * (3 * bits_per_variable - 1)

        for qubit in range(num_qubits):
            circuit.rz(theta[offset + qubit], qubit)
        for qubit in range(num_qubits):
            circuit.rx(theta[offset + num_qubits + qubit], qubit)

        circuit.barrier()
        parameter_index = offset + 2 * num_qubits
        for variable in range(num_variables):
            start = variable * bits_per_variable
            for qubit in range(bits_per_variable - 1):
                circuit.rzz(theta[parameter_index], start + qubit, start + qubit + 1)
                parameter_index += 1
        circuit.barrier()

    return circuit


def build_dense_qcbm_circuit(num_qubits: int, num_layers: int) -> QuantumCircuit:
    """
    Backwards-compatible alias for the original all-to-all QCBM ansatz name.
    """
    return build_ansatz_1_circuit(num_qubits, num_layers)


def build_zx_qcbm_circuit(
    num_variables: int,
    bits_per_variable: int,
    num_layers: int,
) -> QuantumCircuit:
    """
    Backwards-compatible alias for the structured copula ansatz.
    """
    return build_ansatz_2_circuit(num_variables, bits_per_variable, num_layers)


def get_copula_samples(
    histogram: Mapping[str, int],
    num_qubits: int,
    bits_per_variable: int,
    *,
    seed: int,
) -> np.ndarray:
    """
    Decode bitstring samples into copula-space samples in ``[0, 1)``.
    """
    bit_rows: list[list[int]] = []
    counts: list[int] = []
    for bitstring, count in histogram.items():
        normalized = bitstring.replace(" ", "")
        normalized = normalized[-num_qubits:].rjust(num_qubits, "0")
        bit_rows.append([1 if bit == "1" else 0 for bit in normalized])
        counts.append(int(count))

    if not bit_rows:
        raise ValueError("Cannot decode copula samples from an empty histogram.")

    expanded_bits = np.repeat(np.asarray(bit_rows, dtype=np.uint8), counts, axis=0)
    rng = np.random.default_rng(seed)
    rng.shuffle(expanded_bits, axis=0)

    num_variables = num_qubits // bits_per_variable
    copula_samples = np.empty((expanded_bits.shape[0], num_variables), dtype=float)
    weights = 2.0 ** -np.arange(1, bits_per_variable + 1, dtype=float)
    bucket_width = 2.0**bits_per_variable

    for variable in range(num_variables):
        start = variable * bits_per_variable
        stop = start + bits_per_variable
        fractions = expanded_bits[:, start:stop] @ weights
        fractions += rng.random(expanded_bits.shape[0]) / bucket_width
        copula_samples[:, variable] = np.clip(
            fractions,
            np.finfo(float).eps,
            1.0 - np.finfo(float).eps,
        )

    return copula_samples


def convert_to_real_space(
    copula_samples: np.ndarray,
    t_fits: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
) -> np.ndarray:
    """
    Map copula-space samples back into standardized real-space returns.
    """
    num_variables = copula_samples.shape[1]
    real_space = np.empty_like(copula_samples, dtype=float)
    for variable in range(num_variables):
        degrees, location, scale = (float(value) for value in t_fits[variable])
        real_space[:, variable] = t.ppf(
            copula_samples[:, variable],
            degrees,
            location,
            scale,
        )

    return real_space * stds + means


def _get_ansatz_family(benchmark_case: BenchmarkCase) -> str:
    nested_ansatz = benchmark_case.data.get("ansatz")
    if isinstance(nested_ansatz, Mapping):
        family = str(nested_ansatz.get("family", "")).strip().lower()
        if family in {_ANSATZ_1, _ANSATZ_2}:
            return family

    legacy_ansatz = str(benchmark_case.data.get("anstaz", "")).strip().upper()
    if legacy_ansatz == "ZX" or "NEW_ANSATZ" in benchmark_case.instance_name.upper():
        return _ANSATZ_2

    return _ANSATZ_1


def _get_num_layers(case_data: Mapping[str, Any]) -> int:
    trained_model = case_data.get("trained_model")
    if isinstance(trained_model, Mapping) and "num_layers" in trained_model:
        return int(trained_model["num_layers"])
    return int(case_data["num_layers"])


def _get_num_variables(case_data: Mapping[str, Any]) -> int:
    portfolio = case_data.get("portfolio")
    if isinstance(portfolio, Mapping) and "num_variables" in portfolio:
        return int(portfolio["num_variables"])
    return int(case_data["num_variable"])


def _get_bits_per_variable(benchmark_case: BenchmarkCase) -> int:
    portfolio = benchmark_case.data.get("portfolio")
    if isinstance(portfolio, Mapping) and "qubits_per_variable" in portfolio:
        return int(portfolio["qubits_per_variable"])
    return _bits_per_variable(
        benchmark_case.num_qubits,
        _get_num_variables(benchmark_case.data),
    )


def _get_parameters(case_data: Mapping[str, Any]) -> list[float]:
    trained_model = case_data.get("trained_model")
    if isinstance(trained_model, Mapping) and "parameters" in trained_model:
        return list(trained_model["parameters"])
    return list(case_data["theta"])


def _get_t_fit_parameters(case_data: Mapping[str, Any]) -> list[list[float]]:
    marginals = case_data.get("marginals")
    if isinstance(marginals, Mapping) and "fit_parameters" in marginals:
        return list(marginals["fit_parameters"])
    return list(case_data["t-fits"])


def _get_means(case_data: Mapping[str, Any]) -> list[float]:
    marginals = case_data.get("marginals")
    if isinstance(marginals, Mapping) and "means" in marginals:
        return list(marginals["means"])
    return list(case_data["means"])


def _get_standard_deviations(case_data: Mapping[str, Any]) -> list[float]:
    marginals = case_data.get("marginals")
    if isinstance(marginals, Mapping) and "standard_deviations" in marginals:
        return list(marginals["standard_deviations"])
    return list(case_data["std"])


def _get_var_confidence_level(case_data: Mapping[str, Any]) -> float:
    benchmark_spec = case_data.get("benchmark_spec")
    if isinstance(benchmark_spec, Mapping) and "var_confidence_level" in benchmark_spec:
        return float(benchmark_spec["var_confidence_level"])
    return _ALPHA


def _bits_per_variable(num_qubits: int, num_variables: int) -> int:
    if num_variables <= 0 or num_qubits % num_variables != 0:
        raise ValueError(
            f"num_qubits={num_qubits} must be divisible by num_variable={num_variables}."
        )
    return num_qubits // num_variables


def _get_reference_var(case_data: Mapping[str, Any]) -> float:
    """
    Retrieve the stored reference VaR from the benchmark case payload.
    """
    nested_reference_metrics = case_data.get("reference_metrics")
    if isinstance(nested_reference_metrics, Mapping):
        reference_var = float(nested_reference_metrics.get("reference_var_95", 0.0))
        if reference_var > 0.0:
            return reference_var

    legacy_reference_metrics = case_data.get("data")
    if isinstance(legacy_reference_metrics, list) and legacy_reference_metrics:
        reference_var = float(legacy_reference_metrics[0])
        if reference_var > 0.0:
            return reference_var

    qcbm_metrics = case_data.get("qcbm")
    ratios = case_data.get("ratios")
    if (
        isinstance(qcbm_metrics, list)
        and qcbm_metrics
        and isinstance(ratios, list)
        and ratios
        and float(ratios[0]) != 0.0
    ):
        return float(qcbm_metrics[0]) / float(ratios[0])

    raise ValueError(
        "Copula benchmark case is missing a usable reference VaR in "
        "data['data'][0] or the equivalent qcbm/ratios fields."
    )


def _histogram_to_counts(histogram: Mapping[str, float | int], shots: int) -> dict[str, int]:
    """
    Convert backend histograms into integer counts with an exact total ``shots``.
    """
    if not histogram:
        raise ValueError("Expected a non-empty measurement histogram.")

    keys = list(histogram.keys())
    values = np.asarray([float(value) for value in histogram.values()], dtype=float)
    if np.any(values < 0.0):
        raise ValueError("Measurement histograms must not contain negative counts.")

    if np.all(np.isclose(values, np.round(values))) and np.isclose(values.sum(), shots):
        return {
            key: int(round(value))
            for key, value in zip(keys, values, strict=True)
            if int(round(value)) > 0
        }

    total_weight = float(values.sum())
    if total_weight <= 0.0:
        raise ValueError("Measurement histograms must have positive total weight.")

    scaled = values * shots / total_weight
    counts = np.floor(scaled).astype(int)
    remainder = shots - int(counts.sum())
    if remainder > 0:
        remainders = scaled - counts
        order = np.argsort(-remainders, kind="stable")
        counts[order[:remainder]] += 1

    return {key: int(count) for key, count in zip(keys, counts, strict=True) if int(count) > 0}
