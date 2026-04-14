"""
Helpers and interfaces for circuit-based benchmark runners.

Copyright (c) 2025 IonQ, Inc. All rights reserved.
Licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0
International (CC BY-NC-ND 4.0) License.
For details, visit: https://creativecommons.org/licenses/by-nc-nd/4.0/
"""

import json
import random
from abc import abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, cast

import numpy as np
import pandas as pd
from qiskit import QuantumCircuit, transpile

from apps_benchmark.core.backend import (
    AbstractAsyncBackend,
    AbstractBackend,
    JobData,
    MeasurementBatch,
)
from apps_benchmark.core.benchmark import (
    AbstractAlgoRunner,
    BenchmarkHeader,
    BenchmarkSubmissionRecord,
)
from apps_benchmark.primitives.benchmark_case import BenchmarkCase


@dataclass
class BaselineScore(BenchmarkHeader):
    mean: float
    std: float
    shots_per_draw: int
    num_draws: int


@dataclass
class CircuitStats(BenchmarkHeader):
    num_circuits: int
    num_1q_gates: list[int]
    num_2q_gates: list[int]


@dataclass(kw_only=True)
class QCBenchmarkSubmissionRecord(BenchmarkSubmissionRecord):
    job_id: str
    job_data: JobData
    measurements: MeasurementBatch = field(default_factory=list)

    def __post_init__(self) -> None:
        try:
            json.dumps(self.job_data)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"job_data must be JSON-serializable: {exc}") from exc


class CircuitBenchmarkRunner(AbstractAlgoRunner):
    """
    Abstract class for implementing quantum circuit benchmarks consisting of
    evaluating a list of quantum circuits and computing scores based on the
    observed measurement histograms.

    This class inherits from :class:`.AbstractAlgoRunner` and implements all
    its required methods, except ``name``. In exchange this class requires
    implementing its more concrete ``get_benchmark_circuits`` method, which
    should return a list of ``QuantumCircuit``s that need to be executed, and
    its ``merit_figures_from_measurements`` method, which computes the
    benchmark's merit figures from the output histograms.
    """

    @abstractmethod
    def get_benchmark_circuits(self, benchmark_case: BenchmarkCase) -> list[QuantumCircuit]:
        """
        Get the list of quantum circuits that should be executed.
        """
        pass

    @abstractmethod
    def merit_figures_from_measurements(
        self,
        measurements: MeasurementBatch,
        benchmark_case: BenchmarkCase,
    ) -> dict[str, Any]:
        """
        Compute relevant merit figures from the given list of ``measurements``
        of the ``self.get_benchmark_circuits`` list.

        Note that 'benchmark_qc_hist', 'job_id', and 'job_data' are reserved
        keys and cannot be used in the merit figures dictionary.
        """
        pass

    def est_rnd_baseline_score(
        self, benchmark_case: BenchmarkCase, shots: int, num_draws: int = 1_000
    ) -> BaselineScore:
        """
        Estimate the score obtained that would be obtained by this solution
        algorithm on the given ``benchmark_case`` if the measurements for
        each benchmark circuit were drawn uniformly at random from the set of
        all possible bit-strings on ``benchmark_case.num_qubits`` qubits.
        """
        num_circuits = len(self.get_benchmark_circuits(benchmark_case))
        nq = benchmark_case.num_qubits
        baseline_scores = np.zeros(num_draws)
        for draw in np.arange(num_draws):
            # Use numpy for speed if possible
            measurements: MeasurementBatch
            if nq < 64:
                measurements = [
                    {f"{x:0{nq}b}": 1 for x in np.random.randint(2**nq, size=shots)}
                    for _ in range(num_circuits)
                ]
            else:
                measurements = [
                    {f"{random.randint(0, 2**nq - 1):0{nq}b}": 1 for _ in range(shots)}
                    for _ in range(num_circuits)
                ]

            mf = self.merit_figures_from_measurements(measurements, benchmark_case)
            baseline_scores[draw] = float(mf["score"])

        header = asdict(benchmark_case)
        [header.pop(key, None) for key in ["data", "solution_algorithms", "open_solution_algorithms"]]
        return BaselineScore(
            **header,
            solution_algorithm=self.name(),
            mean=baseline_scores.mean(),
            std=baseline_scores.std(),
            shots_per_draw=shots,
            num_draws=num_draws,
        )

    def get_circuit_stats(
        self,
        benchmark_case: BenchmarkCase,
    ) -> CircuitStats:
        """
        Count the number of gates in each benchmark circuit.

        The count is provided once the circuits are transpiled (using
        ``qiskit``'s native transpiler with heavy optimization) into the
        ``cx, u`` universal gate set.
        """
        stats = []
        circuits = self.get_benchmark_circuits(benchmark_case)
        for qc in circuits:
            transpiled = transpile(qc, basis_gates=["cx", "u"], optimization_level=3)
            ops = transpiled.count_ops()
            stats.append((ops["u"], ops["cx"]))

        sqg, tqg = zip(*stats, strict=True) if stats else ((), ())
        instance_data = asdict(benchmark_case)
        instance_data.pop("data")
        instance_data.pop("solution_algorithms")
        instance_data.pop("open_solution_algorithms", None)
        return CircuitStats(
            **instance_data,
            solution_algorithm=self.name(),
            num_circuits=len(circuits),
            num_1q_gates=list(sqg),
            num_2q_gates=list(tqg),
        )

    def submit_benchmark_circuits(
        self, benchmark_case: BenchmarkCase, backend: AbstractAsyncBackend, shots: int
    ) -> QCBenchmarkSubmissionRecord:
        """
        Submit benchmark circuits for execution on the given ``backend``.
        """
        tic = pd.Timestamp.now(tz="UTC")
        circuits = self.get_benchmark_circuits(benchmark_case)
        job_name = self._get_benchmark_job_name(benchmark_case.instance_name, shots)

        job_id, job_data = backend.submit(circuits, shots=shots, job_name=job_name)
        instance_id = benchmark_case.instance_id
        if instance_id is None:
            raise ValueError("BenchmarkCase.instance_id must be set before submitting circuits.")
        return QCBenchmarkSubmissionRecord(
            benchmark_category=self.benchmark_category,
            problem_type=benchmark_case.problem_type,
            instance_name=benchmark_case.instance_name,
            instance_id=instance_id,
            solution_algorithm=self.name(),
            num_qubits=benchmark_case.num_qubits,
            backend=backend.name(),
            shots_per_qc=shots,
            total_shots=shots * len(circuits),
            start_time=tic,
            end_time=pd.NaT,
            time_to_soln=pd.NaT,
            adjusted_tts=pd.NaT,
            last_retrieval=tic,
            status="submitted",
            score=float("nan"),
            problem_specific_data={},
            job_data=job_data,
            job_id=job_id,
        )

    def _get_benchmark_job_name(self, instance_name: str, shots: int) -> str:
        job_name = f"Benchmarking algorithm {self.name().upper()} on"
        job_name += f"problem instance {instance_name} using {shots} shots"
        return job_name

    def setup_algo_inputs(self, benchmark_case: BenchmarkCase) -> tuple[list[QuantumCircuit], str]:
        self._benchmark_qc = self.get_benchmark_circuits(benchmark_case)
        return (self._benchmark_qc, benchmark_case.instance_name)

    def execute_benchmark_algo(
        self,
        algo_inputs: tuple[Any, ...],
        backend: AbstractBackend,
        shots: int,
        **kwargs: Any,
    ) -> Any:
        circuits, instance_name = algo_inputs
        job_name = self._get_benchmark_job_name(instance_name, shots)
        ret = backend.run(circuits, shots=shots, job_name=job_name)
        self._shots = shots  # record number of shots
        return ret

    def compute_merit_figures(
        self,
        algo_output: Any,
        benchmark_case: BenchmarkCase,
    ) -> dict[str, Any]:
        measurements, job_id, job_data = algo_output
        mf = dict(self.merit_figures_from_measurements(measurements, benchmark_case))
        mf["total_shots"] = self._shots * len(measurements)
        mf["benchmark_qc_hist"] = measurements
        mf["job_id"] = job_id
        mf["job_data"] = job_data
        return mf

    def run_benchmark(
        self,
        benchmark_case: BenchmarkCase,
        backend: AbstractBackend,
        shots: int,
        **kwargs: Any,
    ) -> QCBenchmarkSubmissionRecord:
        """
        Re-implements the super class's method to include quantum circuit and
        result histogram caching.
        """
        ret = super().run_benchmark(benchmark_case, backend, shots, **kwargs)

        # Extract qc measurements and job ID
        measurements = cast(MeasurementBatch, ret.problem_specific_data.pop("benchmark_qc_hist"))
        job_id = cast(str, ret.problem_specific_data.pop("job_id"))
        job_data = cast(JobData, ret.problem_specific_data.pop("job_data"))
        return QCBenchmarkSubmissionRecord(
            **asdict(ret), measurements=measurements, job_id=job_id, job_data=job_data
        )
