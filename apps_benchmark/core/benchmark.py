"""
Abstract base class for benchmark algorithm runners.

This module defines the interface that all benchmark algorithms must implement.

Copyright (c) 2025 IonQ, Inc. All rights reserved.
Licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0
International (CC BY-NC-ND 4.0) License.
For details, visit: https://creativecommons.org/licenses/by-nc-nd/4.0/
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pandas as pd
from pandas.api.typing import NaTType

if TYPE_CHECKING:
    from apps_benchmark.core.backend import AbstractBackend
    from apps_benchmark.primitives.benchmark_case import BenchmarkCase


@dataclass
class BenchmarkHeader:
    # Identification
    benchmark_category: str  # e.g., "chemistry"
    problem_type: str  # e.g., "hydrogen_lattice_vqe"
    instance_name: str  # e.g., "h002_chain_0_75"
    instance_id: str  # e.g., "610cfb55"
    solution_algorithm: str  # e.g., "vqe_puccd"
    num_qubits: int  # Number of qubits used


@dataclass
class BenchmarkSubmissionRecord(BenchmarkHeader):
    """
    Results from a single benchmark execution.

    This dataclass captures all information about a benchmark run,
    including timing, scores, and problem-specific data.
    """

    # Execution details
    backend: str  # Backend name
    shots_per_qc: int  # Shots per circuit
    total_shots: int  # Total shots across all circuits

    # Timing
    start_time: pd.Timestamp  # When execution started (UTC)
    end_time: pd.Timestamp | NaTType  # When execution ended (UTC)
    time_to_soln: pd.Timedelta | NaTType  # Total execution time
    adjusted_tts: pd.Timedelta | NaTType  # Adjusted time-to-solution
    last_retrieval: pd.Timestamp  # Time the record was last updated

    # Results
    status: str  # "submitted", "done", "failed"
    score: float  # Benchmark score (problem-specific meaning)

    # Additional data
    problem_specific_data: dict[str, Any] = field(default_factory=dict)


class AbstractAlgoRunner(ABC):
    """
    Abstract base class for benchmark algorithm runners.

    Each algorithm (e.g., VQE, QAOA) should implement this interface.
    The run_benchmark() method orchestrates the full execution flow.
    """

    @abstractmethod
    def name(self) -> str:
        """
        Return the algorithm name.

        Returns:
            str: Algorithm identifier (e.g., 'vqe_puccd', 'qaoa', 'grover')

        Example:
            >>> runner.name()
            'vqe_puccd'
        """
        pass

    @abstractmethod
    def setup_algo_inputs(self, benchmark_case: BenchmarkCase) -> tuple[Any, ...]:
        """
        Parse problem instance and create algorithm inputs.

        This method converts the generic BenchmarkCase into
        algorithm-specific inputs (e.g., Hamiltonian, ansatz for VQE).

        Args:
            benchmark_case: Problem to solve

        Returns:
            tuple[Any, ...]: Algorithm-specific inputs

        Example (VQE):
            >>> benchmark_case = BenchmarkCase.load_from_database("h2.json")
            >>> hamiltonian, ansatz, initial_params = runner.setup_algo_inputs(benchmark_case)
            >>> isinstance(hamiltonian, SparsePauliOp)
            True

        Note:
            Return type is flexible - return whatever your algorithm needs.
            Common patterns:
            - VQE: (hamiltonian, ansatz, initial_parameters)
            - QAOA: (cost_hamiltonian, mixer_hamiltonian, p_layers)
            - Grover: (oracle, target_states)
        """
        pass

    @abstractmethod
    def execute_benchmark_algo(
        self,
        algo_inputs: tuple[Any, ...],
        backend: AbstractBackend,
        shots: int,
        **kwargs: Any,
    ) -> Any:
        """
        Execute the algorithm using given inputs and backend.

        This is where the actual quantum algorithm runs. Use the backend
        to execute circuits and return the raw algorithm output.

        Args:
            algo_inputs: Output from setup_algo_inputs()
            backend: Backend to execute circuits on
            shots: Number of shots per circuit
            **kwargs: Additional algorithm-specific parameters

        Returns:
            Any: Algorithm output (flexible type)

        Example (VQE):
            >>> hamiltonian, ansatz, init_params = algo_inputs
            >>> result = runner.execute_benchmark_algo(algo_inputs, backend, shots=1000)
            >>> result  # Could be optimized energy, final parameters, etc.
            {'energy': -1.137, 'params': [0.114], 'num_iterations': 10}

        Note:
            - Use backend.run() or backend.submit() to execute circuits
            - Handle errors from backend (will propagate up)
            - Return whatever compute_merit_figures() needs
        """
        pass

    @abstractmethod
    def compute_merit_figures(
        self, algo_output: Any, benchmark_case: BenchmarkCase
    ) -> dict[str, Any]:
        """
        Compute benchmark scores and metrics from algorithm output.

        Args:
            algo_output: Output from execute_benchmark_algo()
            benchmark_case: Original problem instance

        Returns:
            dict[str, Any]: Metrics dictionary. MUST contain:
                - 'total_shots' (int): Total shots used
                - 'score' (float): Benchmark score
                Other keys are stored in problem_specific_data.

        Example (VQE):
            >>> merit = runner.compute_merit_figures(algo_output, benchmark_case)
            >>> merit
            {
                'total_shots': 10000,
                'score': 0.998,  # e.g., accuracy metric
                'final_energy': -1.137,
                'energy_error': 0.0001,
                'num_iterations': 10
            }

        Note:
            - 'score' meaning is problem-specific (accuracy, approximation ratio, etc.)
            - All keys except 'total_shots' and 'score' go into problem_specific_data
        """
        pass

    @property
    def benchmark_category(self) -> str:
        """
        Get the benchmark category from module path.

        Auto-derived from __module__. For example, if this runner is in:
            apps_benchmark.benchmarks.chemistry.algorithms.vqe_runner
        Then benchmark_category returns: "chemistry"

        Returns:
            str: Category name

        Raises:
            ValueError: If module path has fewer than 3 parts

        Note:
            You typically don't need to override this.
        """
        parts = self.__module__.split(".")
        if len(parts) < 3:
            raise ValueError(
                f"Cannot derive benchmark_category from module '{self.__module__}'. "
                f"Expected format: '*.*.category.*.runner_name' but got {len(parts)} parts. "
                f"Module path must have at least 3 dot-separated components."
            )
        return parts[-3]

    def run_benchmark(
        self,
        benchmark_case: BenchmarkCase,
        backend: "AbstractBackend",
        shots: int,
        **kwargs: Any,
    ) -> BenchmarkSubmissionRecord:
        """
        Orchestrate full benchmark execution.

        This method is provided by the base class and orchestrates:
        1. setup_algo_inputs()
        2. execute_benchmark_algo()
        3. compute_merit_figures()
        4. Package results into BenchmarkSubmissionRecord

        You do not need to override this method.

        Args:
            benchmark_case: Problem to solve
            backend: Backend to execute on
            shots: Number of shots
            **kwargs: Additional algorithm parameters

        Returns:
            BenchmarkSubmissionRecord: Complete benchmark results

        Example:
            >>> record = runner.run_benchmark(benchmark_case, backend, shots=1000)
            >>> record.status
            'done'
            >>> record.score
            0.998
        """
        start_time = pd.Timestamp.now(tz="UTC")

        algo_inputs = self.setup_algo_inputs(benchmark_case)
        algo_output = self.execute_benchmark_algo(algo_inputs, backend, shots, **kwargs)
        merit_figures = self.compute_merit_figures(algo_output, benchmark_case)

        end_time = pd.Timestamp.now(tz="UTC")

        problem_specific_data = {
            k: v for k, v in merit_figures.items() if k not in ["total_shots", "score"]
        }
        instance_id = benchmark_case.instance_id
        if instance_id is None:
            raise ValueError("BenchmarkCase.instance_id must be set before running a benchmark.")

        return BenchmarkSubmissionRecord(
            benchmark_category=self.benchmark_category,
            problem_type=benchmark_case.problem_type,
            instance_name=benchmark_case.instance_name,
            instance_id=instance_id,
            solution_algorithm=self.name(),
            num_qubits=benchmark_case.num_qubits,
            backend=backend.name(),
            shots_per_qc=shots,
            total_shots=merit_figures["total_shots"],
            start_time=start_time,
            end_time=end_time,
            time_to_soln=end_time - start_time,
            adjusted_tts=end_time - start_time,
            last_retrieval=end_time,
            status="done",
            score=merit_figures["score"],
            problem_specific_data=problem_specific_data,
        )
