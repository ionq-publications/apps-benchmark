"""
Interactive benchmark orchestration.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

import pandas as pd

from apps_benchmark.core.backend import AbstractAsyncBackend, AbstractBackend, JobStatus
from apps_benchmark.core.benchmark import BenchmarkSubmissionRecord
from apps_benchmark.core.execution import (
    classify_solution_algorithms,
    find_benchmark_case_by_uuid,
    get_available_runner_names,
    get_shots_for_case,
    iter_benchmark_case_entries,
    load_backend,
    load_benchmark_case_entry,
    load_runner,
    resolve_shots,
    select_solution_algorithm,
)
from apps_benchmark.core.qc_benchmark_runner import (
    CircuitBenchmarkRunner,
    CircuitStats,
    QCBenchmarkSubmissionRecord,
)
from apps_benchmark.errors import (
    BackendValidationError,
    BenchmarkCaseError,
    BenchmarkError,
    BenchmarkValidationError,
)
from apps_benchmark.primitives.benchmark_case import BenchmarkCase

BENCHMARK_CASE_COLUMNS = [
    "benchmark_category",
    "problem_type",
    "instance_name",
    "instance_id",
    "num_qubits",
    "solution_algorithms",
    "open_solution_algorithms",
    "all_solutions_open",
    "builtin",
    "file",
]

ALGORITHM_COLUMNS = [
    "benchmark_category",
    "algorithm",
    "status",
    "builtin",
    "case_count",
]

SUBMISSION_COLUMNS = [field.name for field in fields(QCBenchmarkSubmissionRecord)]
RESULT_COLUMNS = [field.name for field in fields(BenchmarkSubmissionRecord)]
STATS_COLUMNS = [field.name for field in fields(CircuitStats)]


class Benchmarker:
    def __init__(
        self,
        *,
        include_builtin: bool = True,
        include_diy: bool = True,
    ) -> None:
        self.include_builtin = include_builtin
        self.include_diy = include_diy

    def get_benchmark_cases(
        self,
        benchmark_category: str | Sequence[str] | None = None,
        problem_type: str | Sequence[str] | None = None,
        instance_name: str | Sequence[str] | None = None,
        instance_id: str | Sequence[str] | None = None,
        solution_algorithm: str | Sequence[str] | None = None,
        min_qubits: int = 0,
        max_qubits: int | None = None,
        runnable_only: bool = False,
    ) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for category, entry in self._case_entries():
            if not self._matches(category, benchmark_category):
                continue
            case = load_benchmark_case_entry(entry)
            row = self._case_row(category, entry, case)
            if not self._matches(row["problem_type"], problem_type):
                continue
            if not self._matches(row["instance_name"], instance_name):
                continue
            if not self._matches(row["instance_id"], instance_id):
                continue
            if not self._matches_any(row["solution_algorithms"], solution_algorithm):
                continue
            if row["num_qubits"] < min_qubits:
                continue
            if max_qubits is not None and row["num_qubits"] > max_qubits:
                continue
            if runnable_only:
                available = get_available_runner_names(row["benchmark_category"])
                _, shipped, _ = classify_solution_algorithms(case, available)
                if not shipped:
                    continue
            rows.append(row)

        return pd.DataFrame(rows, columns=BENCHMARK_CASE_COLUMNS)

    def get_algorithms(
        self,
        benchmark_category: str | Sequence[str] | None = None,
        *,
        include_open: bool = True,
    ) -> pd.DataFrame:
        counts: dict[tuple[str, str, str, bool], int] = {}
        for category, entry in self._case_entries():
            if not self._matches(category, benchmark_category):
                continue
            case = load_benchmark_case_entry(entry)
            available = get_available_runner_names(category)
            open_algorithms, shipped_algorithms, unregistered_algorithms = (
                classify_solution_algorithms(case, available)
            )
            if unregistered_algorithms:
                raise BenchmarkError(
                    f"Case '{case.instance_name}' lists algorithms with no registered runners: "
                    f"{', '.join(unregistered_algorithms)}. Mark them in "
                    "open_solution_algorithms if they are intentionally "
                    "bring-your-own-solver benchmarks."
                )
            statuses: list[tuple[str, str]] = [(algorithm, "shipped") for algorithm in shipped_algorithms]
            if include_open:
                statuses.extend((algorithm, "open") for algorithm in open_algorithms)

            for algorithm, status in statuses:
                key = (category, algorithm, status, bool(entry.get("builtin", False)))
                counts[key] = counts.get(key, 0) + 1

        rows = [
            {
                "benchmark_category": category,
                "algorithm": algorithm,
                "status": status,
                "builtin": builtin,
                "case_count": case_count,
            }
            for (category, algorithm, status, builtin), case_count in sorted(counts.items())
        ]
        return pd.DataFrame(rows, columns=ALGORITHM_COLUMNS)

    def run_benchmarks(
        self,
        benchmark_cases: pd.DataFrame,
        backend: AbstractBackend | str | Sequence[AbstractBackend | str],
        shots: int | Sequence[int] | None = None,
        solution_algorithms: str | Sequence[str] | None = None,
        **runner_kwargs: Any,
    ) -> pd.DataFrame:
        records: list[BenchmarkSubmissionRecord] = []
        for backend_instance in self._normalize_backends(backend):
            for shots_value in self._normalize_shots(shots):
                for _, row in benchmark_cases.iterrows():
                    case = self._load_case_from_row(row)
                    for algorithm in self._algorithms_for_case(case, solution_algorithms):
                        runner = load_runner(case.benchmark_category, algorithm)
                        resolved_shots = resolve_shots(
                            user_shots=shots_value,
                            benchmark_shots=get_shots_for_case(case),
                        )
                        records.append(
                            runner.run_benchmark(
                                case,
                                backend_instance,
                                shots=resolved_shots,
                                **runner_kwargs,
                            )
                        )

        return pd.DataFrame([asdict(record) for record in records], columns=RESULT_COLUMNS)

    def submit_benchmarks(
        self,
        benchmark_cases: pd.DataFrame,
        backend: AbstractAsyncBackend | str | Sequence[AbstractAsyncBackend | str],
        *,
        record_db: Path | str,
        shots: int | Sequence[int] | None = None,
        solution_algorithms: str | Sequence[str] | None = None,
    ) -> pd.DataFrame:
        records: list[QCBenchmarkSubmissionRecord] = []
        for backend_instance in self._normalize_async_backends(backend):
            for shots_value in self._normalize_shots(shots):
                for _, row in benchmark_cases.iterrows():
                    case = self._load_case_from_row(row)
                    for algorithm in self._algorithms_for_case(case, solution_algorithms):
                        runner = load_runner(case.benchmark_category, algorithm)
                        if not isinstance(runner, CircuitBenchmarkRunner):
                            raise BenchmarkValidationError(
                                f"Runner '{algorithm}' does not support async circuit submission."
                            )
                        resolved_shots = resolve_shots(
                            user_shots=shots_value,
                            benchmark_shots=get_shots_for_case(case),
                        )
                        records.append(
                            runner.submit_benchmark_circuits(
                                case,
                                backend_instance,
                                shots=resolved_shots,
                            )
                        )

        frame = pd.DataFrame([asdict(record) for record in records], columns=SUBMISSION_COLUMNS)
        self._write_submission_record(frame, Path(record_db))
        return frame

    def load_submission_record(
        self,
        record_db: Path | str,
    ) -> pd.DataFrame:
        path = Path(record_db)
        if not path.exists():
            raise BenchmarkCaseError(f"Submission record does not exist: {path}")
        return self._hydrate_submission_record(pd.read_json(path, orient="index"))

    def update_submission_record(
        self,
        record_db: Path | str,
        backend: AbstractAsyncBackend | str,
    ) -> pd.DataFrame:
        path = Path(record_db)
        backend_instance = self._normalize_async_backend(backend)
        frame = self.load_submission_record(path)
        retrieval_time = pd.Timestamp.now(tz="UTC")

        for index, row in frame.iterrows():
            if row["backend"] != backend_instance.name():
                continue

            status = backend_instance.job_status(str(row["job_id"]))
            status_value = status.value if isinstance(status, JobStatus) else str(status)
            frame.at[index, "status"] = status_value
            frame.at[index, "last_retrieval"] = retrieval_time

            if status == JobStatus.FAILED and pd.isna(row["end_time"]):
                frame.at[index, "end_time"] = retrieval_time
                frame.at[index, "time_to_soln"] = retrieval_time - row["start_time"]
                frame.at[index, "adjusted_tts"] = retrieval_time - row["start_time"]
                continue

            if status != JobStatus.DONE or pd.notna(row["end_time"]):
                continue

            retrieval_start = pd.Timestamp.now(tz="UTC")
            measurements, completion_time = backend_instance.retrieve_results(
                str(row["job_id"]),
                dict(row["job_data"]),
            )
            end_time = pd.Timestamp.now(tz="UTC")
            case_path, category = self._find_case_for_record(str(row["instance_id"]))
            case = BenchmarkCase.load_from_database(case_path)
            runner = load_runner(category, str(row["solution_algorithm"]))
            if not isinstance(runner, CircuitBenchmarkRunner):
                raise BenchmarkValidationError(
                    f"Runner '{row['solution_algorithm']}' does not support circuit scoring."
                )

            merit_figures = runner.merit_figures_from_measurements(measurements, case)
            start_time = pd.Timestamp(row["start_time"])
            completion_timestamp = pd.to_datetime(completion_time, utc=True)
            frame.at[index, "measurements"] = measurements
            frame.at[index, "score"] = float(merit_figures["score"])
            frame.at[index, "problem_specific_data"] = {
                key: value
                for key, value in merit_figures.items()
                if key not in {"score", "total_shots"}
            }
            frame.at[index, "end_time"] = end_time
            frame.at[index, "time_to_soln"] = end_time - start_time
            frame.at[index, "adjusted_tts"] = completion_timestamp - start_time + (
                end_time - retrieval_start
            )

        self._write_submission_record(frame, path)
        return self._hydrate_submission_record(frame)

    def get_benchmark_circuit_stats(
        self,
        benchmark_cases: pd.DataFrame,
        solution_algorithms: str | Sequence[str] | None = None,
    ) -> pd.DataFrame:
        records: list[CircuitStats] = []
        for _, row in benchmark_cases.iterrows():
            case = self._load_case_from_row(row)
            for algorithm in self._algorithms_for_case(case, solution_algorithms):
                runner = load_runner(case.benchmark_category, algorithm)
                if not isinstance(runner, CircuitBenchmarkRunner):
                    raise BenchmarkValidationError(
                        f"Runner '{algorithm}' does not provide circuit stats."
                    )
                records.append(runner.get_circuit_stats(case))

        return pd.DataFrame([asdict(record) for record in records], columns=STATS_COLUMNS)

    def _case_entries(self) -> list[tuple[str, dict[str, Any]]]:
        return iter_benchmark_case_entries(
            include_builtin=self.include_builtin,
            include_diy=self.include_diy,
        )

    def _case_row(
        self,
        category: str,
        entry: dict[str, Any],
        case: BenchmarkCase,
    ) -> dict[str, Any]:
        open_algorithms = case.open_solution_algorithms or []
        return {
            "benchmark_category": case.benchmark_category or category,
            "problem_type": case.problem_type,
            "instance_name": case.instance_name,
            "instance_id": case.instance_id,
            "num_qubits": case.num_qubits,
            "solution_algorithms": case.solution_algorithms,
            "open_solution_algorithms": open_algorithms,
            "all_solutions_open": bool(case.solution_algorithms)
            and set(case.solution_algorithms) == set(open_algorithms),
            "builtin": bool(entry.get("builtin", False)),
            "file": Path(str(entry["file"])),
        }

    def _load_case_from_row(self, row: pd.Series) -> BenchmarkCase:
        try:
            return BenchmarkCase.load_from_database(Path(row["file"]))
        except Exception as exc:
            raise BenchmarkCaseError(f"Failed to load benchmark case from row: {exc}") from exc

    def _algorithms_for_case(
        self,
        benchmark_case: BenchmarkCase,
        requested_algorithms: str | Sequence[str] | None,
    ) -> list[str]:
        available = get_available_runner_names(benchmark_case.benchmark_category)
        if requested_algorithms is None:
            return [select_solution_algorithm(benchmark_case, available)]

        selected = []
        for algorithm in self._as_list(requested_algorithms):
            if algorithm not in benchmark_case.solution_algorithms:
                continue
            selected.append(select_solution_algorithm(benchmark_case, available, algorithm))
        if not selected:
            raise BenchmarkError(
                f"No requested algorithms match case '{benchmark_case.instance_name}'."
            )
        return selected

    def _normalize_backends(
        self,
        backend: AbstractBackend | str | Sequence[AbstractBackend | str],
    ) -> list[AbstractBackend]:
        if isinstance(backend, str) or isinstance(backend, AbstractBackend):
            backends = [backend]
        else:
            backends = list(backend)
        return [load_backend(item) if isinstance(item, str) else item for item in backends]

    def _normalize_async_backends(
        self,
        backend: AbstractAsyncBackend | str | Sequence[AbstractAsyncBackend | str],
    ) -> list[AbstractAsyncBackend]:
        if isinstance(backend, str) or isinstance(backend, AbstractBackend):
            backends = [backend]
        else:
            backends = list(backend)
        return [self._normalize_async_backend(item) for item in backends]

    def _normalize_async_backend(
        self,
        backend: AbstractAsyncBackend | str,
    ) -> AbstractAsyncBackend:
        backend_instance = load_backend(backend) if isinstance(backend, str) else backend
        if not isinstance(backend_instance, AbstractAsyncBackend):
            raise BackendValidationError(
                f"Backend '{backend_instance.name()}' does not support async submission."
            )
        return backend_instance

    def _normalize_shots(self, shots: int | Sequence[int] | None) -> list[int | None]:
        if shots is None:
            return [None]
        if isinstance(shots, int):
            return [shots]
        values = list(shots)
        if not all(isinstance(value, int) for value in values):
            raise BenchmarkValidationError("Shots must be integers.")
        return values

    def _write_submission_record(self, frame: pd.DataFrame, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_json(path, orient="index", date_unit="ns")

    def _hydrate_submission_record(self, frame: pd.DataFrame) -> pd.DataFrame:
        for column in ("start_time", "end_time", "last_retrieval"):
            if column in frame:
                frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")
        for column in ("time_to_soln", "adjusted_tts"):
            if column in frame:
                frame[column] = pd.to_timedelta(frame[column], errors="coerce")
        return frame

    def _find_case_for_record(self, instance_id: str) -> tuple[Path, str]:
        result = find_benchmark_case_by_uuid(instance_id)
        if result is None:
            raise BenchmarkCaseError(f"Benchmark case '{instance_id}' not found.")
        return result

    def _as_list(self, value: str | Sequence[str]) -> list[str]:
        return [value] if isinstance(value, str) else list(value)

    def _matches(self, value: Any, expected: str | Sequence[str] | None) -> bool:
        if expected is None:
            return True
        return value in self._as_list(expected)

    def _matches_any(
        self,
        values: Sequence[str],
        expected: str | Sequence[str] | None,
    ) -> bool:
        if expected is None:
            return True
        return bool(set(values) & set(self._as_list(expected)))
