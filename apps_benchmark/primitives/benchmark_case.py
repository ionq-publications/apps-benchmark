"""
BenchmarkCase dataclass for benchmark problem definitions.

This module defines the standard format for all benchmark problems.

Copyright (c) 2025 IonQ, Inc. All rights reserved.
Licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0
International (CC BY-NC-ND 4.0) License.
For details, visit: https://creativecommons.org/licenses/by-nc-nd/4.0/
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass
class BenchmarkCase:
    """
    Standard format for benchmark problem instances.

    This class defines a common interface for all benchmark problems.
    Instances are typically loaded from JSON files.

    Attributes:
        benchmark_category: Category name (e.g., "chemistry", "qft")
        problem_type: Type of problem (e.g., "hydrogen_lattice_vqe", "qft")
        instance_name: Human-readable name for this specific instance
        num_qubits: Number of qubits required to run this problem
        solution_algorithms: List of compatible algorithm runner names. The first
            algorithm in this list is used by default when running via CLI. To use
            alternative algorithms, pass --algorithm=<name> to the run command.
            Example: ["qft", "qft_lcu", "qft_hidden_shift"] allows running the
            same problem with different solution approaches.
        data: Problem-specific parameters and reference values
        instance_id: Unique identifier (auto-generated from UUID if not provided)
    """

    benchmark_category: str
    problem_type: str
    instance_name: str
    num_qubits: int
    solution_algorithms: list[str]
    data: dict[str, Any]
    instance_id: str | None = None

    def __post_init__(self) -> None:
        """Generate instance_id if not provided."""
        if self.instance_id is None:
            self.instance_id = uuid.uuid4().hex[:8]

    def dump(
        self,
        db_name: str | Path,
        converter: dict[str, Callable[[Any], Any]] | None = None,
    ) -> None:
        """
        Write this problem instance to JSON file.

        This convenience method saves the problem instance to a JSON file.
        The optional converter is used to convert values in self.data
        to objects that are compatible with json.dump.

        Args:
            db_name: Path to JSON file to write
            converter: Optional dict mapping keys in self.data to converter
                      functions. Used to convert non-serializable objects
                      (e.g., numpy arrays) to JSON-compatible formats.

        Example:
            >>> import numpy as np
            >>> problem = BenchmarkCase(
            ...     benchmark_category="chemistry",
            ...     problem_type="test",
            ...     instance_name="example",
            ...     num_qubits=2,
            ...     solution_algorithms=["vqe"],
            ...     data={"matrix": np.array([[1, 2], [3, 4]])},
            ... )
            >>> # Convert numpy array to list for JSON
            >>> problem.dump("problem.json", converter={"matrix": lambda x: x.tolist()})
        """
        instance_data = asdict(self)
        converter = converter or {}

        # Apply converters to data fields
        for key in converter:
            if key in instance_data["data"]:
                instance_data["data"][key] = converter[key](instance_data["data"][key])

        with open(db_name, "w") as db:
            json.dump(instance_data, db, indent=2, sort_keys=True)
            db.write("\n")

    @classmethod
    def load_from_database(cls, problem_path: str | Path) -> BenchmarkCase:
        """
        Load a BenchmarkCase from JSON file.

        Args:
            problem_path: Path to .json file

        Returns:
            BenchmarkCase: Loaded instance

        Raises:
            ValueError: If file doesn't have .json extension
            FileNotFoundError: If file doesn't exist
            json.JSONDecodeError: If file is not valid JSON

        Example:
            >>> problem = BenchmarkCase.load_from_database(
            ...     "~/local_dev/benchmarks/chemistry/benchmark_cases/h2.json"
            ... )
            >>> problem.num_qubits
            2
        """
        path = Path(problem_path)
        if path.suffix != ".json":
            raise ValueError(f"Problem file must have .json extension: {problem_path}")

        with open(path) as f:
            data = json.load(f)

        return cls(**data)
