# DIY Benchmark Guide

This guide shows you how to create your own custom benchmark algorithm for the apps-benchmark framework.

## Overview

A benchmark algorithm in apps-benchmark implements the `AbstractAlgoRunner` interface, which requires four key methods:

1. `name()` - Return your algorithm's identifier
2. `setup_algo_inputs()` - Parse the problem and prepare inputs
3. `execute_benchmark_algo()` - Run your quantum algorithm
4. `compute_merit_figures()` - Calculate scores and metrics

## Quick Start Example

Here's a minimal example of a custom benchmark:

```python
from typing import Any
from apps_benchmark.core.benchmark import AbstractAlgoRunner
from apps_benchmark.core.backend import AbstractBackend
from apps_benchmark.primitives.benchmark_case import BenchmarkCase
from qiskit import QuantumCircuit

class MySimpleBenchmark(AbstractAlgoRunner):
    """A simple example benchmark that runs a parametrized circuit."""

    def name(self) -> str:
        """Return the algorithm name."""
        return "my_simple_algo"

    def setup_algo_inputs(
        self, benchmark_case: BenchmarkCase
    ) -> tuple[QuantumCircuit, list[float]]:
        """
        Parse the benchmark case and create algorithm inputs.

        Returns:
            tuple: (circuit, parameters)
        """
        # Extract problem data from benchmark_case
        num_qubits = benchmark_case.num_qubits

        # Create a simple parametrized circuit
        circuit = QuantumCircuit(num_qubits)
        for i in range(num_qubits):
            circuit.rx(0.1, i)  # Will be parameterized
        circuit.measure_all()

        # Initial parameters
        params = [0.5] * num_qubits

        return circuit, params

    def execute_benchmark_algo(
        self,
        algo_inputs: tuple[QuantumCircuit, list[float]],
        backend: AbstractBackend,
        shots: int,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Execute the algorithm on the backend.

        Returns:
            dict: Algorithm results
        """
        circuit, params = algo_inputs

        # Execute circuit on backend
        results, job_id, job_data = backend.run([circuit], shots=shots)

        # Extract measurement counts
        counts = results[0]

        # Return raw output
        return {
            "counts": counts,
            "job_id": job_id,
            "circuits_executed": 1,
        }

    def compute_merit_figures(
        self, algo_output: dict[str, Any], benchmark_case: BenchmarkCase
    ) -> dict[str, Any]:
        """
        Compute benchmark scores from algorithm output.

        Returns:
            dict: Must contain 'total_shots' and 'score'
        """
        counts = algo_output["counts"]

        # Calculate total shots
        total_shots = sum(counts.values())

        # Example score: probability of all-zeros state
        score = counts.get("0" * benchmark_case.num_qubits, 0) / total_shots

        # Return metrics (must include total_shots and score)
        return {
            "total_shots": total_shots,
            "score": score,
            "circuits_executed": algo_output["circuits_executed"],
            "job_id": algo_output["job_id"],
        }
```

## Step-by-Step Guide

### Step 1: Choose Your Category

Benchmarks are organized by category. Create your benchmark in the appropriate folder:

```
apps_benchmark/benchmarks/
├── chemistry/          # Molecular simulation algorithms
├── hidden_shift/       # Hidden shift benchmark family
├── image_loading/      # Image loading benchmark family
├── optimization/       # Combinatorial optimization
├── qft/                # Closed QFT benchmark family
└── unstructured_search/  # Search algorithms
```

Or create a new category:

```bash
mkdir -p apps_benchmark/benchmarks/my_category/algorithms
touch apps_benchmark/benchmarks/my_category/__init__.py
touch apps_benchmark/benchmarks/my_category/algorithms/__init__.py
```

### Step 2: Create Your Algorithm File

Create `apps_benchmark/benchmarks/my_category/algorithms/my_algo_runner.py`:

```python
from typing import Any
from apps_benchmark.core.benchmark import AbstractAlgoRunner
from apps_benchmark.core.backend import AbstractBackend
from apps_benchmark.primitives.benchmark_case import BenchmarkCase

class MyAlgoRunner(AbstractAlgoRunner):
    """Your algorithm description."""

    def name(self) -> str:
        return "my_algo"

    def setup_algo_inputs(self, benchmark_case: BenchmarkCase) -> tuple[Any, ...]:
        # TODO: Parse benchmark_case and prepare inputs
        pass

    def execute_benchmark_algo(
        self,
        algo_inputs: tuple[Any, ...],
        backend: AbstractBackend,
        shots: int,
        **kwargs: Any,
    ) -> Any:
        # TODO: Execute your algorithm
        pass

    def compute_merit_figures(
        self, algo_output: Any, benchmark_case: BenchmarkCase
    ) -> dict[str, Any]:
        # TODO: Calculate metrics
        return {
            "total_shots": 1000,  # Required
            "score": 0.95,        # Required
            # Add any other metrics here
        }
```

### Step 3: Implement setup_algo_inputs()

This method extracts data from the `BenchmarkCase` and prepares algorithm-specific inputs:

```python
def setup_algo_inputs(self, benchmark_case: BenchmarkCase) -> tuple[Any, ...]:
    """Parse problem and create inputs."""

    # Access problem data
    num_qubits = benchmark_case.num_qubits
    data = benchmark_case.data  # Problem-specific data

    # Example: extract a Hamiltonian
    hamiltonian_dict = data.get("hamiltonian")
    hamiltonian = self._parse_hamiltonian(hamiltonian_dict)

    # Create circuit/ansatz
    ansatz = self._create_ansatz(num_qubits)

    # Initial parameters
    initial_params = [0.0] * ansatz.num_parameters

    # Return everything your algorithm needs
    return hamiltonian, ansatz, initial_params
```

### Step 4: Implement execute_benchmark_algo()

This method runs your quantum algorithm using the backend:

```python
def execute_benchmark_algo(
    self,
    algo_inputs: tuple[Any, ...],
    backend: AbstractBackend,
    shots: int,
    **kwargs: Any,
) -> Any:
    """Execute the algorithm."""
    hamiltonian, ansatz, initial_params = algo_inputs

    # Build circuits with current parameters
    circuits = self._build_circuits(ansatz, initial_params)

    # Execute on backend
    results, job_id, job_data = backend.run(circuits, shots=shots)

    # Process results (e.g., run optimization loop)
    final_energy = self._optimize(hamiltonian, ansatz, backend, shots)

    # Return whatever compute_merit_figures() needs
    return {
        "final_energy": final_energy,
        "total_circuits": len(circuits),
        "job_id": job_id,
    }
```

### Step 5: Implement compute_merit_figures()

Calculate your benchmark score and metrics:

```python
def compute_merit_figures(
    self, algo_output: dict[str, Any], benchmark_case: BenchmarkCase
) -> dict[str, Any]:
    """Compute scores and metrics."""

    # Get results
    final_energy = algo_output["final_energy"]

    # Get expected answer from benchmark_case
    expected_energy = benchmark_case.data.get("ground_state_energy")

    # Calculate error
    energy_error = abs(final_energy - expected_energy)

    # Define score (e.g., accuracy metric)
    score = 1.0 / (1.0 + energy_error)

    # REQUIRED: Must return total_shots and score
    # Additional metrics are optional
    return {
        "total_shots": 1000 * algo_output["total_circuits"],
        "score": score,
        "final_energy": final_energy,
        "energy_error": energy_error,
        "expected_energy": expected_energy,
    }
```

## Best Practices

### Use Type Hints
```python
def setup_algo_inputs(
    self, benchmark_case: BenchmarkCase
) -> tuple[SparsePauliOp, QuantumCircuit, list[float]]:
    # Explicit return types help with debugging
    pass
```

### Handle Errors Gracefully
```python
from apps_benchmark.errors import BenchmarkError

def execute_benchmark_algo(self, algo_inputs, backend, shots, **kwargs):
    try:
        results, job_id, job_data = backend.run(circuits, shots)
    except Exception as e:
        raise BenchmarkError(f"Execution failed: {e}") from e
```

### Document Your Score Metric
```python
def compute_merit_figures(self, algo_output, benchmark_case):
    """
    Compute merit figures.

    Score definition:
        - 1.0 = perfect accuracy (energy error = 0)
        - 0.5 = energy error = 1.0
        - 0.0 = completely wrong
    """
    # ... implementation
```

### Keep State Minimal
The runner should be stateless - all data flows through method parameters:
```python
# GOOD: Stateless
class MyRunner(AbstractAlgoRunner):
    def execute_benchmark_algo(self, algo_inputs, backend, shots, **kwargs):
        # All data passed as parameters
        pass

# BAD: Storing state
class MyRunner(AbstractAlgoRunner):
    def __init__(self):
        self.cached_circuits = []  # Avoid this
```

## Testing Your Benchmark

Create `tests/benchmarks/test_my_algo.py`:

```python
import pytest
from apps_benchmark.benchmarks.my_category.algorithms.my_algo_runner import MyAlgoRunner
from apps_benchmark.backends.mock_backend import MockBackend
from apps_benchmark.primitives.benchmark_case import BenchmarkCase

def test_my_algo_runner():
    """Test basic functionality."""
    runner = MyAlgoRunner()

    # Create a simple benchmark case
    benchmark_case = BenchmarkCase(
        problem_type="test_problem",
        instance_name="test_001",
        num_qubits=3,
        data={"test_param": 42},
    )

    # Use mock backend
    backend = MockBackend()

    # Run benchmark
    record = runner.run_benchmark(benchmark_case, backend, shots=1000)

    # Verify results
    assert record.status == "done"
    assert record.score >= 0.0
    assert record.total_shots > 0
```

## Integration

### Register Your Algorithm

Add to `apps_benchmark/benchmarks/my_category/algorithms/__init__.py`:

```python
from .my_algo_runner import MyAlgoRunner

__all__ = ["MyAlgoRunner"]
```

### Use in CLI

```bash
# Run your algorithm on a specific benchmark case
apps-benchmark run --backend=qiskit_aer_sim_backend --case-uuid=<uuid> --algorithm=my_algo --shots=1000

# Run all benchmarks in your category
apps-benchmark run --backend=qiskit_aer_sim_backend --category=my_category --shots=1000

# If your benchmark case supports multiple algorithms, select yours explicitly
apps-benchmark run --backend=qiskit_aer_sim_backend --case-uuid=<uuid> --algorithm=my_algo
```

If a benchmark case includes algorithms that are intentionally not shipped in
this repo, add an `open_solution_algorithms` field to the case JSON. The CLI
will list those algorithms as open benchmarks and require the user to provide
their own solver instead of trying to import a missing runner.

## Advanced Topics

### Custom Optimization Loops
```python
def execute_benchmark_algo(self, algo_inputs, backend, shots, **kwargs):
    hamiltonian, ansatz, params = algo_inputs

    # Custom optimization
    for iteration in range(max_iterations):
        circuits = [ansatz.bind_parameters(params)]
        results, _, _ = backend.run(circuits, shots)

        # Compute gradient or update params
        params = self._update_params(results, hamiltonian)

    return {"final_params": params, "iterations": iteration}
```

### Multi-Circuit Execution
```python
def execute_benchmark_algo(self, algo_inputs, backend, shots, **kwargs):
    # Build multiple circuits
    circuits = [self._build_circuit(i) for i in range(10)]

    # Execute all at once
    results, job_id, job_data = backend.run(circuits, shots)

    # Process all results
    return {"all_results": results, "num_circuits": len(circuits)}
```

## Next Steps

- See existing implementations in `apps_benchmark/benchmarks/`
- Read [CONTRIBUTING.md](CONTRIBUTING.md) for testing and debugging
- Check [API_REFERENCE.md](API_REFERENCE.md) for detailed API docs
- Review `apps_benchmark/core/benchmark.py:61-267` for the full interface

## Getting Help

- Check existing benchmarks for examples
- Open an issue on GitHub
- Contact App Benchmark Support at apps-benchmark-support@ionq.co
