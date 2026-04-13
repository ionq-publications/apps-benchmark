# DIY Backend Guide

This guide shows you how to create your own custom quantum backend for the apps-benchmark framework.

## Overview

A backend in apps-benchmark implements either:
- `AbstractBackend` - For synchronous backends (local simulators)
- `AbstractAsyncBackend` - For asynchronous backends (cloud QPUs, remote simulators)

Both interfaces require:
1. `name()` - Return backend identifier
2. `run()` - Execute circuits and return results

Async backends additionally need:
3. `submit()` - Submit circuits (non-blocking)
4. `job_status()` - Check job status
5. `retrieve_results()` - Get results from completed job

## Quick Start: Synchronous Backend

Here's a minimal example of a local simulator backend:

```python
from apps_benchmark.core.backend import AbstractBackend, MeasurementBatch, JobData
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
import uuid

class MySimpleBackend(AbstractBackend):
    """A simple Qiskit Aer-based simulator backend."""

    def __init__(self, backend_name: str = "aer_simulator"):
        """Initialize the backend."""
        self.simulator = AerSimulator()
        self._backend_name = backend_name

    def name(self) -> str:
        """Return backend identifier."""
        return f"my_simulator.{self._backend_name}"

    def run(
        self,
        circuits: list[QuantumCircuit],
        shots: int = 1000,
        job_name: str | None = None,
    ) -> tuple[MeasurementBatch, str, JobData]:
        """
        Execute circuits and return results.

        Returns:
            tuple: (results, job_id, job_data)
        """
        # Execute circuits on simulator
        job = self.simulator.run(circuits, shots=shots)
        result = job.result()

        # Extract measurement histograms
        measurement_batch = []
        for i in range(len(circuits)):
            counts = result.get_counts(i)
            measurement_batch.append(counts)

        # Generate job ID
        job_id = str(uuid.uuid4())

        # Serialize job data for reproducibility
        job_data = self.serialize_job_data(
            circuits,
            shots,
            job_name or "unnamed_job"
        )

        return measurement_batch, job_id, job_data
```

## Quick Start: Asynchronous Backend

Here's a minimal example of a cloud backend:

```python
from apps_benchmark.core.backend import (
    AbstractAsyncBackend,
    JobStatus,
    MeasurementBatch,
    JobData,
)
from qiskit import QuantumCircuit
import pandas as pd
import requests

class MyCloudBackend(AbstractAsyncBackend):
    """A cloud-based quantum backend."""

    def __init__(self, api_key: str, base_url: str):
        """Initialize with API credentials."""
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {"Authorization": f"Bearer {api_key}"}

    def name(self) -> str:
        """Return backend identifier."""
        return "my_cloud.qpu"

    def submit(
        self,
        circuits: list[QuantumCircuit],
        shots: int = 1000,
        job_name: str | None = None,
    ) -> tuple[str, JobData]:
        """Submit circuits to cloud backend."""
        # Serialize job data
        job_data = self.serialize_job_data(
            circuits,
            shots,
            job_name or "unnamed_job"
        )

        # Submit to API
        response = requests.post(
            f"{self.base_url}/jobs",
            json={
                "circuits": job_data["circuits"],
                "shots": shots,
                "name": job_name,
            },
            headers=self.headers,
        )
        response.raise_for_status()

        # Extract job ID from response
        job_id = response.json()["job_id"]

        return job_id, job_data

    def job_status(self, job_id: str) -> JobStatus:
        """Check job status."""
        response = requests.get(
            f"{self.base_url}/jobs/{job_id}",
            headers=self.headers,
        )
        response.raise_for_status()

        status_str = response.json()["status"]

        # Map API status to JobStatus enum
        status_map = {
            "submitted": JobStatus.SUBMITTED,
            "queued": JobStatus.QUEUED,
            "running": JobStatus.RUNNING,
            "completed": JobStatus.DONE,
            "failed": JobStatus.FAILED,
        }
        return status_map.get(status_str, JobStatus.QUEUED)

    def retrieve_results(
        self,
        job_id: str,
        job_data: JobData,
    ) -> tuple[MeasurementBatch, pd.Timestamp]:
        """Retrieve results from completed job."""
        response = requests.get(
            f"{self.base_url}/jobs/{job_id}/results",
            headers=self.headers,
        )
        response.raise_for_status()

        data = response.json()

        # Extract measurement histograms
        results = data["results"]

        # Get completion time
        completion_time = pd.Timestamp(data["completed_at"], tz="UTC")

        return results, completion_time
```

## Step-by-Step Guide

### Step 1: Choose Your Backend Type

**Synchronous Backend** (use `AbstractBackend`):
- Local simulators
- Immediate execution
- Results available instantly

**Asynchronous Backend** (use `AbstractAsyncBackend`):
- Cloud QPUs
- Remote simulators
- Job queue systems

### Step 2: Create Backend File

Create `apps_benchmark/backends/my_backend.py`:

```python
from apps_benchmark.core.backend import AbstractBackend
from qiskit import QuantumCircuit

class MyBackend(AbstractBackend):
    """My custom backend implementation."""

    def __init__(self, config: dict):
        """Initialize backend with configuration."""
        self.config = config

    def name(self) -> str:
        """Return backend identifier."""
        return "my_backend"

    def run(
        self,
        circuits: list[QuantumCircuit],
        shots: int = 1000,
        job_name: str | None = None,
    ) -> tuple[list[dict[str, int]], str, dict]:
        """Execute circuits."""
        # TODO: Implement execution
        pass
```

### Step 3: Implement name()

Return a unique identifier for your backend:

```python
def name(self) -> str:
    """Return backend identifier."""
    # Use dot notation for namespacing
    # Examples: "ionq.forte", "qiskit.aer", "my_company.simulator"
    return "my_company.my_backend"
```

### Step 4: Implement run() (Synchronous)

For synchronous backends, execute and return results immediately:

```python
def run(
    self,
    circuits: list[QuantumCircuit],
    shots: int = 1000,
    job_name: str | None = None,
) -> tuple[MeasurementBatch, str, JobData]:
    """Execute circuits synchronously."""
    import uuid

    # Execute circuits on your backend
    results = []
    for circuit in circuits:
        counts = self._execute_single_circuit(circuit, shots)
        results.append(counts)

    # Generate unique job ID
    job_id = str(uuid.uuid4())

    # Serialize job data
    job_data = self.serialize_job_data(circuits, shots, job_name or "job")

    return results, job_id, job_data

def _execute_single_circuit(
    self, circuit: QuantumCircuit, shots: int
) -> dict[str, int]:
    """Execute a single circuit and return counts."""
    # Your execution logic here
    # Must return dict like: {'00': 480, '01': 20, '10': 23, '11': 477}
    pass
```

### Step 5: Implement Async Methods (If Async)

For asynchronous backends, implement three methods:

```python
def submit(
    self,
    circuits: list[QuantumCircuit],
    shots: int = 1000,
    job_name: str | None = None,
) -> tuple[str, JobData]:
    """Submit circuits (non-blocking)."""
    # Serialize circuits
    job_data = self.serialize_job_data(circuits, shots, job_name or "job")

    # Submit to your backend API/service
    job_id = self._submit_to_service(job_data, shots)

    return job_id, job_data

def job_status(self, job_id: str) -> JobStatus:
    """Check job status."""
    # Query your backend for job status
    status_str = self._query_job_status(job_id)

    # Map to JobStatus enum
    from apps_benchmark.core.backend import JobStatus
    if status_str == "completed":
        return JobStatus.DONE
    elif status_str == "running":
        return JobStatus.RUNNING
    elif status_str == "failed":
        return JobStatus.FAILED
    else:
        return JobStatus.QUEUED

def retrieve_results(
    self,
    job_id: str,
    job_data: JobData,
) -> tuple[MeasurementBatch, pd.Timestamp]:
    """Retrieve results from completed job."""
    import pandas as pd

    # Fetch results from your backend
    raw_results = self._fetch_results(job_id)

    # Convert to measurement batch format
    results = self._parse_results(raw_results)

    # Get completion timestamp
    completion_time = pd.Timestamp.now(tz="UTC")

    return results, completion_time
```

### Step 6: Handle Errors

Use custom error types for better debugging:

```python
from apps_benchmark.errors import (
    BackendError,
    BackendConnectionError,
    BackendCredentialError,
)

def run(self, circuits, shots, job_name=None):
    """Execute circuits with error handling."""
    try:
        # Attempt execution
        results = self._execute(circuits, shots)
    except ConnectionError as e:
        raise BackendConnectionError(f"Cannot reach backend: {e}") from e
    except AuthenticationError as e:
        raise BackendCredentialError(f"Invalid credentials: {e}") from e
    except Exception as e:
        raise BackendError(f"Execution failed: {e}") from e

    return results, job_id, job_data
```

### Step 7: Implement Connection Validation (Optional)

Add a method to test connectivity:

```python
def validate_connection(self) -> bool:
    """Test backend connectivity."""
    try:
        # Minimal check (don't execute circuits)
        response = self._ping_backend()
        return response.status_code == 200
    except Exception:
        return False
```

## Result Format

Results must be returned as a list of measurement histograms:

```python
# Single circuit result
counts = {
    "00": 480,   # Bitstring "00" measured 480 times
    "01": 20,    # Bitstring "01" measured 20 times
    "10": 23,    # Bitstring "10" measured 23 times
    "11": 477,   # Bitstring "11" measured 477 times
}

# Multiple circuits
measurement_batch = [
    {"00": 480, "11": 520},  # Circuit 1 results
    {"000": 125, "111": 875},  # Circuit 2 results
]
```

## Configuration and Credentials

### Environment Variables

```python
import os

class MyBackend(AbstractBackend):
    def __init__(self):
        # Load from environment
        self.api_key = os.getenv("MY_BACKEND_API_KEY")
        if not self.api_key:
            raise ValueError("MY_BACKEND_API_KEY not set")

        self.endpoint = os.getenv(
            "MY_BACKEND_ENDPOINT",
            "https://api.mybackend.com"  # Default
        )
```

### Config Dictionary

```python
class MyBackend(AbstractBackend):
    def __init__(self, config: dict | None = None):
        config = config or {}

        self.api_key = config.get("api_key")
        self.timeout = config.get("timeout", 300)
        self.max_retries = config.get("max_retries", 3)
```

## Best Practices

### Use Type Hints
```python
from apps_benchmark.core.backend import MeasurementBatch, JobData

def run(
    self,
    circuits: list[QuantumCircuit],
    shots: int = 1000,
    job_name: str | None = None,
) -> tuple[MeasurementBatch, str, JobData]:
    pass
```

### Add Logging
```python
import logging

logger = logging.getLogger(__name__)

class MyBackend(AbstractBackend):
    def run(self, circuits, shots, job_name=None):
        logger.info(f"Submitting {len(circuits)} circuits with {shots} shots")
        results, job_id, job_data = self._execute(circuits, shots)
        logger.info(f"Job {job_id} completed")
        return results, job_id, job_data
```

### Handle Timeouts
```python
import time

class MyAsyncBackend(AbstractAsyncBackend):
    def run(self, circuits, shots, job_name=None):
        """Run with timeout."""
        job_id, job_data = self.submit(circuits, shots, job_name)

        timeout = 300  # 5 minutes
        start = time.time()

        while time.time() - start < timeout:
            status = self.job_status(job_id)
            if status == JobStatus.DONE:
                break
            elif status == JobStatus.FAILED:
                raise BackendError(f"Job {job_id} failed")
            time.sleep(2)
        else:
            raise TimeoutError(f"Job {job_id} timed out after {timeout}s")

        results, completion_time = self.retrieve_results(job_id, job_data)
        return results, job_id, job_data
```

### Cache Results (Optional)
```python
class MyBackend(AbstractBackend):
    def __init__(self):
        self._result_cache = {}

    def run(self, circuits, shots, job_name=None):
        # Create cache key
        cache_key = self._make_cache_key(circuits, shots)

        # Check cache
        if cache_key in self._result_cache:
            return self._result_cache[cache_key]

        # Execute and cache
        results, job_id, job_data = self._execute(circuits, shots)
        self._result_cache[cache_key] = (results, job_id, job_data)

        return results, job_id, job_data
```

## Testing Your Backend

Create `tests/backends/test_my_backend.py`:

```python
import pytest
from qiskit import QuantumCircuit
from apps_benchmark.backends.my_backend import MyBackend

def test_backend_name():
    """Test backend identifier."""
    backend = MyBackend()
    assert backend.name() == "my_company.my_backend"

def test_simple_circuit():
    """Test executing a simple circuit."""
    backend = MyBackend()

    # Create simple circuit
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure_all()

    # Execute
    results, job_id, job_data = backend.run([circuit], shots=1000)

    # Verify results
    assert len(results) == 1
    assert isinstance(results[0], dict)
    assert sum(results[0].values()) == 1000  # Total counts = shots
    assert job_id is not None

def test_error_handling():
    """Test that errors are raised correctly."""
    backend = MyBackend()

    with pytest.raises(ValueError):
        backend.run([], shots=1000)  # Empty circuit list
```

## Integration

### Register Your Backend

Add to `apps_benchmark/backends/__init__.py`:

```python
from .my_backend import MyBackend

__all__ = ["MyBackend", ...]
```

### Use in CLI

```bash
# Your backend is now available
apps-benchmark run qaoa --backend my_company.my_backend --shots 1000
```

## Advanced Topics

### Custom Serialization

Override serialization for custom circuit formats:

```python
def serialize_job_data(self, circuits, shots, job_name):
    """Serialize to custom format."""
    return {
        "circuits": [self._to_custom_format(qc) for qc in circuits],
        "shots": shots,
        "job_name": job_name,
        "backend_version": "1.0.0",
    }

def de_serialize_job_data(self, job_data):
    """Deserialize from custom format."""
    circuits = [self._from_custom_format(c) for c in job_data["circuits"]]
    return {
        "circuits": circuits,
        "shots": job_data["shots"],
        "job_name": job_data["job_name"],
    }
```

### Batch Processing

Handle multiple circuits efficiently:

```python
def run(self, circuits, shots, job_name=None):
    """Run with batching for large circuit lists."""
    batch_size = 100
    all_results = []

    for i in range(0, len(circuits), batch_size):
        batch = circuits[i:i + batch_size]
        batch_results = self._execute_batch(batch, shots)
        all_results.extend(batch_results)

    job_id = str(uuid.uuid4())
    job_data = self.serialize_job_data(circuits, shots, job_name or "job")

    return all_results, job_id, job_data
```

## Next Steps

- See existing backends in `apps_benchmark/backends/`
- Read [CONTRIBUTING.md](CONTRIBUTING.md) for testing guidelines
- Check [API_REFERENCE.md](API_REFERENCE.md) for detailed API docs
- Review `apps_benchmark/core/backend.py:34-333` for full interface

## Getting Help

- Check existing backend implementations for examples
- Open an issue on GitHub
- Contact App Benchmark Support at apps-benchmark-support@ionq.co
