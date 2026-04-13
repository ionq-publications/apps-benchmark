# Registry System Design

This document details how the registry system works for auto-discovery and management of backends and benchmarks.

## Overview

The registry system tracks all available backends and benchmarks (both built-in and DIY) in JSON files stored in the `local_dev` directory.

**Registry Files:**
- `<local_dev>/backends.json` - All registered backends
- `<local_dev>/benchmarks.json` - All registered benchmarks

**Configuration:**
The `local_dev` directory location is configurable via `~/config_local_dev.json`:
- Default location: `~/local_dev/`
- Auto-created on first run with default configuration
- To change the location, edit `~/config_local_dev.json` directly

**Key Features:**
- Auto-discovery of built-in components on first run
- Manual registration of DIY components via CLI
- Validation during registration
- Atomic file updates (no partial writes)

---

## Registry Initialization

### First Run Behavior

When apps-benchmark is run for the first time:

```
User runs any apps-benchmark command
    ↓
Load local_dev location from ~/config_local_dev.json
    - If config doesn't exist: Create with default ~/local_dev/
    ↓
Check if <local_dev>/ exists
    - If NO: Create directory
    ↓
Check if <local_dev>/backends.json exists
    - If NO: Initialize empty registry
    ↓
Auto-discover built-in backends
    - Scan apps_benchmark/backends/ directory
    - For each .py file (except __init__.py):
        - Extract module name
        - Import module
        - Find backend class (subclass of AbstractBackend)
        - Add to registry with builtin=true
    ↓
Write <local_dev>/backends.json
    ↓
Repeat for benchmarks:
    - Scan apps_benchmark/benchmarks/ directories
    - For each category:
        - Find algorithm runners
        - Find problem instances (*.json)
        - Add to registry with builtin=true
    ↓
Write <local_dev>/benchmarks.json
    ↓
Continue with user's command
```

### Directory Creation

```python
import os
from pathlib import Path
from apps_benchmark.utils.config import get_local_dev_dir_from_config

def get_local_dev_dir():
    """Get local_dev directory path from configuration."""
    return get_local_dev_dir_from_config()

def ensure_local_dev_dir():
    """Create local_dev directory structure if it doesn't exist."""
    local_dev = get_local_dev_dir()
    local_dev.mkdir(exist_ok=True)

    backends_dir = local_dev / "backends"
    backends_dir.mkdir(exist_ok=True)

    benchmarks_dir = local_dev / "benchmarks"
    benchmarks_dir.mkdir(exist_ok=True)

    return local_dev
```

### Empty Registry Initialization

```python
def initialize_empty_registry(registry_path: Path):
    """Create empty registry file."""
    if registry_path.name == "backends.json":
        empty_registry = {
            "version": "1.0",
            "backends": {}
        }
    else:  # benchmarks.json
        empty_registry = {
            "version": "1.0",
            "builtin_benchmarks": {},
            "diy_benchmarks": {}
        }

    write_registry_atomic(registry_path, empty_registry)
```

---

## Auto-Discovery of Built-in Components

### Built-in Backend Discovery

```python
import importlib
import inspect
from pathlib import Path
from apps_benchmark.core.backend import AbstractBackend

def discover_builtin_backends() -> dict:
    """
    Scan apps_benchmark/backends/ and auto-discover backend classes.

    Returns:
        Dict mapping backend_name -> backend_info
    """
    backends = {}

    # Find backends directory in installed package
    import apps_benchmark.backends
    backends_dir = Path(apps_benchmark.backends.__file__).parent

    # Scan for .py files
    for py_file in backends_dir.glob("*.py"):
        if py_file.name.startswith("_"):
            continue  # Skip __init__.py, __pycache__, etc.

        module_name = py_file.stem
        full_module = f"apps_benchmark.backends.{module_name}"

        try:
            # Import module
            module = importlib.import_module(full_module)

            # Find backend classes
            for name, obj in inspect.getmembers(module, inspect.isclass):
                # Check if it's a backend (subclass of AbstractBackend)
                if issubclass(obj, AbstractBackend) and obj is not AbstractBackend:
                    backend_name = module_name  # Use module name as backend name

                    backends[backend_name] = {
                        "module": full_module,
                        "class": name,
                        "builtin": True,
                        "location": "/backends",
                        "registered_at": pd.Timestamp.now(tz="UTC").isoformat()
                    }
                    break  # One backend per module

        except Exception as e:
            # Log warning but don't fail
            print(f"Warning: Failed to discover backend in {module_name}: {e}")

    return backends
```

### Built-in Benchmark Discovery

```python
def discover_builtin_benchmarks() -> dict:
    """
    Scan apps_benchmark/benchmarks/ and auto-discover benchmarks.

    Returns:
        Dict mapping category -> benchmark_info
    """
    benchmarks = {}

    import apps_benchmark.benchmarks
    benchmarks_dir = Path(apps_benchmark.benchmarks.__file__).parent

    # Scan for category directories
    for category_dir in benchmarks_dir.iterdir():
        if not category_dir.is_dir() or category_dir.name.startswith("_"):
            continue

        category_name = category_dir.name

        # Find algorithm runners
        algorithms_dir = category_dir / "algorithms"
        runners = []
        if algorithms_dir.exists():
            for py_file in algorithms_dir.glob("*_runner.py"):
                runner_name = py_file.stem.replace("_runner", "")
                runners.append(runner_name)

        # Find problem instances
        instances_dir = category_dir / "benchmark_cases"
        benchmark_cases = []
        if instances_dir.exists():
            for json_file in instances_dir.glob("*.json"):
                try:
                    # Load JSON to get metadata
                    with open(json_file) as f:
                        data = json.load(f)

                    benchmark_cases.append({
                        "uuid": data.get("instance_id", "unknown"),
                        "name": data.get("instance_name", json_file.stem),
                        "file": str(json_file),
                        "builtin": True
                    })
                except Exception as e:
                    print(f"Warning: Failed to load {json_file}: {e}")

        benchmarks[category_name] = {
            "location": "/benchmarks",
            "runners": runners,
            "benchmark_cases": benchmark_cases
        }

    return benchmarks
```

---

## DIY Component Registration

### Backend Registration Process

```python
def register_diy_backend(backend_name: str) -> None:
    """
    Register a DIY backend from <local_dev>/backends/.

    Args:
        backend_name: Name of backend (e.g., "my_custom_backend")

    Raises:
        BackendValidationError: If backend fails validation
        FileNotFoundError: If backend file not found
    """
    # 1. Construct paths
    local_dev = get_local_dev_dir()
    backend_file = local_dev / "backends" / f"{backend_name}.py"

    if not backend_file.exists():
        raise FileNotFoundError(
            f"Backend file not found: {backend_file}\n"
            f"Please create the file before registering."
        )

    # 2. Expected class name (CamelCase conversion)
    expected_class_name = snake_to_camel(backend_name)

    # 3. Import module dynamically
    spec = importlib.util.spec_from_file_location(backend_name, backend_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # 4. Look for class
    backend_class = getattr(module, expected_class_name, None)
    if backend_class is None:
        available_classes = [
            name for name, obj in inspect.getmembers(module, inspect.isclass)
            if not name.startswith("_")
        ]
        raise BackendValidationError(
            f"Based on '{backend_name}' expected class '{expected_class_name}' "
            f"but did not find it. Backend not registered.\n\n"
            f"Available classes in {backend_name}.py: {available_classes}\n"
            f"Did you mean to name your class '{expected_class_name}'?"
        )

    # 5. Validate interface
    validate_backend_interface(backend_class)

    # 6. Test connection (optional)
    try:
        backend_instance = backend_class()
        backend_instance.validate_connection()
    except Exception as e:
        print(f"Warning: Connection validation failed: {e}")
        print("Backend will be registered but may not work correctly.")

    # 7. Update registry
    registry_path = local_dev / "backends.json"
    registry = load_registry(registry_path)

    registry["backends"][backend_name] = {
        "module": str(backend_file),
        "class": expected_class_name,
        "builtin": False,
        "location": "/backends-diy",
        "registered_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "validated": True
    }

    write_registry_atomic(registry_path, registry)

    print(f"✓ Backend '{backend_name}' registered successfully.")
```

### Benchmark Registration Process

```python
def register_diy_benchmark(
    benchmark_name: str,
    category: str
) -> None:
    """
    Register a DIY benchmark from <local_dev>/benchmarks/.

    Args:
        benchmark_name: Name of benchmark algorithm
        category: Benchmark category (e.g., "optimization")

    Raises:
        BenchmarkValidationError: If benchmark fails validation
        FileNotFoundError: If files not found
    """
    # 1. Construct paths
    local_dev = get_local_dev_dir()
    category_dir = local_dev / "benchmarks" / category

    if not category_dir.exists():
        raise FileNotFoundError(
            f"Category directory not found: {category_dir}\n"
            f"Please create: mkdir -p {category_dir}/algorithms"
        )

    algorithms_dir = category_dir / "algorithms"
    runner_file = algorithms_dir / f"{benchmark_name}_runner.py"

    if not runner_file.exists():
        raise FileNotFoundError(
            f"Runner file not found: {runner_file}\n"
            f"Please create the file before registering."
        )

    # 2. Expected class name
    expected_class_name = snake_to_camel(benchmark_name) + "Runner"

    # 3. Import runner module
    spec = importlib.util.spec_from_file_location(benchmark_name, runner_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # 4. Look for runner class
    runner_class = getattr(module, expected_class_name, None)
    if runner_class is None:
        available_classes = [
            name for name, obj in inspect.getmembers(module, inspect.isclass)
            if not name.startswith("_")
        ]
        raise BenchmarkValidationError(
            f"Expected class '{expected_class_name}' not found in {runner_file.name}.\n"
            f"Available classes: {available_classes}"
        )

    # 5. Validate runner interface
    validate_runner_interface(runner_class)

    # 6. Scan for problem instances
    instances_dir = category_dir / "benchmark_cases"
    benchmark_cases = []

    if instances_dir.exists():
        for json_file in instances_dir.glob("*.json"):
            try:
                with open(json_file) as f:
                    data = json.load(f)

                # Validate required fields
                required_fields = [
                    "benchmark_category", "problem_type", "instance_name",
                    "num_qubits", "solution_algorithms", "data"
                ]
                for field in required_fields:
                    if field not in data:
                        raise ValueError(f"Missing required field: {field}")

                benchmark_cases.append({
                    "uuid": data.get("instance_id", "unknown"),
                    "name": data["instance_name"],
                    "file": str(json_file),
                    "builtin": False
                })

            except Exception as e:
                print(f"Warning: Skipping invalid problem instance {json_file}: {e}")

    # 7. Update registry
    registry_path = local_dev / "benchmarks.json"
    registry = load_registry(registry_path)

    registry["diy_benchmarks"][benchmark_name] = {
        "category": category,
        "location": "/benchmarks-diy",
        "runner_module": str(runner_file),
        "benchmark_cases": benchmark_cases,
        "registered_at": pd.Timestamp.now(tz="UTC").isoformat()
    }

    write_registry_atomic(registry_path, registry)

    print(f"✓ Benchmark '{benchmark_name}' registered successfully.")
    print(f"  Found {len(benchmark_cases)} problem instance(s).")
```

---

## Interface Validation

### Backend Validation

```python
def validate_backend_interface(backend_class) -> None:
    """
    Validate that backend class implements required interface.

    Args:
        backend_class: Backend class to validate

    Raises:
        BackendValidationError: If validation fails
    """
    from apps_benchmark.core.backend import AbstractBackend

    # Check inheritance
    if not issubclass(backend_class, AbstractBackend):
        raise BackendValidationError(
            f"Backend must inherit from AbstractBackend"
        )

    # Check required methods
    required_methods = ["name", "run"]

    for method_name in required_methods:
        if not hasattr(backend_class, method_name):
            raise BackendValidationError(
                f"Backend does not implement required method '{method_name}'"
            )

        method = getattr(backend_class, method_name)
        if not callable(method):
            raise BackendValidationError(
                f"'{method_name}' must be a callable method"
            )

    # For async backends, check additional methods
    from apps_benchmark.core.backend import AbstractAsyncBackend
    if issubclass(backend_class, AbstractAsyncBackend):
        async_methods = ["submit", "job_status", "retrieve_results"]
        for method_name in async_methods:
            if not hasattr(backend_class, method_name):
                raise BackendValidationError(
                    f"Async backend missing required method '{method_name}'"
                )
```

### Benchmark Runner Validation

```python
def validate_runner_interface(runner_class) -> None:
    """
    Validate that runner class implements required interface.

    Args:
        runner_class: Runner class to validate

    Raises:
        BenchmarkValidationError: If validation fails
    """
    from apps_benchmark.core.benchmark import AbstractAlgoRunner

    # Check inheritance
    if not issubclass(runner_class, AbstractAlgoRunner):
        raise BenchmarkValidationError(
            f"Runner must inherit from AbstractAlgoRunner"
        )

    # Check required methods
    required_methods = [
        "name",
        "setup_algo_inputs",
        "execute_benchmark_algo",
        "compute_merit_figures"
    ]

    for method_name in required_methods:
        if not hasattr(runner_class, method_name):
            raise BenchmarkValidationError(
                f"Runner does not implement required method '{method_name}'"
            )

        method = getattr(runner_class, method_name)
        if not callable(method):
            raise BenchmarkValidationError(
                f"'{method_name}' must be a callable method"
            )
```

---

## Registry File Operations

### Atomic Writes

Registry files are written atomically to prevent corruption.

```python
import json
import tempfile
from pathlib import Path

def write_registry_atomic(registry_path: Path, registry_data: dict) -> None:
    """
    Write registry file atomically (temp file + rename).

    Args:
        registry_path: Path to registry file
        registry_data: Registry data to write
    """
    # Write to temporary file first
    temp_fd, temp_path = tempfile.mkstemp(
        dir=registry_path.parent,
        prefix=".registry_",
        suffix=".tmp"
    )

    try:
        with os.fdopen(temp_fd, 'w') as f:
            json.dump(registry_data, f, indent=2)

        # Atomic rename
        temp_path_obj = Path(temp_path)
        temp_path_obj.replace(registry_path)

    except Exception as e:
        # Clean up temp file on error
        Path(temp_path).unlink(missing_ok=True)
        raise RegistryError(f"Failed to write registry: {e}")
```

### Registry Loading

```python
def load_registry(registry_path: Path) -> dict:
    """
    Load registry from JSON file.

    Args:
        registry_path: Path to registry file

    Returns:
        Registry data dict

    Raises:
        RegistryError: If file is corrupted or invalid
    """
    try:
        with open(registry_path) as f:
            data = json.load(f)

        # Validate version
        if data.get("version") != "1.0":
            raise RegistryError(f"Unsupported registry version: {data.get('version')}")

        return data

    except FileNotFoundError:
        # Return empty registry structure
        if "backends" in registry_path.name:
            return {"version": "1.0", "backends": {}}
        else:
            return {
                "version": "1.0",
                "builtin_benchmarks": {},
                "diy_benchmarks": {}
            }

    except json.JSONDecodeError as e:
        raise RegistryError(f"Corrupted registry file {registry_path}: {e}")
```

---

## Conflict Handling

### Duplicate Names

```python
def check_duplicate_backend(backend_name: str, registry: dict) -> None:
    """
    Check if backend name already exists in registry.

    Args:
        backend_name: Name to check
        registry: Current registry data

    Raises:
        BackendValidationError: If name exists
    """
    if backend_name in registry["backends"]:
        existing = registry["backends"][backend_name]
        location = "built-in" if existing["builtin"] else "DIY"

        raise BackendValidationError(
            f"Backend '{backend_name}' already registered ({location}).\n"
            f"Please choose a different name or remove the existing backend first."
        )
```

### Corrupted Registry Recovery

```python
def recover_corrupted_registry(registry_path: Path) -> None:
    """
    Attempt to recover from corrupted registry.

    Args:
        registry_path: Path to corrupted registry file
    """
    print(f"Warning: Registry file {registry_path} is corrupted.")
    print("Attempting to recover...")

    # Backup corrupted file
    backup_path = registry_path.with_suffix(".json.backup")
    if registry_path.exists():
        import shutil
        shutil.copy(registry_path, backup_path)
        print(f"Corrupted registry backed up to: {backup_path}")

    # Regenerate registry
    if "backends" in registry_path.name:
        print("Regenerating backends registry...")
        backends = discover_builtin_backends()
        registry = {"version": "1.0", "backends": backends}
    else:
        print("Regenerating benchmarks registry...")
        benchmarks = discover_builtin_benchmarks()
        registry = {
            "version": "1.0",
            "builtin_benchmarks": benchmarks,
            "diy_benchmarks": {}
        }

    write_registry_atomic(registry_path, registry)
    print(f"✓ Registry regenerated: {registry_path}")
```

---

## Registry Operations (Future)

These operations are not in v1.0 but documented for future implementation.

### Remove Backend

```python
def remove_backend(backend_name: str) -> None:
    """
    Remove a backend from the registry (future feature).

    Args:
        backend_name: Name of backend to remove

    Note:
        Only DIY backends can be removed, not built-ins.
    """
    local_dev = get_local_dev_dir()
    registry_path = local_dev / "backends.json"
    registry = load_registry(registry_path)

    if backend_name not in registry["backends"]:
        raise BackendNotFoundError(f"Backend '{backend_name}' not found")

    backend_info = registry["backends"][backend_name]
    if backend_info["builtin"]:
        raise BackendError("Cannot remove built-in backend")

    del registry["backends"][backend_name]
    write_registry_atomic(registry_path, registry)

    print(f"✓ Backend '{backend_name}' removed from registry.")
```

### Validate Registry

```python
def validate_registry() -> None:
    """
    Validate all entries in registry (future feature).

    Checks:
    - All registered backends/benchmarks still exist
    - All implement required interfaces
    - All problem instance files are valid
    """
    # Implementation for future release
    pass
```

### Regenerate Registry

```bash
# Future CLI command
apps-benchmark registry --regenerate
```

---

## Naming Conventions

### Snake Case to CamelCase

```python
def snake_to_camel(snake_str: str) -> str:
    """
    Convert snake_case to CamelCase.

    Args:
        snake_str: String in snake_case

    Returns:
        String in CamelCase

    Examples:
        >>> snake_to_camel("my_custom_backend")
        'MyCustomBackend'
        >>> snake_to_camel("simple_qaoa")
        'SimpleQaoa'
    """
    components = snake_str.split('_')
    return ''.join(x.title() for x in components)
```

### File to Class Naming

| File Name | Expected Class Name |
|-----------|-------------------|
| `my_backend.py` | `MyBackend` |
| `aws_braket.py` | `AwsBraket` |
| `simple_qaoa_runner.py` | `SimpleQaoaRunner` |
| `vqe_puccd_runner.py` | `VqePuccdRunner` |

---

## Summary

### Registry Lifecycle

1. **First Run**: Auto-discover built-ins, initialize registries
2. **Add DIY**: User registers custom components via CLI
3. **Validation**: Interface checks during registration
4. **Usage**: Load components from registry for execution
5. **Updates**: Atomic writes prevent corruption

### Key Design Principles

- **Auto-discovery**: Built-ins automatically registered
- **Explicit DIY registration**: Users must run `add` command
- **Fail-fast validation**: Catch errors at registration time
- **Atomic operations**: Registry updates succeed completely or not at all
- **Clear errors**: Helpful messages guide users to fix issues

### Registry Locations

- **Backends**: `<local_dev>/backends.json` (default: `~/local_dev/backends.json`)
- **Benchmarks**: `<local_dev>/benchmarks.json` (default: `~/local_dev/benchmarks.json`)
- **Local dev code**: `<local_dev>/backends/` and `<local_dev>/benchmarks/` (default: `~/local_dev/`)
- **Configuration**: `~/config_local_dev.json` (defines local_dev location)

### Future Enhancements

- Remove command for unregistering components
- Validate command for checking registry integrity
- Regenerate command for rebuilding from scratch
- Version migration for registry format updates
