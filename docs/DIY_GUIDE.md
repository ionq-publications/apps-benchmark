# DIY Plugin Guide
This guide shows you how to create and register your own backends and benchmarks with apps-benchmark.

There are a set of IonQ supported and tested benchmarks, and backends built into the python package.

There is also a configurable 'local_dev' system where you can build plugins for your own benchmark or your
own backends.  Benchmarks, as you probably already know, are quantum algorithms to test under different
configurations, to compare performance.  Backends are quantum solvers (processors) that can be simulators or
actual hardware from a quantum computing vendor.

## Table of Contents
1. [Local Dev folder for Plugins](#local-dev-folder-summary)
1. [Creating a Custom Backend](#creating-a-custom-backend)
2. [Creating a Custom Benchmark](#creating-a-custom-benchmark)
3. [Common Issues and Solutions](#common-issues-and-solutions)
4. [Testing Your Plugin](#testing-your-plugin)

---

## Local Dev Folder Summary

By default, apps-benchmark uses `~/local_dev/` as the location for custom plugins.
The first `apps-benchmark` command that needs local-dev state reads `~/config_local_dev.json` and
creates it if it is missing.
That file stores the `local_dev_dir` path, which defaults to `~/local_dev/`.
Whenever apps-benchmark checks `~/config_local_dev.json`, it uses the configured `<local_dev>` folder.
If the folder specified there does not exist, commands that need local-dev storage create the folder
structure on demand.

```
~/local_dev/
├── backends/           # Your custom backends go here
├── benchmarks/         # Your custom benchmarks go here
├── backends.json       # Backend registry (auto-managed)
└── benchmarks.json     # Benchmark registry (auto-managed)
```

### Changing the Local Dev Directory Location

**Important:** When you change the `local_dev_dir` in the config file and run any `apps-benchmark` command, it will automatically:
1. Create the new directory structure at the specified location
2. Initialize fresh `backends.json` and `benchmarks.json` registry files
3. Create the `backends/` and `benchmarks/` subdirectories

This means you can easily switch between different plugin directories for different projects by changing the config file. Each directory will have its own independent set of registered plugins.

We encourage you to submit these plugins to the Github Repo for review, and if you would like, inclusion in the base Apps-Benchmark project.  

## Notes on unique ID's 
We use unique IDs to avoid mistakes or developer error here's how **--case-uuid** and **instance_id** work together:                                                               

### instance_id (JSON Field)
 - Location: Defined in benchmark case JSON files (e.g., ~/local_dev/benchmarks/chemistry/benchmark_cases/*.json)
 - Purpose: Unique identifier for each benchmark problem instance
 - Format: Short alphanumeric string (e.g., "1bbca30e", "610cfb55", "abc12345")
 - Auto-generation: If not provided, automatically generates an 8-character hex UUID (apps_benchmark/primitives/benchmark_case.py:36)
 - Required: Yes - the registry validation enforces that every benchmark case must have a valid instance_id (apps_benchmark/core/registry.py:99-100)

### case-uuid (CLI Flag)
 - Location: CLI command argument for apps-benchmark run
 - Purpose: Tells the CLI which specific benchmark problem instance to execute
 - Value: Must match an instance_id from a benchmark case JSON file
 - Usage: apps-benchmark run --backend=<backend_name> --case-uuid=<instance_id>

### Key Points
 - Mutual exclusivity: Either --case-uuid OR --category is required (apps_benchmark/cli.py:648-650)
 - UUID vs Category:
- --case-uuid: Runs a single specific benchmark instance
- --category: Runs all benchmarks in a category (e.g., "chemistry")  

--- 

## Creating a Custom Backend
Please see [DIY_BACKEND](DIY_BACKEND.md) for specifics on making a DIY Backend. 

We strongly recommend the first backend you make is just a re-named copy of `qiskit_aer_sim_backend.py` 
as a way to test if the plugin system overall is working well for you. If not you can reach out to 
App Benchmark Support at apps-benchmark-support@ionq.co 

## Creating a Custom Benchmark 

A benchmark is a set of **Benchmark Cases** you want to run, to quantify how quantum solvers (or QPU's)
work on a specific algorithm, to review or compare performance. Please see [DIY_BENCHMARK](DIY_BENCHMARK.md) 
for specifics on making a DIY Backend. 

---

## Common Issues and Solutions

### Issue 1: Class Name Mismatch

**Error:**
```
BackendValidationError: Based on 'my_backend' expected class 'MyBackend'
but did not find it. Backend not registered.

Available classes in my_backend.py: ['WrongName']
Did you mean to name your class 'MyBackend'?
```

**Solution:**
- File: `my_backend.py` → Class must be: `MyBackend`
- File: `my_custom_backend.py` → Class must be: `MyCustomBackend`
- Follow CamelCase conversion from snake_case filename

### Issue 2: Missing Abstract Methods

**Error:**
```
BackendValidationError: Cannot instantiate abstract class MyBackend
with unimplemented abstract methods: 'run'
```

**Solution:**
Implement all required methods from `AbstractBackend`:
- `name(self) -> str`
- `run(self, circuits, shots, job_name) -> Tuple[List[dict], str, dict]`

For `AbstractAlgoRunner`, implement:
- `name(self) -> str`
- `setup_algo_inputs(self, benchmark_case) -> Tuple[Any, ...]`
- `execute_benchmark_algo(self, algo_inputs, backend, shots, **kwargs) -> Any`
- `compute_merit_figures(self, algo_output, benchmark_case) -> Dict[str, Any]`

### Issue 3: File Not Found

**Error:**
```
FileNotFoundError: Backend file not found: ~/local_dev/backends/my_backend.py
Please create the file before registering.
```

**Solution:**
Ensure the file exists at the expected location with the correct name.

### Issue 4: Wrong Directory Structure

**Error:**
```
FileNotFoundError: Runner file not found:
~/local_dev/benchmarks/chemistry/algorithms/my_vqe_runner.py
```

**Solution:**
Benchmark files must follow this structure:
```
~/local_dev/benchmarks/
└── {category}/
    ├── algorithms/
    │   └── {name}_runner.py
    └── benchmark_cases/
        └── {instance}.json
```

---

## Testing Your Plugin

### Test Your Backend

1. **Self-test** (basic connectivity check):
```bash
apps-benchmark run --self-test --backend=my_simulator_backend
```

2. **Run with built-in benchmarks**:
```bash
apps-benchmark run --backend=my_simulator_backend --case-uuid=610cfb55
```

3. **Category run**:
```bash
apps-benchmark run --backend=my_simulator_backend --category=chemistry
```

### Test Your Benchmark

1. **Single instance**:
```bash
apps-benchmark run --backend=qiskit_aer_sim_backend --case-uuid=abc12345
```

2. **Category run** (if you have multiple instances):
```bash
apps-benchmark run --backend=qiskit_aer_sim_backend --category=chemistry
```

---

## Advanced Topics

Once you have your basic benchmark working, you can extend it with more sophisticated patterns. This section covers configuration management, multi-algorithm benchmarks, and integration with external libraries.

### Custom Backend with Configuration

```python
class MyConfigurableBackend(AbstractBackend):
    def __init__(self, config_file: str = None):
        if config_file:
            with open(config_file) as f:
                self.config = json.load(f)
        else:
            self.config = {"default": True}

    def name(self) -> str:
        return "my_configurable"

    def run(self, circuits, shots=1000, job_name=None):
        # Use self.config in your logic
        ...
```

### Benchmark with Multiple Algorithms

Your problem instance can list multiple algorithms:

```json
{
  "solution_algorithms": ["my_vqe", "my_qaoa", "my_custom"],
  ...
}
```

Each algorithm needs its own runner file in `algorithms/`.

**Selecting which algorithm to run:**

By default, the CLI uses the first algorithm in the list. To run with a different algorithm, use the `--algorithm` flag:

```bash
# Uses the first algorithm: my_vqe
apps-benchmark run --backend=qiskit_aer_sim_backend --case-uuid=abc12345

# Explicitly select my_qaoa
apps-benchmark run --backend=qiskit_aer_sim_backend --case-uuid=abc12345 --algorithm=my_qaoa

# Run all benchmarks in a category with a specific algorithm
# Benchmarks that don't support the specified algorithm will be skipped
apps-benchmark run --backend=qiskit_aer_sim_backend --category=chemistry_x --algorithm=my_custom
```

The CLI will display which algorithms are available and which one is being used:
```
  Available algorithms: my_vqe, my_qaoa, my_custom
  Using algorithm: my_qaoa
```

### Using External Dependencies

Your plugin can use any Python packages you have installed:

```python
import my_custom_simulator
import my_optimization_library

class MyBackend(AbstractBackend):
    def __init__(self):
        self.simulator = my_custom_simulator.Simulator()
    ...
```

---

## Reference Documentation

- **[API_REFERENCE.md](API_REFERENCE.md)** - Complete interface specifications
- **[DIY_BACKEND.md](DIY_BACKEND.md)** - Full backend implementation guide
- **[DIY_BENCHMARK.md](DIY_BENCHMARK.md)** - Full benchmark implementation guide
- **[DIY_REGISTRY.md](DIY_REGISTRY.md)** - Registry system internals

---

## Getting Help

If you encounter issues:

1. Verify file and class naming conventions
2. Ensure all abstract methods are implemented
3. Contact App Benchmark Support at apps-benchmark-support@ionq.co  
