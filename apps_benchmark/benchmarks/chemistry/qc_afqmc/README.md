# QC-AFQMC Benchmark Specification

QC-AFQMC benchmark specification for hydrogen chain systems
(H4, H6 linear chains at 2.0 Å, STO-3G).

This benchmark is in the open category, so no solution runner is
provided.

## Contents

- `qcafqmc_schema.json` — JSON Schema for the QC-AFQMC BenchmarkCase
- `benchmark_cases/` — instance JSON files
- `electronic_structure_data/` — HDF5 integrals and orbital data
- `shadows/` — HDF5 shadow measurement archives

## Instance Structure

Each instance JSON follows the `BenchmarkCase` format:

- `data.system` — geometry, basis, electrons, active space
- `data.setup` — trial state, shadow, and AFQMC parameters
- `data.reference` — SCF, VQE, and FCI reference energies
- `data.analysis` — trace extraction, equilibration, reblocking
- `data.scoring` — merit function
- `data.artifacts` — paths to HDF5 data files
- `open_solution_algorithms` — benchmark solvers that are tagged as open and
  therefore not shipped by apps-benchmark
- `instance_id` — optional repo-specific identifier used for registry indexing

## Scoring

```
abs(result.energy_ha - data.reference.fci_energy_ha) * 627.509 kcal/mol/Ha
```

## Notes

- The analysis rule uses `reblock_by_autocorr` from
  [ipie](https://github.com/JoonhoLee-Group/ipie) after discarding
  `data.analysis.equilibration_blocks_discarded` equilibration blocks.
  Reported uncertainty is the SEM from reblocking.
- `population_control_method: "pair_branch"` follows the `ipie` default.
- In the shadow HDF5 files, `shadow_measurements` is stored as a
  JSON-encoded list of bitstrings.
