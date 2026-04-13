# QLBM Benchmark Specification

QLBM benchmark specification for a 2D advection-diffusion instance.

This benchmark is in the open category, so no solution runner is provided.

## Contents

- `qlbm_schema.json` - JSON Schema for the upstream `qlbm_instance.json` payload
- `benchmark_cases/` - checked-in instance JSON files used by apps-benchmark

## Notes

- The shipped instance is a faithful transcription of the source
  `qlbm_instance.json` payload into valid JSON.
- The schema intentionally matches that upstream payload shape.
- The case tags `qlbm` in `open_solution_algorithms` to mark it as a
  bring-your-own-solver benchmark.
- The only repo-specific addition is the optional `instance_id` field, which is
  added to checked-in cases for registry indexing in apps-benchmark.
