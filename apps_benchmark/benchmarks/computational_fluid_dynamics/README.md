# QLBM Benchmark Specification

QLBM benchmark specification for a 2D advection-diffusion instance.

This benchmark is data-first, so no solution runner is provided.

## Contents

- `qlbm_schema.json` - JSON Schema for the public QLBM BenchmarkCase
- `benchmark_cases/` - instance JSON files

## Notes

- The shipped instance is a faithful transcription of the source QLBM payload
  into valid JSON.
- The only repo-specific addition is `instance_id`, which is required for
  registry indexing in apps-benchmark.
