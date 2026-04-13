# Image Loading Benchmark

This benchmark evaluates image-loading instances using shipped grayscale images
and precompiled tensor-network, matrix-product-state circuits.

The shipped benchmark surface is two image instances:

- `mnist_5`
- `imagenet_sketch_shark`

Each benchmark case represents one image and a sweep over depth-`D` `.qpy`
circuits. The benchmark score is the lowest MSE observed across the configured
depth sweep, not the score from any single depth.

## Shipped Assets

This repo includes the `.png` images and `.qpy` circuits needed to run the
benchmark. The circuit-generation code is not included.

The machine-readable [asset_manifest.json](./asset_manifest.json) records the
shipped asset names and SHA-256 hashes.

## Scoring

The target image is converted to grayscale pixel intensities, flattened, and
normalized with the Euclidean norm before comparison to the measured output.

Each depth circuit is measured in the computational basis. The observed
histogram is converted to a reconstructed image vector and normalized with the
same Euclidean norm convention. The per-depth MSE is then:

`sum((x_q - x_i)^2)`

where `x_q` is the reconstructed normalized intensity distribution from the
measured histogram and `x_i` is the normalized original image intensity
distribution.

The benchmark `score` is the minimum MSE across the configured depths.

## Depth Sweeps

- `mnist_5`: depths `1, 2, 3, 4, 5, 6`
- `imagenet_sketch_shark`: depths `3, 4, 5, 6, 7, 9, 11, 13`

This repo also includes a shark `D10` circuit that is not part of the benchmark
sweep.

## Shot Recommendation

The image-loading runs behind the paper used `10_000` shots for each circuit
evaluation. The shipped case metadata therefore sets
`recommended_minimum_shots_per_qc` to `10000` for both image-loading cases so
CLI runs default to that setting while still allowing config and direct CLI
overrides.
