# Machine Learning Benchmarks

This category currently contains a fixed quantum convolutional neural network benchmark
for a small MNIST binary classification task.

## Cases

There are four QCNN cases:

- `qcnn_09q_digits_1_0`
- `qcnn_09q_digits_1_7`
- `qcnn_16q_digits_1_0`
- `qcnn_16q_digits_1_7`

The 9-qubit cases use `3x3` images. The 16-qubit cases use `4x4` images.

## Inputs

Each case includes:

- an evaluation image file
- an evaluation label file
- three trained ansatz circuits, one each for `X`, `Y`, and `Z`
- one parameter file for the small classical classifier

Each run uses the first 50 evaluation examples listed in the case file.

## Runtime Flow

For each image, the benchmark:

1. Resizes the image to the target shape if needed.
2. L2-normalizes the image.
3. Encodes one pixel per qubit with `Ry` rotations.
4. Applies a linear `CX` chain.
5. Appends the stored trained ansatz for `X`, `Y`, and `Z`.
6. Runs all three circuits.
7. Converts the three measurement histograms into one three-value feature vector.
8. Applies the stored `3 -> 2 -> 2` classical classifier.

That gives one prediction per image.

## Score

The score is classification accuracy on the 50-example evaluation slice.

The runner also returns:

- `confusion_matrix`
- `predictions`
- `labels`
- `observable_triplets`
- `classifier_outputs`

## What Is Precomputed

These parts are fixed in the case assets:

- the trained quantum ansatz circuits
- the classifier parameter file
- the evaluation images and labels

These parts happen at runtime:

- image resize when the asset is larger than the target shape
- image normalization
- circuit construction
- circuit execution
- observable extraction
- classifier inference
