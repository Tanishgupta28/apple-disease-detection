# Four-class rebuild status

New work is implemented under `apple_disease/` with `configs/four_class.json`.
It uses TensorFlow/Keras and the recovered historical fruit class labels.
The existing `src/`, `scripts/`, six-class results and older teaching notebooks
remain evidence of the separate PyTorch AppleLeaf9 project; their metrics must
not be attributed to this four-class implementation.

Install with `python -m pip install -r requirements-four-class.txt`.
Run tests with `python -m pytest tests/four_class -q`.
Prepare data with `python -m apple_disease.data` (only when no manifest exists).
Train with `python -m apple_disease.train --model ann --output artifacts/ann`
or `--model cnn --output artifacts/cnn`. Each run refuses to overwrite existing
artifacts. `python -m apple_disease.report` publishes compact measured evidence.

See [DATASET.md](DATASET.md) for actual counts and the fruit-versus-leaf mismatch.
Raw images and large checkpoints are excluded from Git. All new reports are under
`results/four_class/`. This work does not establish a 3,000-image leaf experiment.

K-means example: `results/local_examples/kmeans_example.png`, generated from
`Train/Blotch_Apple/1.jpg`. For K=2, color clusters merge background with fruit;
K=3 separates more background; K=4 additionally splits illumination on green fruit.
Dark lesions remain mixed with other dark pixels. Cluster colors have no biological
meaning. No lesion masks exist, so no segmentation accuracy/IoU is claimed.

Manual near-duplicate review: the first 12 non-exact candidate pairs were inspected
in `results/local_examples/duplicate_review.png`. They show recompressed/resized copies
of the same photographs, supporting conservative grouping. This is a sample review,
not exhaustive expert verification of all 96 candidates.

The ANN's ten GLCM inputs are property mean/std over 12 distance-angle combinations.
The published `toy_texture_examples.png` is an explicitly synthetic mathematical
example, not additional training data. Real dataset images are not redistributed
because their license has not been established.
