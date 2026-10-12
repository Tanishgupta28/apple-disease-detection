# New four-class experiments, October 2026

All values here are measurements from this rebuild. Original internship evidence
remains in `original_work/` and `docs/legacy_experiments.md`. The prior six-class
PyTorch benchmark in `results/` has a different dataset, framework and label space.

## Protocol fixed before model comparison

Seed 42, one audited manifest, RGB 96x96, batch 16, train-only inverse-frequency
weights, Adam, sparse cross-entropy, validation-loss checkpoint selection and early
stopping. ANN: maximum 60 epochs/patience 10; CNN: 20/patience 6; ResNet50 frozen
head: 5/patience 3, final residual block fine-tuning: 4/patience 3. The configuration
was committed with the first milestone before ResNet50 training. Test scores are
reported for all three predeclared models on the same 119 images; no hyperparameters
are changed after examining those scores. This is a descriptive single-seed comparison,
not a test-driven architecture search. Best model is selected by validation loss.

| Model | Test accuracy | Macro precision | Macro recall | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|---:|
| ANN (GLCM) | 0.487395 | See CSV | See CSV | 0.443899 | See CSV |
| Custom CNN | 0.462185 | See CSV | See CSV | 0.296919 | See CSV |
| ResNet50 selected frozen head | 0.663866 | See CSV | See CSV | 0.654552 | See CSV |

Exact metrics, prediction probabilities, per-class reports and confusion matrices:
`results/four_class/model_comparison.csv` and each model's `metrics.json`.
All three use identical `manifest_sha256` values. Checkpoint hashes bind results to
saved models; compact reports are committed and `.keras` files stay in local `artifacts/`.

## What actually trained

- ANN: all 60 budgeted epochs; best checkpoint chosen by validation loss.
- CNN: early stopping after 9 epochs; best checkpoint was epoch 3. Training accuracy
  rises while validation loss worsens. This is a weak baseline, not a successful
  high-accuracy classifier. BatchNorm's small-data running statistics, overfitting,
  augmentation, and source artifacts are plausible contributors; no ablation establishes
  a single cause. Macro F1 exposes the uneven performance hidden by overall accuracy.
- ResNet50: 5 frozen-head epochs, then 4 fine-tuning epochs from the best frozen checkpoint.
  Fine-tuning did **not** improve validation loss (best head ~0.6825 versus fine-tune
  ~0.7293). Therefore the frozen-head checkpoint remains selected. Do not call the
  deployed checkpoint "fine-tuned". The fine-tuning experiment did run and its
  history/checkpoint are retained locally under `artifacts/resnet50/fine_tune/`.

CPU fit times recorded by the trainer: ANN ~15.4s, CNN ~22.8s, ResNet50 ~110.3s.
These include fit callbacks/checkpoint work and both ResNet training phases; they
exclude image loading/feature extraction, initial model construction and test evaluation.
They are neither GPU nor inference-latency measurements.

## Architecture and tensor shapes

ANN: 10 texture values -> train-adapted Normalization -> Dense(32, ReLU, L2=1e-4)
-> Dropout(0.25) -> Dense(16, ReLU) -> Dense(4, Softmax). Normalization statistics
are embedded in the saved model and exported as `scaler.json`. Feature order and
GLCM configuration are saved in metadata. No class weights are applied to test metrics.

CNN (batch dimension omitted): 96x96x3 -> augmentation -> /255 ->
Conv3x3(16)/BN/ReLU/MaxPool -> 48x48x16 -> Conv3x3(32)/BN/ReLU/MaxPool ->
24x24x32 -> Conv3x3(64)/BN/ReLU/MaxPool -> 12x12x64 -> global average pool(64)
-> Dropout(0.3) -> Softmax(4). Full parameter counts and shapes are in `architecture.txt`.

ResNet50: 96x96x3 -> augmentation -> RGB-to-BGR and ImageNet mean subtraction ->
official ImageNet backbone without the 1,000-class head -> 3x3x2048 -> global
average pool(2048) -> Dropout(0.3) -> Softmax(4). Learning rates are 1e-3 for the
head and 1e-5 for fine-tuning. Only `conv5_block3*` convolution weights unfreeze;
every BatchNormalization layer remains frozen and the backbone uses `training=False`.
The model is recompiled after trainability changes. Smoke tests confirm early weights
and BN statistics stay fixed while final-block convolution weights change.

## Texture and segmentation

GLCM quantizes luminance to 32 levels. Distances 1,2,4 pixels and angles 0,45,90,135
degrees give 12 matrices, symmetric and normalized. Mean/std across these matrices
yield ten features (contrast, homogeneity, energy, correlation, dissimilarity).
Whole-image texture is the baseline; K-means is an independent visualization.
No unvalidated "disease cluster" mask feeds classification. Background texture and
image resizing can influence features, a deliberate limitation to explain in interviews.

## Evidence limits

Only 119 test images, related-image groups within splits, historical test exposure,
unverified source labels/license, no orchard holdout, no calibration or unknown-image
detector, no multiseed confidence intervals. A confident Softmax answer does not prove
the uploaded object is an apple or the disease label is correct. Training-only
augmentation changes apparent train accuracy; it is not directly comparable to
deterministic validation/test accuracy.

Technical references:
- [Keras ResNet preprocessing](https://keras.io/api/applications/resnet/)
- [TensorFlow transfer learning and BN behavior](https://www.tensorflow.org/guide/keras/transfer_learning)
- [scikit-image GLCM example](https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_glcm.html)
