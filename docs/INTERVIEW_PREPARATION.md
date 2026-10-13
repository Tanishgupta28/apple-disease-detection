# Apple Disease Detector: interview preparation

Read this before the October 13, 2026 interview. It describes code and measurements
that actually exist. Suggested explanations describe the project; they are not a
claim that you personally performed every new rebuild step during the internship.

## First: know exactly what the evidence supports

There are three distinct bodies of work in this repository:

1. **Historical internship notebooks:** nine unchanged notebooks in `original_work/`.
   Their saved outputs show 307 training, 75 validation and 120 test images with
   Blotch, Rot, Scab and Normal labels. Only some notebooks store genuine test scores.
   See `docs/legacy_experiments.md`; a maximum validation accuracy is not test accuracy.
2. **Prior six-class leaf project:** existing PyTorch code under `src/`, AppleLeaf9
   data documentation and results directly under `results/`. Its reported 97.07%
   score belongs to that separate dataset and task. It is not the new Keras score.
3. **New October 2026 rebuild:** `apple_disease/`, TensorFlow/Keras, four historical
   fruit labels, an audited 270/70/119 split, ANN/CNN/ResNet50 training, standalone
   K-means and GLCM, YOLO workflow and a Streamlit demo. These are newly run experiments.

The recovered four-class classification images include **fruit**, not a verified
leaf dataset. There are **502 files, 472 unique images, 459 included images**.
The resume's "3,000+ labelled leaf images" cannot be established for this classification
experiment. State that clearly if asked. Separate older YOLO assets contain leaf boxes
with Healthy/Black rot/Scab/Rust labels; that does not establish that the fruit ANN/CNN
was trained on 3,000 leaves. Do not merge evidence from unrelated tasks to justify a claim.

## A 60-second explanation

“This project explores apple disease image analysis using two kinds of classifiers.
The ANN uses hand-designed texture measurements from a gray-level co-occurrence
matrix. The CNN learns image features directly, and ResNet50 reuses ImageNet features.
The rebuild audits duplicates and uses the same held-out split for all three models.
The recovered four-class data contains fruit photographs: 472 unique images, with
459 retained after exclusions. On 119 test images, ANN accuracy is 48.74%, CNN is
46.22%, and selected ResNet50 is 66.39%. K-means illustrates color clustering;
it does not automatically identify lesions. YOLO is a separate bounding-box workflow.
The main lesson is that data provenance, leakage checks, and honest evaluation matter
as much as architecture. The recovered evidence does not verify the resume's
3,000-plus leaf-image claim.”

## A two-minute explanation

“The aim is to study how image features can help classify apple disease categories.
The code now has a reproducible four-class fruit experiment using Blotch, Rot, Scab,
and Normal. I would distinguish the original internship notebooks from the newer
rebuild because they are different experiments.

The recovered dataset had 502 files. Hashes found 472 unique images. We excluded
redundant copies, conflicting-label pairs, and training images related to original
test images. The final split is 270 training, 70 validation, and 119 test images.
Related photographs stay together, and the manifest records membership and hashes.
This limits known leakage, although an independent orchard dataset would be stronger.

The first baseline converts an image to grayscale, quantizes it to 32 levels, and
measures neighboring-pixel texture using GLCM. Ten numerical features feed a small
ANN, with normalization fitted on training features only. The second model is a
three-block CNN that learns spatial filters from RGB pixels. The third uses an
ImageNet-pretrained ResNet50 backbone. Its head is trained first; then the final
residual block is fine-tuned at a lower learning rate with BatchNorm frozen.

We select checkpoints using validation loss. Fine-tuning did not improve validation
loss, so the chosen ResNet50 is the frozen-backbone checkpoint. Its test accuracy
is 66.39%, compared with 48.74% for ANN and 46.22% for the small CNN. Per-class metrics
and macro F1 reveal errors that a single accuracy number hides.

The app shows model probabilities, texture values and K-means color clusters.
YOLO separately handles real bounding boxes, with dataset checks and a Roboflow
import workflow. Custom disease detection has not been established as a verified
new benchmark. The main limitations are small, imperfectly sourced data, fruit versus
leaf scope, historical test exposure, and lack of independent field validation.”

## Beginner vocabulary and an end-to-end walkthrough

An **image** is a grid of numbers. RGB stores red, green and blue values per pixel,
normally from 0 to 255. A **tensor** is a multidimensional array. A batch of 16 RGB
images resized to 96x96 has shape `(16, 96, 96, 3)` in TensorFlow's channels-last format.
A **label** is the expected category. A **feature** is a useful measurement, either
designed by a person or learned from data. A **parameter** is a number adjusted by training.

Follow one image through the project:

1. `data.py` reads its folder label: `Blotch_Apple -> 0`, `Rot_Apple -> 1`,
   `Scab_Apple -> 2`, `Normal_Apple -> 3`. This order is explicit, not alphabetical.
2. `preprocessing.py` decodes the image, applies EXIF orientation, converts to RGB,
   and rejects invalid or oversized input. Data auditing also stores content hashes.
3. Exact copies are removed. Similar-image components are kept in one split.
   Conflicting-label components are excluded. Original Test has priority over Train.
4. Remaining original training groups are split with seed 42 into train/validation,
   separately within each class. Exact membership is saved in `manifest.csv`.
5. All models use 96x96 inputs. ANN extracts GLCM values; CNN scales pixels by 1/255;
   ResNet50 reverses RGB to BGR and subtracts ImageNet means.
6. Training sees labels and adjusts parameters. Adam uses gradients of sparse
   cross-entropy. Class weights increase the influence of underrepresented classes.
7. Validation is observed after epochs. Its loss controls checkpoint saving and early
   stopping; no gradients or normalization fitting use validation images.
8. The best checkpoint is reloaded. Test images are processed deterministically.
   Probabilities, confusion matrices and metrics are saved, with model and manifest hashes.
9. The app loads the saved model and matching metadata, follows the same preprocessing,
   and displays four probabilities. K-means and GLCM tabs offer separate inspection.

## Understand each architecture

### ANN: ten measurements to four probabilities

`10 features -> Normalization -> Dense(32, ReLU) -> Dropout(0.25) -> Dense(16, ReLU)
-> Dense(4, Softmax)`.

A Dense neuron computes a weighted sum plus a bias, then an activation function.
ReLU returns `max(0, x)`. The first hidden layer uses a small L2 penalty, discouraging
large weights. Dropout randomly disables some activations during training to reduce
dependence on particular neurons. At inference dropout is inactive. The model has
969 total parameters, including normalization state. Its compactness makes it a useful
texture baseline, but it cannot recover spatial information omitted from its ten inputs.

### CNN: learn local patterns directly

`96x96x3 -> training augmentation -> /255 -> Conv(16)/BN/ReLU/Pool -> 48x48x16
-> Conv(32)/BN/ReLU/Pool -> 24x24x32 -> Conv(64)/BN/ReLU/Pool -> 12x12x64
-> global average pool -> 64 -> Dropout(0.3) -> Dense(4, Softmax)`.

A convolution slides a small learned filter over an image. Reusing its weights helps
recognize a local pattern in different positions. Pooling reduces spatial size.
BatchNormalization stabilizes activations, while its moving averages are used at inference.
Global average pooling turns each feature map into one average instead of creating a huge
flattened vector. This CNN has 24,180 total parameters. Small does not guarantee good accuracy.

### ResNet50: reuse features, then cautiously adapt

The backbone is Keras ResNet50 with official ImageNet weights and `include_top=False`.
At 96x96 input its final feature maps are 3x3x2048. Global averaging yields 2048 numbers,
followed by dropout and the four-class Softmax head. Total parameters: 23,595,908.

A residual block learns a transformation that is added to a shortcut of its input.
This helps gradients flow through a deep network. ImageNet supplies reusable features;
it does not supply apple disease diagnoses. Phase one freezes the backbone and trains
the new head at learning rate 0.001. Phase two reloads the best head checkpoint,
unfreezes only `conv5_block3*` convolution layers, and uses learning rate 0.00001.
All backbone BN layers stay frozen; the backbone call uses `training=False`.

Training from scratch starts with random parameters. Transfer learning starts with
features learned on a different task. Fine-tuning updates some of those pretrained
features. Here fine-tuning ran but the frozen model remained better by validation loss.

### YOLO: boxes and categories

The separate workflow uses Ultralytics YOLO11n. A detection model predicts candidate
bounding boxes and class scores using image features at multiple spatial scales.
YOLO11's normal detection pipeline uses nonmaximum suppression to remove strongly
overlapping redundant predictions. Implementations vary by YOLO version; do not claim
all versions use exactly the original YOLO grid formula or identical loss functions.

A detection label includes `class_id center_x center_y width height`, normalized by
image dimensions. The workflow accepts real annotation exports. It never creates
full-frame boxes from image classification labels. The generic pretrained example
predicted “cake” on an apple image; this is a recorded inference failure, not evidence
of disease detection. Custom disease detection training/evaluation is incomplete.

## Numerical K-means example

Suppose grayscale pixel values are `[10, 12, 14, 200, 202, 204]`, and K=2 starts with
centers 10 and 200. Assign each number to its nearest center: the first cluster is
`[10,12,14]`, and the second is `[200,202,204]`. Recompute centers as means: 12 and 202.
Reassigning makes no change, so the algorithm has converged.

The implementation clusters three-dimensional Lab colors instead of one-dimensional
gray values. It minimizes within-cluster squared distances, uses ten initializations,
and fits at most 10,000 seeded sampled pixels before assigning every pixel. Lab separates
lightness from two color axes, making color distance more interpretable than raw RGB.
No labels are supplied: K-means is unsupervised. A dark cluster could be a shadow,
background, text watermark, or lesion. K=2/3/4 examples demonstrate that ambiguity.

## Numerical GLCM example

Consider this two-level image and rightward neighboring pixels at distance one:

```text
0 0 1
0 1 1
```

The four directed pairs are `(0,0), (0,1), (0,1), (1,1)`. Their count matrix is
`[[1,2],[0,1]]`. The symmetric option adds its transpose, giving `[[2,2],[2,2]]`.
Normalize by eight: every entry in the 2x2 probability matrix is 0.25.

- Contrast is `sum((i-j)^2 * P[i,j])`: off-diagonal terms give **0.5**.
- Dissimilarity uses `abs(i-j)` instead of the square: also **0.5** here.
- Homogeneity is `sum(P[i,j]/(1+(i-j)^2))`: `0.25+0.125+0.125+0.25 = 0.75`.
- Energy is `sqrt(sum(P[i,j]^2))`: `sqrt(4*0.25^2) = 0.5`.
- Correlation measures dependence between the paired gray values after centering and
  scaling by their standard deviations. This symmetric uniform example has correlation **0**.

For a constant image, only one diagonal cell has probability one: contrast and
dissimilarity are zero; homogeneity and energy are one. Correlation is mathematically
undefined when variance is zero; scikit-image's convention returns one. The tests
verify that finite behavior. Our real feature extractor uses 32 quantization levels,
three distances, four angles, and the mean/std of each of five properties: ten features.
The toy checkerboard/gradient illustration is educational data, never disease training data.

## Read the actual results correctly

| Model | Test accuracy | Macro precision | Macro recall | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|---:|
| ANN | 48.74% | 0.4557 | 0.4530 | 0.4439 | 0.4795 |
| CNN | 46.22% | 0.2390 | 0.4173 | 0.2969 | 0.3358 |
| ResNet50 (frozen head selected) | 66.39% | 0.6769 | 0.6461 | 0.6546 | 0.6703 |

All rows use the same 119-image test set. ResNet50 has validation accuracy 81.43%
and selected validation loss 0.6825; the gap to test is a warning about sample and
distribution variability. Fine-tuning's minimum validation loss was about 0.7293,
so it was not selected. The CNN's macro F1 is much lower than accuracy because its
performance is uneven across classes. Open the confusion matrices and per-class reports.
Specifically, the CNN predicts no Scab or Normal test images at all (zero recall for
both classes). ResNet50 identifies 31/37 Rot images, but confuses 11/30 Blotch images
with Scab. These are concrete failure patterns you can explain without speculation.

Accuracy counts all correct predictions divided by all images. Precision asks how
often predictions of a class are correct. Recall asks how many actual examples of that
class were found. F1 is their harmonic mean. Macro F1 weights each class equally;
weighted F1 weights each class by test support. A row in our confusion matrix is a
true class, a column a predicted class. Neither probabilities nor these metrics
establish expert disease diagnosis, independent field performance, or a causal explanation.

## Interview questions and understandable answers

### 1. What problem does this project solve?

It explores classification of apple fruit photographs into four inherited labels and
compares feature-based and image-based learning. The demo helps inspect the model's
output. It is a learning/research prototype, not a validated diagnostic product.
Leaf detection is handled as a separate experimental workflow with its own label space.

### 2. Why does the resume say leaves if this dataset shows fruit?

The recovered files and the historical folder names establish a fruit-class experiment.
The leaf-only description is not supported for that dataset. Explain the discrepancy
directly and distinguish the separate leaf assets. Do not rebrand fruit photographs
as leaves to make the wording fit; the audit is an improvement in evidence quality.

### 3. Can you verify the 3,000-image statement?

Not for the rebuilt fruit classifier: there are 502 files and 472 unique decoded images.
Only 459 remain in the new split. Additional leaf/detection assets belong to another
task and label set. Augmented copies are not new independent originals. A defensible
claim needs the relevant manifest and actual model training evidence together.

### 4. What is the difference between a feature and a label?

A feature is an input measurement, such as GLCM contrast or a learned convolutional
activation. A label is the expected output category. ANN features are extracted from
pixels; labels come from folders. Labels must not leak into features, filenames,
preprocessing decisions or splits in ways that give the model the answer.

### 5. Why use four classes and an explicit order?

They are the historical fruit labels. The index order is Blotch=0, Rot=1, Scab=2,
Normal=3, stored in code and metadata. The final layer's four positions are meaningless
without this mapping. Automatically sorting folder names would give a different order
and could display the wrong disease even with correct model scores.

### 6. Why separate training, validation and testing?

Training updates parameters. Validation chooses checkpoints and stopping points.
Test estimates performance after those choices. Repeatedly changing the model because
of test results effectively turns test into another validation set. This project
reports three predeclared models without tuning their settings after test inspection.

### 7. What is data leakage here?

An example is a resized copy of the same photograph appearing in training and testing.
The model could memorize it instead of learning transferable patterns. Another example
is fitting a feature scaler using all data. The pipeline addresses both with grouped
image splitting and train-only normalization, but cannot detect every source relationship.

### 8. How do exact and near duplicates differ?

Exact byte hashes detect identical files. Decoded-pixel hashes also detect identical
pixels stored differently. Perceptual hashes find similar visual patterns despite some
resizing, compression, rotation or reflection. Perceptual matches can be false positives,
so they are candidate relationships rather than definitive proof of image identity.

### 9. Why preserve the original test partition?

It avoids casually redistributing historically held-out files and makes the new audit
traceable. When a training image is related to an original test image, the training
image is excluded. The test set is still historically exposed, so an independently
collected orchard dataset would give stronger evidence of generalization.

### 10. Why are the split ratios not exactly 80/20 within Train?

Related-image groups cannot be divided. The seeded algorithm chooses complete groups
per class until it reaches the validation target. A group may carry the count slightly
past the target. Preventing a known duplicate relationship across splits is more
valuable than forcing an exact percentage by separating related examples.

### 11. What is class imbalance, and how is it handled?

Some labels have more training examples. The loss weight for class c is N/(4*n_c),
computed only from training counts. This gives rarer classes more influence during
optimization. Validation and test retain their natural distribution and use unweighted
metrics; macro F1 makes minority-class failures easier to see.

### 12. What preprocessing is shared?

All paths decode safely, apply EXIF orientation, and convert to RGB. Classification
uses bilinear resizing to 96x96. Each model then receives its own expected representation.
The inference helper repeats the same resize and metadata-defined feature extraction.
Using a different channel order or scaling at inference would invalidate the learned mapping.

### 13. Why 96x96 instead of 224x224?

The verified environment is CPU-only and the dataset is small, so 96x96 permits a
bounded, reproducible comparison. ResNet50 without its top accepts that size. The
tradeoff is lost fine detail, especially small lesions. A higher resolution is a
possible future experiment, not an improvement measured in this rebuild.

### 14. What is augmentation and when is it used?

It creates randomized views during training: horizontal flips, modest rotations and
zoom. Keras random layers are inside the CNN/ResNet graphs and inactive for inference,
validation and test. Augmentation can help robustness but does not increase the number
of unique source images or justify a larger dataset claim.

### 15. How does K-means work?

Choose K centers, assign each pixel feature to its nearest center, update each center
to the mean of assigned features, and repeat. Different starting centers can produce
different solutions, so this code uses a seed and ten initializations. It minimizes
color-distance variation within clusters; it does not optimize lesion detection accuracy.

### 16. Why Lab color instead of raw RGB for clustering?

Lab separates lightness and two color axes and makes distances more closely related
to perceived color difference. The code uses OpenCV's float Lab conversion. It does
not add spatial coordinates, so distant pixels with similar colors can share a cluster.
That is useful for inspection but insufficient for complete object segmentation.

### 17. How do you know which cluster is diseased?

The code does not make that claim. Cluster IDs are arbitrary and unsupervised. Dark
pixels could belong to lesions, shadows, text or background. Establishing a disease
mask would require a justified selection method and independently reviewed pixel masks.
No lesion ground truth exists here, so no lesion IoU or segmentation accuracy is reported.

### 18. What does GLCM measure?

It counts how frequently a gray value occurs beside another at a chosen offset.
Normalize those counts to obtain probabilities, then summarize the matrix. This
describes texture relationships rather than just average brightness. Multiple distances
and angles provide several views of texture at different spacings and directions.

### 19. Why quantize to 32 gray levels?

An unquantized 256-level GLCM has 65,536 cells for each offset, often sparse on a small
image. Thirty-two levels need 1,024 cells, reducing memory and sensitivity to tiny
intensity changes. Quantization sacrifices detail. The choice is explicit in configuration
and metadata; it is not claimed to be an optimized biological parameter.

### 20. What do the five GLCM properties mean?

Contrast emphasizes large neighbor differences by squaring them. Dissimilarity uses
absolute differences. Homogeneity is higher when mass lies near the diagonal, meaning
neighbors often look alike. Energy is high for concentrated, orderly co-occurrences.
Correlation describes statistical dependence between paired values, with a special
constant-image convention in the library.

### 21. Why are there ten ANN features instead of five?

Each property is computed for 3 distances times 4 angles. The code keeps its mean
and standard deviation over those 12 values. Five properties times two summaries
gives ten numbers. Mean captures typical texture; standard deviation captures variation
across the chosen offsets, but averaging discards specific directional identities.

### 22. Why is feature scaling important for the ANN?

Contrast may be much larger numerically than energy. Without scaling, magnitude alone
can dominate optimization. Keras Normalization stores per-feature training mean and
variance and applies the same transformation later. The layer is adapted before training
using only the training feature matrix; saved metadata also names the feature order.

### 23. How is ANN different from CNN?

This ANN sees a ten-number summary and uses fully connected layers. This CNN sees
the image grid and learns local filters with shared weights. Both are neural networks;
“ANN versus CNN” here means a feature-based dense baseline versus a convolutional
architecture. A CNN can represent patterns that the hand-designed summary discards.

### 24. What is a convolutional filter?

A small learned weight array slides over the input and combines a neighborhood into
an activation. A filter can respond to edges, colors or increasingly complex patterns.
Its weights are learned through loss gradients, not manually specified as “disease
filters.” Weight sharing makes convolution more efficient than a dense connection per pixel.

### 25. Why ReLU, pooling and global average pooling?

ReLU adds nonlinearity, allowing layers to represent more than a linear transformation.
Max pooling reduces spatial size and keeps strong local responses. Global average
pooling summarizes each final channel with one value. It produces a small classifier
head and avoids the many parameters that flattening 12x12x64 into dense layers would need.

### 26. What are Softmax and cross-entropy?

Softmax converts four scores to nonnegative values that sum to one. Cross-entropy
penalizes low probability on the correct label; for a single true class it is essentially
minus the log of that probability. Sparse cross-entropy accepts integer labels 0–3,
so this code does not need one-hot label vectors.

### 27. What do backpropagation and Adam do?

Backpropagation computes how each trainable parameter affects loss using the chain
rule. Adam uses those gradients and running gradient statistics to update weights.
The learning rate controls update size. Neither guarantees a global optimum, and a
lower training loss alone does not establish better performance on unseen photographs.

### 28. What is overfitting and how is it addressed?

Overfitting means learning patterns too specific to training data. Symptoms include
improving training performance while validation worsens. This code uses dropout,
small model capacity, training augmentation, an ANN L2 penalty, and early stopping.
These reduce risk but cannot repair unreliable labels, a tiny test set, or a biased data source.

### 29. Why did the CNN perform poorly?

The measured result is 46.22% accuracy and macro F1 0.2969. Training accuracy rose
while validation loss deteriorated. Small data, batch-normalization statistics, weak
representation and background shortcuts are possible factors. No controlled ablation
isolated the cause, so describe them as hypotheses rather than claiming a proven explanation.

### 30. What is BatchNormalization doing?

During CNN training it normalizes activations using batch statistics and updates
moving estimates. At inference it uses those estimates. With a pretrained backbone,
small new batches can damage useful pretrained statistics. The ResNet workflow freezes
BN and explicitly calls the backbone in inference mode while still allowing selected
convolution weights to receive gradients during fine-tuning.

### 31. Why use transfer learning on this dataset?

A small dataset may not support learning a large vision model from random weights.
ImageNet pretraining gives reusable visual representations. The new head learns
how those features relate to the four fruit labels. Transfer can still fail when the
new domain differs greatly or the model relies on irrelevant features; it needs measured validation.

### 32. How does a residual connection help?

Instead of forcing stacked layers to learn a complete transformation, a residual
block adds their output to a shortcut. This provides a path for information and
gradients through a deep network. It makes optimization easier in many settings;
it does not make ResNet automatically the best model for every small dataset.

### 33. Why recompile after unfreezing layers?

The training function and optimizer must reflect the new set of trainable variables
and the new learning rate. `fine_tune()` sets the backbone trainable, restricts it
to the final residual block while excluding BN, then recompiles with a fresh Adam
optimizer. Without careful reconfiguration, the intended layers may not be updated correctly.

### 34. Did fine-tuning improve the result?

It did not improve the selection metric. The frozen head reached validation loss
about 0.6825; fine-tuning's best was about 0.7293. Therefore the saved selected model
is the frozen one. This is a completed, unsuccessful improvement experiment—useful
evidence that a plausible method can fail on a particular small dataset.

### 35. How are checkpoint selection and model comparison protected?

Checkpoints are chosen by validation loss. All experiments share a manifest hash,
and exported metrics include the evaluated checkpoint hash. The report generator
rejects mismatched splits or changed checkpoints. Manifest hashing normalizes line
endings so a Windows/Linux checkout does not appear to change the data membership.

### 36. What is precision versus recall? Give an example.

If a model predicts Scab for 20 images and 15 are actually Scab, precision is 15/20=75%.
If the test set contains 30 Scab images and it finds 15, recall is 15/30=50%.
Precision concerns the reliability of a positive prediction; recall concerns how many
true cases are missed. They answer different questions and must be interpreted together.

### 37. Why macro F1 as well as accuracy?

Accuracy counts each image equally, so common classes can dominate it. Macro F1
averages classwise F1 equally, exposing neglect of smaller classes. Weighted F1 uses
class support instead. The CNN's lower macro F1 highlights its uneven predictions;
look at the confusion matrix before concluding that a superficially similar accuracy means equal usefulness.

### 38. What is classification versus detection versus segmentation?

Classification produces a category for an entire image. Detection returns categories
and bounding boxes for objects. Segmentation labels pixels or regions. This project's
K-means output is unsupervised color segmentation, not validated disease segmentation.
The four-class Softmax model produces no object coordinates; Grad-CAM or a crop would
not automatically turn it into an annotated object detector.

### 39. What information does a YOLO annotation contain?

For a 200x100 image, a box centered at (100,50) with width 80 and height 40 becomes
`class_id 0.5 0.5 0.4 0.4`. These values encode location and size. The validator checks
five finite numbers, legal class IDs, normalized values, positive dimensions and
edges inside the image. It also requires explicit split membership and label files.

### 40. Why not use a full-image box for every classification image?

The classification label says what the image contains, not the object's actual
location. Assigning the entire frame manufactures supervision and cannot establish
localization quality. The validator flags a dataset overwhelmingly filled with such
boxes for review. A real object can fill a frame, so that is a conservative screening
rule, not a mathematical proof of falsification.

### 41. What are IoU and mAP?

Intersection-over-union is the overlapping area of predicted and ground-truth boxes
divided by their union. Average precision summarizes precision-recall behavior over
confidence thresholds for a class under a matching rule. mAP averages AP over classes;
mAP@0.5 uses IoU 0.5, while mAP@0.5:0.95 averages over thresholds 0.5 to 0.95 in steps
of 0.05. They are detection metrics, not classification accuracy.

### 42. What does Roboflow contribute?

It can organize images, manage annotations, preserve dataset versions and export
YOLO-compatible files. It is not the CNN architecture and does not automatically
validate biological labels. The project documents annotation and version review,
supports local ZIP import, and optionally downloads a specific version using an
environment API key. Remote download was not verified without credentials.

### 43. How should Roboflow splits and augmentation be handled?

Group related originals before splitting. Keep augmented variants of each original
in its training group and avoid augmenting validation/test images. Record project,
version, license, annotation target and class order. The local import verifies files
and box format but cannot prove that the service's split separates orchards or source photographs.

### 44. What tests actually increase confidence?

Tests verify corrupted-image handling, class mapping, known constant-texture values,
seeded clustering, duplicate separation, normalized four-score outputs, save/reload
equivalence, preprocessing parity, frozen-layer behavior, annotation errors and missing
checkpoints. App startup and shared inference are tested. These show software contracts
work; they do not prove disease accuracy or replace held-out evaluation.

### 45. Can Softmax confidence be trusted as a probability of correctness?

Not automatically. Softmax normalizes scores relative to the known classes, even for
a completely unrelated input. A model can be confidently wrong. The app's 60% warning
is a display heuristic, not a calibrated decision threshold. Calibration needs separate
validation analysis, and unknown-image rejection needs additional design and evaluation.

### 46. Why are the raw dataset and model files absent from Git?

Raw-image redistribution rights are unverified, and large training artifacts clutter
history. Manifests, configurations, predictions, metrics and checkpoint hashes are
committed. The evaluated `.keras` files remain in local `artifacts/`. A fresh checkout
handles their absence clearly. Do not imply the older six-class release contains these new models.

### 47. What makes this experiment reproducible?

Seeded splitting and initialization, deterministic TensorFlow operations, saved class
order and preprocessing, immutable run directories, data/checkpoint hashes, histories,
and recorded package versions support reproduction. Exact floating-point results can
still vary across hardware and library versions. The actual manifest is stronger
evidence than saying only “I used an 80/20 random split.”

### 48. What would you improve first with more time?

Obtain a clearly licensed, independently collected and expert-reviewed dataset that
matches the claimed task. Split by source/fruit/orchard, preserve an external test set,
and evaluate calibration and unknown inputs. Then run targeted validation-only ablations
and multiple seeds. Merely enlarging the network is unlikely to resolve data provenance or label errors.

## Code-specific cross-questions

### 49. Where is the ANN scaler fitted, and can inference refit it?

`models.ann(training_features)` calls `Normalization.adapt()` exactly on the provided
training matrix. `train.py` constructs that matrix from rows whose split is train.
Inference loads the serialized normalization layer and never calls adapt. `scaler.json`
is an inspectable copy of statistics; inference uses the saved layer so no separate
scaler file can accidentally be mismatched.

### 50. Does K-means feed the ANN?

No. In this implementation ANN features come from the whole resized image. K-means
is a separate visualization. Passing a chosen cluster into GLCM without defining
which cluster and how masks affect neighbor counts would introduce arbitrary choices.
A segmented-feature baseline could be a future controlled experiment, not something
already measured here.

### 51. Can calling the backbone with training=False stop fine-tuning gradients?

It controls behavior of layers such as BN and dropout; it is distinct from each
layer's `trainable` flag. Selected convolution weights can still receive gradients.
The ResNet test performs a training update and verifies a final-block kernel changes
while an early kernel and BN moving mean remain unchanged. That verifies the exact intended behavior.

### 52. Why is selected_phase “head” even though a fine_tune directory exists?

The directory preserves the experiment; selection is conditional on validation loss.
The trainer reloads the best head, runs fine-tuning, and copies its best checkpoint
to the selected path only if its validation loss is lower. Here it was higher, so
`best.keras` remains the head-stage model. The test score refers only to that selected file.

### 53. What happens if someone modifies an image after preparation?

`load_arrays()` checks its current SHA-256 against the manifest before decoding it.
A mismatch raises an error instead of silently changing training/test input. A test
edits a fixture image and confirms rejection. Near-duplicate screening, in contrast,
is done during preparation; content hashes protect the resulting decision afterward.

### 54. What happens if an image has one color or is invalid?

GLCM returns finite constant-texture values following scikit-image's convention.
K-means refuses K clusters if the sampled image has fewer distinct colors than K.
The app shows that explanation rather than crashing. Invalid bytes and oversized
images are rejected by the shared decoder. Those cases are software input errors,
not additional disease classes.

### 55. What is the main limitation of your duplicate grouping code?

A 64-bit hash with a threshold can miss cropped/recolored copies or merge different
photographs. Transitive components can grow through chains of matches. It cannot
identify the orchard, fruit, photographer or acquisition session. The code reduces
known image-level leakage; it cannot guarantee source independence without better metadata.

### 56. Why does the app disable disease detection when generic YOLO weights exist?

Generic COCO weights contain ordinary object categories, not the trained disease
mapping. A separate registry must identify a trained disease detector with class
names and a matching checkpoint hash. The UI does not present the generic inference
demo as disease functionality. Even a correctly formatted registry still needs a
reviewed training/evaluation record before deployment claims.

## What to study tonight

Read `docs/DATASET.md`, this guide, `docs/EXPERIMENTS.md`, then the small modules
`features.py`, `models.py`, `resnet.py`, `data.py`, and `train.py` under `apple_disease/`.
Open the three confusion matrices and explain one failure pattern from each. Practice
the GLCM counting example on paper and follow a 96x96 tensor through the CNN.
Run the app and inspect probabilities on both a clear fruit image and an unrelated
image; do not treat either anecdote as a benchmark.

Completed: audited classification pipeline, all three classifiers trained/evaluated,
both ResNet training phases, K-means/GLCM examples, tests, app, YOLO validation/training
code and generic pretrained inference. Not established: a new custom disease detector
benchmark, credentialed Roboflow download, 3,000+ unique historical classification
leaf images, segmentation ground truth, calibration, or independent orchard performance.

The recovered YOLO split was actively checked and rejected: validation filename
`b5b00f6d794132d3.jpg` and test filename `00b5b00f6d794132d3.jpg` contain identical
bytes. This is a concrete example of why checking only filename overlap is insufficient.
The disease detector remains unavailable until data provenance and splitting are repaired.
