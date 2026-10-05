# Interview notes: Apple Disease Detection

Read this alongside `README.md`, `docs/methodology.md`, the measured results, and `src/models.py`. The preserved internship notebooks are in `original_work/`; their audit is `docs/legacy_experiments.md`. Never substitute an old validation score for the new test result.

## 1. Explain the project

The input is a photograph of an apple leaf. The output is one of six classes: Apple Scab, Rust, Frogeye Leaf Spot, Powdery Mildew, Alternaria Leaf Spot, Healthy. This is **image classification**: one label for the entire image. It does not locate each lesion with a bounding box, segment diseased pixels, or estimate severity. A leaf with multiple diseases may not fit the single-label assumption.

**Interview Answer**

“My NIT internship project explored apple disease classification using CNNs. I preserved and audited that work, then rebuilt a six-class leaf benchmark using AppleLeaf9. I compared a small custom CNN with MobileNetV2 and ResNet50, used two-stage transfer learning, selected the model using validation macro F1, and evaluated it on a held-out test split. I also built Grad-CAM visualizations and a Streamlit upload demo.”

## 2. Dataset and train/validation/test split

A **training set** updates model parameters. A **validation set** selects checkpoints, models and hyperparameters. A **test set** estimates performance after decisions are fixed. Here seed 42 gives approximately 70/15/15 partitions. After the data audit there are 9,378/2,014/2,013 images. Seeded random allocation is reproducible; it does not magically eliminate bias or leakage.

**Data leakage** means evaluation benefits from information unavailable in a true unseen example. An original leaf and its flipped copy in different splits are leakage. Split related source images first, then augment only training images. SHA-256 checks exact bytes; decoded pixel hashes detect identical pixels; perceptual hashes group close visual variants. We quarantined 56 similar images with contradictory labels. No orchard/leaf acquisition IDs are provided, so an independent orchard benchmark remains a limitation.

**Interview Answer**

“I separated learning, model selection and final reporting. Training updates weights, validation selects the best checkpoint, and the test set is used after selection. I audited readable files and duplicates, kept perceptually related images in the same split, and applied augmentation only afterward to training. This reduces leakage, although missing orchard and leaf IDs mean I cannot prove every related image was detected.”

## 3. CNN, convolution, kernels and feature maps

A **CNN** learns spatial features using shared filters. A **convolution** slides a small learned **kernel** over an image or feature map and forms weighted sums. A 3×3 kernel sees a local neighborhood. Weight sharing reduces parameters compared with connecting every pixel to every neuron. Different kernels can respond to edges, colors or textures; deeper features combine local patterns.

A **feature map** is the spatial output of a filter. Multiple output channels represent multiple learned responses. A layer with input channels C_in and output channels C_out uses C_out × C_in × kernel_height × kernel_width weights, plus biases if present. Later maps become smaller through pooling/downsampling. This does not mean we know every channel's precise biological meaning.

**Interview Answer**

“A convolutional network learns local patterns with small filters that slide across an image. Each filter produces a feature map, and the same filter weights are reused at every location. Early layers can capture edges and color changes; later layers combine them into richer patterns. This is useful for leaf symptoms because spots and texture changes are spatial patterns, but I still need error analysis to check what the model actually learns.”

## 4. ReLU, pooling and global pooling

**ReLU** is max(0,x); it provides nonlinearity and a simple gradient for positive activations. Without nonlinearities, stacking linear layers cannot represent the same range of nonlinear decisions.

**Max pooling** retains the strongest response in a neighborhood and reduces feature-map resolution. Too much downsampling can lose tiny lesion information. **Global average pooling** averages each final channel across spatial positions. It avoids the enormous classifier created by flattening all locations. The new baseline has 316,198 parameters; the old baseline had 8,530,596 and only 307 training images.

**Interview Answer**

“My custom CNN has four convolutional blocks with batch normalization, ReLU, pooling and dropout. ReLU introduces nonlinearity, while pooling reduces spatial resolution. I use global average pooling before the six-class head to keep the classifier small. The old model's flattened dense layer dominated its parameter count; the new baseline is much smaller and easier to explain.”

## 5. Batch normalization, dropout and overfitting

**Batch normalization** standardizes intermediate activations using learned scale/shift and running statistics. Training and evaluation behave differently. A frozen pretrained model must not keep changing its batch-normalization statistics unintentionally. The transfer models keep pretrained batch-norm statistics fixed during head training and fine-tuning.

**Dropout** randomly disables activations during training to discourage dependence on individual units. It is disabled at inference. Spatial dropout in the baseline drops whole feature channels.

**Overfitting** means fitting training-specific patterns that generalize poorly. Increasing training accuracy while validation metrics stagnate is a warning. Smaller heads, augmentation, dropout, early stopping and transfer learning can help. These are tools, not guarantees.

**Interview Answer**

“Overfitting happens when a model learns training-specific patterns rather than transferable ones. I monitor training and validation curves, use a smaller classifier, apply mild training augmentation, and stop when validation macro F1 stops improving. Dropout reduces dependence on particular activations. For transfer learning I also freeze pretrained batch-normalization statistics, so the small task dataset does not accidentally overwrite them.”

## 6. Softmax and categorical cross entropy

The classifier produces six **logits**, unrestricted scores. **Softmax** turns them into nonnegative probabilities summing to one. It is a relative distribution over the six known classes, not a reliable probability that a photograph is an apple leaf. A banana leaf can still receive high confidence.

**Categorical cross entropy** penalizes low probability assigned to the correct class: −log(p_true). PyTorch CrossEntropyLoss receives integer class labels and logits and computes stable log-softmax internally. Do not apply softmax before CrossEntropyLoss. At inference, apply softmax to display probabilities.

**Interview Answer**

“The final layer returns one score per class. Cross entropy compares those scores with the true label and provides the learning signal. In PyTorch I pass logits directly to CrossEntropyLoss because it includes log-softmax internally. I apply softmax only when displaying predictions. Its confidence is relative to the six classes and is not calibrated diagnostic certainty or a way to detect every unrelated image.”

## 7. Augmentation

**Data augmentation** generates plausible variations of a training example. We use horizontal flip, small rotation/translation/zoom and mild brightness/contrast changes. Large crops, extreme color changes or heavy blur can destroy symptoms. Augmentation does not create independent leaves or justify counting synthetic views as new original images.

The model uses a deterministic RGB resize/normalization pipeline for validation/test and upload inference. For deployment, a mismatch between training and inference preprocessing can invalidate accuracy even with the same weights.

**Interview Answer**

“I use mild augmentation to expose the model to reasonable changes in camera position and lighting. The original images are split first, so augmented variants cannot cross the train and test boundaries. Validation and test preprocessing stay deterministic. I also reuse the same inference preprocessing function in the application, avoiding a common bug where a model receives differently normalized uploads.”

## 8. Transfer learning, ImageNet and freezing

**Transfer learning** starts with features learned on another task. **ImageNet** is a large general-image classification dataset; its representations can help when task-specific labeled data are limited. It is not an apple-disease dataset.

**Freezing** disables gradients for selected parameters. First, the pretrained backbone is frozen while a new six-class head learns. MobileNetV2 uses torchvision ImageNet V1 weights; ResNet50 uses V2. The names identify weight releases, not different task classes. Inference reconstructs the model without downloading pretrained weights because the checkpoint already contains learned parameters.

**Interview Answer**

“Training a large CNN from scratch requires substantial labeled data. I used ImageNet-pretrained backbones to start with useful visual representations. First I froze the backbone and trained a new six-class head. This lets the head adapt without immediately disturbing the pretrained features. ImageNet is a starting point, so I still validate how well those features transfer to leaf diseases.”

## 9. Fine-tuning and learning rate

**Fine-tuning** allows selected pretrained weights to adapt. This project unfreezes MobileNetV2's later feature blocks or ResNet50's layer4, starting from the best frozen checkpoint. **Learning rate** controls update size. We use Adam at 0.001 for head/baseline learning and 0.00003 for fine-tuning, a much smaller rate to avoid large destructive updates.

Early stopping monitors validation macro F1. ReduceLROnPlateau halves the rate when validation loss stops improving. ModelCheckpoint-equivalent code saves the best validation checkpoint. Fine-tuning can hurt; the old MobileNetV2 and ResNet50 experiments show this, and the new study keeps frozen candidates eligible.

**Interview Answer**

“After training the head, I unfreeze only later backbone blocks and lower the learning rate substantially. These blocks can adapt to disease patterns while earlier features remain stable. I start from the best frozen checkpoint and select the best fine-tuned checkpoint using validation macro F1. Fine-tuning is an experiment, not an automatic improvement, so the frozen model remains a candidate.”

## 10. Class imbalance and class weights

**Class imbalance** means labels have different sample counts. Apple Scab has 5,382 retained images and Alternaria has 417, so aggregate accuracy could hide weak minority-class performance.

A **class weight** increases the loss contribution of rare training labels. Our formula is N_train/(K × n_class), with K=6. We calculate it only from training counts. Weighting can improve minority recall but may reduce majority accuracy or calibration; it is not a guarantee. We do not artificially weight the final test metrics. A controlled weighted/unweighted ablation is future work rather than a claimed completed result.

**Interview Answer**

“The dataset is imbalanced, so a model can obtain good accuracy while neglecting rare diseases. I calculate class weights from the training labels so minority-class mistakes contribute more to the learning loss. I also use macro F1 for selection because every class contributes equally. I report per-class support and precision/recall so a single overall score cannot conceal a weak class.”

## 11. Precision, recall, F1 and macro F1

For one disease, **precision** = TP/(TP+FP): among predictions of that disease, how many were correct? **Recall** = TP/(TP+FN): among actual examples, how many were detected? **F1** = 2PR/(P+R), balancing precision and recall. Undefined metrics use zero rather than crashing.

**Macro F1** averages each class's F1 equally, not the F1 of averaged precision/recall. **Weighted F1** weights each class's F1 by test support and can be dominated by frequent diseases. **Accuracy** counts total correct predictions. These metrics answer different questions. Single-label macro recall is also average class recall.

**Interview Answer**

“Precision measures how trustworthy a disease prediction is, and recall measures how many true disease examples the model finds. F1 balances both. I selected by macro F1 because it gives each of the six classes equal influence despite imbalance. I also report accuracy, weighted F1 and classwise scores. Weighted F1 reflects the dataset frequency mix, while macro F1 makes weak minority classes more visible.”

## 12. Confusion matrix and error analysis

A **confusion matrix** has true classes on rows and predicted classes on columns. Diagonal counts are correct. Off-diagonal entries identify mistakes such as Scab predicted as Frogeye. Read counts with class support: a frequent class can have more errors even when its recall is good.

We save high-confidence correct predictions, high-confidence wrong predictions, and low-confidence predictions. Confidence is softmax maximum, with a 0.6 warning threshold in the demo. Actual confusion pairs and visual observations are in `docs/error_analysis.md` after evaluation. Similar spots, multiple leaves, backgrounds and lighting are plausible explanations only when visible in examples; we cannot infer a biological diagnosis from scores alone.

**Interview Answer**

“I inspect the confusion matrix to see which diseases are mixed up, rather than stopping at accuracy. Rows show ground truth and columns show predictions. I then look at actual wrong images, including confidently wrong ones. This helps identify visible patterns such as subtle lesions, mixed symptoms or background distractions. Those observations are hypotheses, and I separate them from the measured error counts.”

## 13. ResNet50 and residual/skip connections

**ResNet50** is a 50-layer residual CNN. A **residual block** combines a learned transformation F(x) with a **skip connection**: y=F(x)+x, or a projection when dimensions differ. The shortcut helps information and gradients flow through deep networks. It does not mean the model skips all computation or guarantees better results.

The new head is dropout plus six logits. Phase two trains layer4 and the head while preserving earlier weights and pretrained batch-norm statistics. ResNet50's larger compute/size matters when choosing a practical demo model.

**Interview Answer**

“ResNet uses residual blocks, where the input is added to a learned transformation. The shortcut provides a direct path for information and gradients, making deep networks easier to optimize. I used ResNet50 as a well-known strong comparison, replaced its classifier with six outputs, and fine-tuned its final residual stage. I assess its validation performance alongside its model size and inference cost.”

## 14. MobileNetV2 and depthwise separable convolution

**Depthwise convolution** applies a spatial filter separately to each input channel. A **pointwise 1×1 convolution** mixes channels. Their combination is a **depthwise separable convolution**, reducing computation compared with dense spatial convolution.

MobileNetV2 uses **inverted residual** blocks: expand channels, apply depthwise filtering, then project to a narrow representation. Its **linear bottleneck** avoids an extra nonlinear activation at that projection, preserving useful information in the narrow space. A residual connection is used when compatible shapes allow it. “Mobile” describes its design goal; actual phone performance still requires measurement.

**Interview Answer**

“MobileNetV2 reduces computation using depthwise spatial filtering and pointwise channel mixing. Its inverted residual blocks expand features, filter them, and project them into a narrow linear bottleneck. This makes it a practical deployment candidate compared with larger CNNs. I used the same dataset split and evaluation policy as ResNet50 so I could compare predictive performance and model size fairly.”

## 15. Grad-CAM

**Grad-CAM** weights convolutional feature maps by gradients of a target class score, sums them and applies ReLU to show regions positively associated with that score. We use the predicted class logit, not a post-softmax probability, to avoid cross-class probability coupling. The heatmap is resized and overlaid on the original image.

It is a coarse model explanation, not a lesion mask, proof of causality or proof of correct diagnosis. A bright background region is a reason to inspect shortcuts. Heatmap resolution depends on the target layer. Representative examples include every true class and display actual confidence and ground truth.

**Interview Answer**

“Grad-CAM provides a spatial explanation by using gradients of the predicted class score to weight the final convolutional feature maps. I overlay the result on the leaf to inspect whether highlighted areas include lesions or irrelevant backgrounds. It is helpful for debugging, but it is not a precise segmentation or proof that the model learned the correct biological reason.”

## 16. Why the final model was selected

The winner is **resnet50_finetuned**, with validation accuracy 96.52% and validation macro F1 0.9593. The final fixed test result is 97.07% accuracy and 0.9627 macro F1. Its checkpoint is 94.40 MB. `results/selection.json` records the validation decision and checkpoint hash. Check `README.md` for the populated results table. The rule was fixed before test evaluation: maximum validation macro F1, ties resolved by lower validation cross entropy. Only the selected checkpoint was tested. Do not claim the model was chosen because its test accuracy was highest.

**Interview Answer**

“I selected the final checkpoint using validation macro F1 because the task has imbalanced classes and I want balanced disease recognition. The selected checkpoint and selection rule were saved before opening the test loader. I then evaluated that single model on the held-out test set. Model size and timing are reported as practical context; I did not use test performance to change the choice.”

## 17. What changed from the original internship work?

The original notebooks are authentic history. Their generator outputs confirm 307 train, 75 validation and 120 test examples in four classes. Genuine stored test results exist only for ConvNeXt and the ensemble. Some notebooks use inconsistent ImageNet preprocessing; the baseline and ensemble augment validation. Xception prints two different validation summaries from different histories. Checkpoints mentioned in notebook logs were absent from the Drive folder.

The upgrade adds a sourced six-class leaf dataset, reproducible grouped split, duplicate/label-conflict audit, a smaller understandable baseline, correct shared preprocessing, explicit two-stage transfer learning, validation-only selection, untouched test reporting, per-class evaluation, error examples, Grad-CAM, modular inference, tests and a working demo. New and old scores are not directly comparable because the benchmark changed.

**Interview Answer**

“The internship work helped me explore architectures, but its benchmark was small and some validation/preprocessing choices were inconsistent. I preserved the notebooks and documented the actual stored evidence instead of inflating results. The rebuilt project improves data provenance, duplicate-aware splitting, reproducibility, balanced evaluation and inference consistency. Since the dataset and labels changed, I describe methodological improvements rather than claiming a controlled numerical improvement over the original scores.”

## 18. Limitations and future improvements

- No independently captured orchard/season test set; related scenes may survive perceptual grouping.
- The label set excludes other diseases and assumes one dominant diagnosis per image.
- Source labels can be imperfect; visual review is not expert relabeling.
- Small lesion details may be lost at 224×224; aspect ratio is changed by square resizing.
- Confidence is uncalibrated; non-leaf and unknown diseases can receive high confidence.
- One seed and limited epoch budgets; no confidence intervals or multi-seed stability study.
- Only the selected model is tested; there is no claimed five-model test leaderboard.
- Grad-CAM is approximate; phone deployment latency is unmeasured.

Useful next steps are expert-reviewed labels, acquisition-group metadata, independent orchard testing, unknown-image rejection, calibration on validation data, multi-seed runs, a class-weight ablation and actual target-device measurements. Avoid adding a model zoo merely to enlarge the project.

## 25+ likely questions with short answers

1. **What is the task?** Single-label six-class classification of apple leaf images.
2. **Is it object detection?** No. The output labels an image; it does not output lesion boxes.
3. **Which dataset did you use?** Official AppleLeaf9, licensed CC BY 4.0, with six selected classes.
4. **How many images did you use?** 13,405 after quarantining 56 conflicting near-duplicate images from 13,461 selected files.
5. **What is the split?** 9,378 train, 2,014 validation, 2,013 test, with seed-42 stratified group allocation.
6. **Why not split augmented images?** Related views could land in different splits and inflate evaluation.
7. **How did you check duplicates?** Byte and decoded-pixel hashes, plus rotation/reflection-aware pHash grouping.
8. **Can you guarantee zero leakage?** No; missing leaf/orchard IDs and undetected variants remain limitations.
9. **Why three architectures?** They provide a simple baseline, lightweight deployment candidate and established residual comparison.
10. **Why a custom CNN?** It demonstrates convolution, normalization, nonlinearities, pooling, dropout and classification without pretrained features.
11. **Why global average pooling?** It keeps the head small instead of flattening a large spatial tensor.
12. **What is a kernel?** A small learned set of weights used to compute local feature responses.
13. **Why ReLU?** It adds nonlinearity and is simple to optimize for positive activations.
14. **What does dropout do?** Randomly removes activations during training, reducing dependence on specific units.
15. **What does batch normalization do?** It normalizes intermediate activations with learned scale/shift and running statistics.
16. **What is transfer learning?** Reusing pretrained representations and adapting a classifier to a new task.
17. **Why freeze the backbone first?** To let the randomly initialized head learn without disturbing pretrained features immediately.
18. **What did you fine-tune?** Later MobileNetV2 blocks or ResNet50 layer4, plus the head.
19. **Why lower the fine-tuning rate?** To adapt pretrained weights gradually and avoid excessive changes.
20. **Can fine-tuning reduce accuracy?** Yes; the frozen candidate remains eligible for selection.
21. **What is a ResNet skip connection?** It adds a block input to its transformed output, helping gradients/information flow.
22. **Why MobileNetV2?** Depthwise separable filtering and inverted residual blocks reduce compute/parameters.
23. **Which loss?** Weighted CrossEntropyLoss on logits; softmax is applied only for prediction display.
24. **How do you handle imbalance?** Training-only class weights and macro-F1-based model selection.
25. **Precision versus recall?** Precision checks predicted positives; recall checks actual positives.
26. **Why macro F1?** It gives rare and common classes equal influence.
27. **Why also weighted F1?** It describes performance under the test class-frequency mix.
28. **How was the model selected?** Maximum validation macro F1, then lower validation loss for ties.
29. **Did you tune on test errors?** No. The selected model is fixed before test evaluation.
30. **What does the confusion matrix show?** True-versus-predicted counts, including which disease pairs are mixed up.
31. **What does Grad-CAM explain?** Spatial features associated with a chosen class score; not an expert diagnosis.
32. **Does high confidence ensure correctness?** No. Softmax is uncalibrated and has no unknown-image class.
33. **How is inference reproducible?** The checkpoint stores class order, architecture, image size and shared preprocessing.
34. **Did training use the GPU?** Yes, the current run uses PyTorch CUDA on an H100 40 GB MIG allocation; provenance is recorded.
35. **Can the demo prescribe treatment?** No; it demonstrates predictions and explanations only.
36. **What is your biggest limitation?** No independent orchard/acquisition-group benchmark to establish external generalization.
37. **Were the old checkpoints recovered?** No; nine notebooks were recovered, with stored outputs but no standalone weights.
38. **Is old EfficientNetB3's 96% a test score?** No, it is a historical validation maximum on a different four-class benchmark.
39. **What would you improve first?** Better group metadata and expert labels, independent testing, then calibration and deployment measurements.
40. **Where are the exact current results?** `results/metrics.json`, `results/model_comparison.csv`, and the README generated from them.
