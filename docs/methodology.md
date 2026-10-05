# Methodology

## Scope and historical context

The NIT internship notebooks are preserved, not rerun or rewritten. The historical four-class dataset has 307/75/120 images; its results are documented separately. The new task is single-label classification of an apple leaf image into five diseases plus Healthy. Changing datasets/classes prevents a direct performance comparison with the internship benchmark.

## Data provenance and split

AppleLeaf9 comes from its official repository, with CC BY 4.0 attribution and the Yang, Duan and Wang (2022) citation in `data/README.md`. The complete archive has 14,582 images. Only the six requested classes are selected; Brown spot, Grey spot and Mosaic are excluded. Measured distributions are in `results/dataset_audit.json`.

All selected files are checked for decoding, dimensions, byte SHA-256 and decoded RGB pixel hashes. Repeated pixels are deduplicated; conflicting exact labels are quarantined. A 64-bit DCT perceptual hash is calculated over normal, rotated and reflected views. Pairs within Hamming distance 4 form connected groups. Groups with conflicting labels are quarantined, and all remaining group members go into the same split. A visual review of matched pairs is saved.

A seed-42, per-class allocation assigns groups to approximately 70/15/15 partitions. Large groups are allocated first to limit deviations from target proportions. This is stratified group allocation, not a naive random image split. `results/split_manifest.csv` records exact membership and hashes. Preparation refuses to overwrite a saved split.

This audit found zero unreadable selected images and zero exact/decoded duplicates, but quarantined 56 conflicting near-duplicate images. It retains 13,405 images in 12,812 groups: 9,378 train, 2,014 validation and 2,013 test. Perceptual checks cannot recover missing acquisition IDs, detect every crop or prove complete absence of source augmentation leakage. A future independent orchard/leaf-group test is needed.

## Preprocessing and augmentation

EXIF orientation is applied; input is RGB, resized to 224×224 without cropping, converted to a float tensor, and normalized with ImageNet RGB means [0.485, 0.456, 0.406] and standard deviations [0.229, 0.224, 0.225]. All three models use this pipeline for controlled comparison. Resizing a non-square image can distort its aspect ratio; this is an explicit limitation.

Only the training loader applies horizontal flip, ±12° rotation, ±5% translation, scale 0.95–1.05, and brightness/contrast jitter 0.12. Validation, test and inference use the same deterministic transformation. Random affine fill can produce narrow border artifacts; transformations are kept small. No transformed images are added to the raw dataset.

Data are decoded on demand with 8 workers, batch size 64, pinned host memory and prefetching through PyTorch DataLoader. Images are not all loaded into RAM. GPU visibility and CUDA versions are checked before training. FP32 is used for simplicity; no system CUDA installation/upgrade was needed. The available H100 is a 40 GB MIG allocation, not exclusive access to all 80 GB.

## Training design

The custom CNN has four Conv2D-equivalent PyTorch Conv2d blocks (32,64,128,192 channels), batch normalization, ReLU, pooling and dropout, followed by global average pooling, dropout and a linear six-class head. It has 316,198 parameters. Training uses logits and PyTorch CrossEntropyLoss, which internally applies log-softmax and negative log likelihood. Prediction applies softmax. Integer labels are equivalent to one-hot categorical cross entropy for single-label classification.

MobileNetV2 loads official torchvision ImageNet V1 weights, and ResNet50 loads ImageNet V2 weights. Each head is dropout plus a linear six-class classifier. The pretrained backbone is initially frozen; in phase two MobileNetV2 features[14:] or ResNet50 layer4 is unfrozen. Pretrained batch-normalization running statistics stay fixed in both phases, while the head remains trainable. Fine-tuning starts from that model's best frozen validation checkpoint.

Adam learning rate is 0.001 for baseline/head training, and 0.00003 for fine-tuning. Maximum epochs are 15 baseline, 8 frozen and 12 fine-tuning. Early stopping monitors validation macro F1 with patience 5 and minimum improvement 0.0005. ReduceLROnPlateau monitors unweighted validation cross entropy, halves the learning rate, and has patience 2. Checkpoint saving and early stopping are implemented directly as readable PyTorch control flow, the equivalent of Keras callbacks. Every phase restores its best validation macro F1 checkpoint, breaking exact ties by lower validation loss.

Class weights are N_train / (6 × class_training_count), because max/min training class imbalance exceeds the configured threshold of 2. Validation/test metrics remain unweighted. Full-epoch training accuracy is measured under augmentation and dropout; final selected-model train metrics are additionally measured with augmentation disabled. The reported training loss is an image-count-weighted average of batch weighted cross-entropy means.

## Selection and untouched evaluation

The five candidates are custom CNN, MobileNetV2 frozen/fine-tuned and ResNet50 frozen/fine-tuned. A single final model is selected by maximum **validation macro F1**, with validation loss as tie breaker. The selection checkpoint hash is persisted before test evaluation. Training opens no test loader.

Only this final selected model receives the held-out test evaluation. Accuracy, macro precision/recall/F1, weighted F1, per-class precision/recall/F1/support and a raw-count confusion matrix are persisted. Other candidates' test results are intentionally unavailable. There is no further tuning based on the test results. Re-running evaluation reuses existing metrics. Deleting them and selecting models after examining test errors would invalidate the untouched-test interpretation.

Seed 42 covers Python, NumPy, PyTorch and loader workers. Deterministic operations are requested with warnings for unavailable deterministic kernels; identical numeric reproduction across devices/library versions is not guaranteed. Exact tested package versions are saved in `docs/package_versions.json`; the GPU environment is recorded in `docs/environment.json`.

## Error analysis and interpretation

Saved test probabilities support high-confidence correct/incorrect and low-confidence examples. The confidence threshold 0.6 is an interface heuristic, not a calibrated reliability guarantee. Grad-CAM uses the predicted class logit and final spatial feature activations, averaged gradients as channel weights, ReLU and normalized heatmaps. It can suggest background shortcuts; it cannot establish causation or confirm a disease diagnosis. Visual observations and actual confusion counts are documented after evaluation.

Timing uses warmed, synchronized batch-one inference on the recorded device; forward-only and preprocess-plus-forward medians are separate. Model loading, file decoding, uploads and Grad-CAM are excluded. Reported size is checkpoint bytes, not RAM or deployment latency.
