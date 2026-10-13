# Detection and Roboflow workflow

Classification predicts one category per image. Detection predicts a list of objects,
each with a bounding box, category and confidence. This module uses Ultralytics
YOLO11n; it is separate from the four-class fruit classifiers. Never convert folder
labels to fabricated full-frame bounding boxes.

## What is implemented and verified

- `apple_disease.detection.validate_yolo`: safe YAML parsing; class mapping/nc checks;
  train/val/test paths or image lists; image decoding; exact decoded-image duplicate
  detection; missing/empty labels; numeric class IDs; normalized box coordinates and
  boundary checks; suspicious full-frame annotation rejection.
- `train_yolo`: validate first, write explicit sanitized image lists, train, load the
  best validation checkpoint, evaluate on test, and save precision/recall/mAP separately.
  The wrapper has a mocked API contract test. That test is not actual YOLO training.
- `predict_yolo`: verified actual inference, drawing boxes and class names, explicit
  PIL RGB input and conversion of Ultralytics' BGR plot back to RGB.
- `roboflow_import`: tested safe local ZIP import, path normalization and class/box
  validation; optional API workflow takes `ROBOFLOW_API_KEY` from the environment.
  Credentialed remote download was not run. No API key is committed.

**Custom disease detector training and a new disease mAP benchmark remain incomplete.**
No suitable detector is registered in the app, so its disease detection control is disabled.

## Actual pretrained inference example

Official [Ultralytics YOLO11n weights](https://huggingface.co/Ultralytics/YOLO11/blob/4839c432499b0dba0559c31a35f8b44c68ff81a6/yolo11n.pt)
were downloaded from the publisher's Hugging Face repository after GitHub release
downloads failed with connection resets/TLS timeouts. Weight SHA-256:
`0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1`.
Ultralytics 8.4.175 / torch 2.14.1 ran on CPU.

The chosen local fruit image `Train/Normal_Apple/1.jpg` produced one **cake** box at
0.6384 confidence. That is a recorded failure on an apple photograph. No ground-truth
object box was available for this fruit image, so no precision, recall or mAP is computed.
This run proves the inference/visualization path works, not that generic COCO weights
diagnose disease. Numerical evidence is in `results/four_class/yolo_pretrained_example.json`.
The annotated image is local at `results/local_examples/yolo_prediction.png` because
the input image's redistribution license is unknown. Ultralytics weights/code have
their own AGPL/commercial terms; the project's MIT license does not replace those terms.

## Recovered historical YOLO evidence

Local `research papaer seen till now/yolo experiment/` contains YOLOv5 notebooks,
VOC XML boxes, YOLO labels and a historical checkpoint. The split lists have 4,766
train, 2,036 val and 2,036 test paths (8,838 total), with no filename overlap. The
directory itself has 8,839 image entries and label entries; entries are not verified
unique-image counts. Its class order is **Healthy, Black rot, Scab, Rust**, different
from the fruit classifier. Three inspected examples show boxes around apple leaves,
not lesion masks. The code did not manufacture those boxes.

The new full-validator attempt **rejected this dataset** on an exact duplicate across
validation and test: `val` contains `b5b00f6d794132d3.jpg`, while `test` contains
`00b5b00f6d794132d3.jpg`. They have the same byte SHA-256
`4ad30b4ac563733bc81ee1aa1d5b6b255a169b522624651ab8b1575124020565`.
Different filenames had hidden the overlap. The validator stops at this failure,
so there is no claim of a complete unique-image audit of the detection corpus.
See `results/four_class/legacy_detection_audit.json`. This split must be repaired
with source-group/duplicate review and annotation provenance established before a
defensible custom disease benchmark. Its historical validation log cannot be treated
as an independent test result.

A local YOLOv5 `results.csv` contains **one epoch (epoch 0)** of historical validation
logging: precision 0.24312, recall 0.8086, mAP@0.5 0.34663, mAP@0.5:0.95 0.19932.
These are unverified historical validation outputs, not new test results and not
evidence that the notebook's requested 2,000 epochs completed. Source/license,
annotation authorship, checkpoint provenance and near-duplicate independence remain
unestablished. The new detector is not trained/deployed from these unchecked assumptions.

## Prepare a real detection export

1. Establish the source and license. Decide whether boxes surround diseased leaves,
   fruit, or lesions; keep that annotation target consistent.
2. In a Roboflow object-detection project, draw/tighten boxes and assign reviewed labels.
   A category attached to a whole image is insufficient. Include explicit reviewed
   negative images if appropriate, with empty label files.
3. Group related originals by source before assigning train/validation/test. Ensure
   transformations of the same image stay together. Do not augment validation/test.
4. Generate a numbered dataset version. Record source URL, license, version, class
   definitions, annotators/review procedure, original image counts and transformations.
5. Export **YOLO detection** format. Verify class order from `data.yaml` against labels.
   Keep raw image ownership and annotation provenance with the export.
6. Run local validation and visually inspect boxes. Investigate near duplicates and
   source relationships: passing format checks alone does not establish ground-truth quality.
7. Train locally and evaluate the selected checkpoint on a held-out detection test set.
   Save box metrics separately from classification reports.

Example portable layout:

```text
data/detection/
  data.yaml
  train/images/*.jpg
  train/labels/*.txt
  valid/images/*.jpg
  valid/labels/*.txt
  test/images/*.jpg
  test/labels/*.txt
```

Example YAML (only if these are the actual reviewed box classes):

```yaml
path: .
train: train/images
val: valid/images
test: test/images
nc: 4
names: [Blotch, Rot, Scab, Normal]
```

Each label row: `class_id center_x center_y width height`, normalized to [0,1].
List-file entries in this project's input convention are relative to YAML `path`,
or absolute. The training wrapper converts them to absolute paths for Ultralytics.
The importer recognizes a common `../train/images` export layout only when the
original target is absent and a sibling local directory exists; it preserves
`data.original.yaml` when correcting that path. It never changes annotation coordinates.

```powershell
python -m pip install -r requirements-detection.txt
python -m apple_disease.roboflow_import local path/to/export.zip --output data/detection
python -m apple_disease.detection validate data/detection/data.yaml
python -m apple_disease.detection train data/detection/data.yaml --output artifacts/yolo_disease --epochs 30 --device cpu
python -m apple_disease.detection predict path/to/apple.jpg --weights artifacts/yolo_disease/train/weights/best.pt
```

For reviewed negative images, pass `--allow-background` to validate and train. Missing
labels remain errors. The dataset must contain positive annotations in every split.
Use `--device 0` only with a working CUDA-enabled PyTorch environment.

For API download, set `ROBOFLOW_API_KEY` in the process environment using your secret
manager, then run (replace the identifiers):

```powershell
python -m apple_disease.roboflow_import download --workspace YOUR_WORKSPACE --project YOUR_PROJECT --version 1 --output data/detection
```

After an actually trained and reviewed disease detector exists, create local
`models/detection.json` with `status: "trained"`, `task: "apple_disease_detection"`,
`weights` pointing to its `.pt` file, ordered `names`, and its `sha256`. The app verifies
the file hash and runtime class names. Do not register generic COCO weights as disease weights.

References: [Ultralytics annotation format](https://docs.ultralytics.com/datasets/detect/),
[training](https://docs.ultralytics.com/modes/train/),
[validation metrics](https://docs.ultralytics.com/modes/val/),
[Roboflow version export](https://docs.roboflow.com/datasets/create-and-upload/download-a-dataset).
