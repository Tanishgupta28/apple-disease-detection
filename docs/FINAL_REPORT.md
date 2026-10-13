# Rebuild completion report

## Delivered

An active TensorFlow/Keras package under `apple_disease/` implements data audits,
reproducible grouped splits, shared RGB preprocessing, Lab K-means, ten GLCM texture
features, ANN, CNN, two-stage ResNet50, training/evaluation, checksum-verified inference,
a Streamlit app, and separate YOLO/Roboflow workflows. New experiment evidence is
under `results/four_class/`. The original notebooks and prior six-class project
remain distinguishable and preserved.

## Actually trained and measured

| Experiment | Completed training | Test accuracy | Macro F1 |
|---|---|---:|---:|
| ANN | 60 epochs | 48.74% | 0.4439 |
| CNN | 9 epochs; epoch 3 selected | 46.22% | 0.2969 |
| ResNet50 | 5 frozen-head + 4 fine-tuning epochs | 66.39% | 0.6546 |

All use the same 119-image held-out test set. ResNet50's frozen-head checkpoint is
selected because fine-tuning worsened validation loss. The CNN never predicts Scab
or Normal on this test set; both have zero recall. No test-driven retraining was done.
Complete precision/recall/weighted F1, per-class metrics, confusion matrices and
probabilities are committed. Trained checkpoints remain local under `artifacts/`.

## Verified classification data

Recovered local `apple_disease_classification` folders match historical Kaggle path
labels/counts, but original publisher/license is not established. Inspection shows
fruit images, including stock-photo watermarks. There are 502 files, 472 unique byte
and pixel images, zero corrupt images, and 459 retained after audit. Counts retained:
Blotch 131, Rot 139, Scab 107, Normal 82. Split: 270 train / 70 validation / 119 test.
No claim of 3,000 unique historical classification leaf images is supported.

## Verification

- 34 active tests passed: data loading/mapping/leakage, image decoding, K-means,
  constant/variable GLCM, model shapes/probabilities, weight updates, saved-model
  loading, ResNet preprocessing/freeze behavior, detection annotations, local ZIP
  import, credential absence, Streamlit startup/invalid upload/texture/segmentation.
- All evaluated checkpoints passed SHA-256 verification; saved predictions were
  reproduced with maximum difference below 1.2e-7. Stored classification metrics were
  independently recalculated from saved predictions and matched.
- Python compilation and `pip check` passed. Streamlit's local health endpoint returned 200.
- The 371 test warnings are upstream Keras/NumPy copy-keyword deprecations, not failures.
- Prior six-class tests were not rerun and are not included in this claim.

Read `test_verification.json` and `verification.json` under `results/four_class/`.

## GitHub milestones

Each milestone was tested, committed to main, pushed, and checked against GitHub main:

1. `82774d6`: audited four-class ANN/CNN pipeline and measured baselines; 9 tests passed.
2. `3afa784`: both ResNet phases and evaluation; 10 tests passed cumulatively.
3. `ee54f9e`: detection/Roboflow workflow; 16 detection tests and actual generic inference.
4. Final demo/documentation milestone: use `git log -1` for its hash; it includes this
   report, 34-test evidence and the 56-question interview guide.

No force push, credential commit, raw image upload or large model commit was used.
The supplied folder initially lacked `.git`; it was attached in place to the existing
remote history. The two colliding local files were first preserved under ignored
`.local_original_backup/`; other pre-existing local files remain untouched.

## Incomplete and limited

Custom disease YOLO training/evaluation is incomplete. The wrapper is implemented and
contract-tested; generic pretrained inference actually ran but predicted “cake” on
an apple. This is not a disease result or a mAP measurement. Older local leaf box
annotations and a one-epoch YOLOv5 validation log are separately documented.
The recovered detection dataset was rejected for an exact validation/test duplicate
under different filenames. The failed audit is saved; a full repaired source-group
split and annotation-provenance review are still needed before a defensible benchmark.
Credentialed Roboflow download was not run; local import works. Source rights and
biological label validity remain unverified. No lesion ground truth, independent
orchard validation, calibrated probabilities, unknown-image rejection or multi-seed
study is claimed. The small recovered fruit data does not substantiate the resume's
3,000+ leaf classification statement.

## Run and study

```powershell
cd C:\Users\skg21\Documents\projects\Apple_disease_detection
.\venv\Scripts\python.exe -m streamlit run app.py
```

Use http://localhost:8501. If the verification server is already running, open that
URL directly. For tests: `.\venv\Scripts\python.exe -m pytest`.

Study in order: `INTERVIEW_PREPARATION.md`, `DATASET.md`, `EXPERIMENTS.md`, then
`apple_disease/features.py`, `models.py`, `resnet.py`, `data.py`, and `train.py`.
Read `YOLO_ROBOFLOW.md` for the exact detection evidence and its limits.

Likely questions: Why fruit rather than leaves? Can you verify 3,000 images? How did
you prevent duplicate leakage? Why fit normalization only on training? What exactly
does a GLCM count? Why doesn't K-means identify lesions? Why does the CNN miss two
classes? Why did fine-tuning fail to improve validation? Why freeze BatchNorm? How
does detection differ from classification, and what would make a YOLO box annotation valid?
