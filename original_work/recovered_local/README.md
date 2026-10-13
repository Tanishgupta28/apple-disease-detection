# Additional local evidence recovered in October 2026

These files are unchanged copies from the supplied Windows workspace. `manifest.json`
records original relative paths, sizes and SHA-256 hashes. Their original date and
authorship are not independently established. No notebook was rerun during recovery.

- `leaf_sampling.ipynb`: source describes sampling up to 250 images from each of
  scab, black_rot, cedar_rust and healthy. It is a different label space from the
  historical fruit-class notebooks. Source code is not proof a downstream model trained.
- `texture_imports_only.ipynb`: imports GLCM/KMeans and neural-network libraries.
  It does not contain an implemented or evaluated ANN/GLCM/K-means pipeline. Imports
  alone do not substantiate completed experiments. The functional texture pipeline
  under `apple_disease/` is newly implemented work.
- `yolov5_training_attempt.ipynb`: path adjustment, subset/resizing code and a command
  requesting 2,000 YOLOv5 epochs. A requested epoch count is not evidence of completion.
- `yolov5_epoch0.csv`: recovered one-epoch historical validation log. It is not a new
  held-out test result; see `docs/YOLO_ROBOFLOW.md` for scope and limitations.

Raw data and legacy checkpoints remain local, excluded from Git. Existing original
internship notebooks in the sibling directory remain preserved.
