Reproducible NIT internship project rebuild: six-class AppleLeaf9 benchmark.

The attached resnet50_finetuned.pth is the best validation-selected checkpoint. It was selected by validation macro F1 before test evaluation.

Architecture: resnet50
Validation accuracy: 96.5243%
Validation macro F1: 0.959283
SHA-256: 48fde2b0f9232e7ef39b0c89a8667296c45141bd6512557fe4e2eef1e04eec2c

Download with `python scripts/download_model.py`, then launch `streamlit run app.py`. The checkpoint stores label ordering and preprocessing metadata. Raw data and other experimental checkpoints are not included. See README and data/README.md for AppleLeaf9 CC BY 4.0 attribution and measured test results.
