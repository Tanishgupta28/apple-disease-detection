# Model artifacts

Training saves the best validation checkpoint for each of the five experiments as `.pth`. These binary files are ignored by Git. `selected_model.json` records the architecture, six-class ordering, preprocessing, checkpoint filename, SHA-256, measured validation results and hardware for the winner. Results selection is based on validation macro F1, never test accuracy.

The final selected checkpoint is published as a GitHub release asset when training and verification complete. Download it without putting the dataset or other checkpoints in Git history:

```bash
gh release download v1.0.0 --repo Tanishgupta28/apple-disease-detection --pattern '*.pth' --dir models
```

Only load checkpoints from sources you trust. Inference loads weights with `weights_only=True` and reconstructs the architecture without downloading ImageNet weights. Check its SHA-256 against `selected_model.json` before sharing or using a copied artifact.
