"""Read-only verification of evaluated local checkpoints and committed evidence."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .data import load_manifest, manifest_digest, sha256
from .inference import load_classifier, predict_classifier, available_detector
from .preprocessing import read_image
from .evaluation import classification_metrics


def main():
    manifest = load_manifest("results/four_class/manifest.csv")
    fingerprint = manifest_digest("results/four_class/manifest.csv")
    report = {"manifest_sha256": fingerprint, "split_counts": manifest.split.value_counts().to_dict(), "models": {}}
    for name in ("ann", "cnn", "resnet50"):
        directory = Path("artifacts") / name
        model, info = load_classifier(directory)
        if info["manifest_sha256"] != fingerprint:
            raise ValueError("Model uses a different dataset split")
        table = pd.read_csv(f"results/four_class/{name}/predictions.csv")
        expected_test = manifest[manifest.split == "test"]
        if table.path.tolist() != expected_test.path.tolist() or table.label.tolist() != expected_test.label.tolist():
            raise ValueError("Prediction rows differ from manifest test rows")
        measured = classification_metrics(table.label.to_numpy(), table[[f"p_{c}" for c in info["classes"]]].to_numpy())
        saved = json.loads(Path(f"results/four_class/{name}/metrics.json").read_text())
        for metric in ("accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted"):
            if not np.isclose(measured[metric], saved[metric], atol=1e-12):
                raise ValueError(f"Stored {name} {metric} differs from prediction-derived result")
        first = table.iloc[0]
        image_path = Path("apple_disease_classification") / first.path
        if sha256(image_path) != expected_test.iloc[0].sha256:
            raise ValueError("Verification image changed")
        result = predict_classifier(model, info, read_image(image_path))
        expected = first[[f"p_{c}" for c in info["classes"]]].to_numpy(dtype=float)
        actual = np.array(list(result["probabilities"].values()))
        np.testing.assert_allclose(actual, expected, atol=1e-5)
        report["models"][name] = {"checkpoint_hash_verified": True, "saved_metrics_recomputed": True,
                                  "saved_prediction_reproduced": True,
                                  "max_probability_difference": float(np.max(np.abs(actual - expected)))}
    report["custom_disease_detector_available"] = available_detector() is not None
    Path("results/four_class/verification.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
