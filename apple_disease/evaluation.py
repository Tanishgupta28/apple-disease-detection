"""Held-out classification metrics. Detection results use a separate schema."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from . import CLASSES


def validate_probabilities(probabilities):
    values = np.asarray(probabilities)
    if values.ndim != 2 or values.shape[1] != 4 or not np.isfinite(values).all():
        raise ValueError("Expected N x 4 finite probabilities")
    if np.any(values < 0) or np.any(values > 1) or not np.allclose(values.sum(axis=1), 1, atol=1e-5):
        raise ValueError("Invalid Softmax probabilities")
    return values


def classification_metrics(labels, probabilities):
    probabilities = validate_probabilities(probabilities)
    prediction = probabilities.argmax(axis=1)
    report = classification_report(labels, prediction, labels=list(range(4)),
                                   target_names=CLASSES, output_dict=True, zero_division=0)
    return {"accuracy": float(accuracy_score(labels, prediction)),
            "precision_macro": report["macro avg"]["precision"],
            "recall_macro": report["macro avg"]["recall"],
            "f1_macro": report["macro avg"]["f1-score"],
            "f1_weighted": report["weighted avg"]["f1-score"],
            "per_class": {c: report[c] for c in CLASSES},
            "confusion_matrix": confusion_matrix(labels, prediction, labels=list(range(4))).tolist(),
            "n_images": len(labels)}


def save_evaluation(output, frame, probabilities, metadata):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    metrics = classification_metrics(frame.label.to_numpy(), probabilities)
    metrics.update(metadata)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    predictions = frame[["path", "label", "class_name", "split"]].copy()
    predictions["prediction"] = probabilities.argmax(axis=1)
    for i, name in enumerate(CLASSES):
        predictions[f"p_{name}"] = probabilities[:, i]
    predictions.to_csv(output / "predictions.csv", index=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    ConfusionMatrixDisplay(np.asarray(metrics["confusion_matrix"]), display_labels=CLASSES).plot(cmap="Blues")
    plt.title(f"{metadata['model']} | held-out test n={len(frame)}")
    plt.tight_layout(); plt.savefig(output / "confusion_matrix.png", dpi=140); plt.close()
    return metrics
