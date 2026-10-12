"""Copy only compact experiment evidence into Git; checkpoints stay local."""
import json
import shutil
from pathlib import Path
import pandas as pd
from .data import sha256


def main():
    target = Path("results/four_class")
    records = []
    manifest_hash = sha256(target / "manifest.csv")
    for name in ("ann", "cnn", "resnet50"):
        source = Path("artifacts") / name
        if not (source / "metrics.json").exists():
            continue
        metrics = json.loads((source / "metrics.json").read_text())
        if metrics["manifest_sha256"] != manifest_hash:
            raise ValueError("Cannot compare experiments using different splits")
        if metrics["checkpoint_sha256"] != sha256(source / "best.keras"):
            raise ValueError("Checkpoint differs from evaluated checkpoint")
        dest = target / name; dest.mkdir(exist_ok=True)
        for file in source.iterdir():
            if file.is_file() and file.suffix in (".json", ".csv", ".png", ".txt"):
                shutil.copy2(file, dest / file.name)
        record = {k: metrics[k] for k in ("model", "accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted", "n_images", "training_seconds", "parameters", "selected_phase", "manifest_sha256", "checkpoint_sha256")}
        record["validation_loss"] = metrics["validation"]["loss"]
        record["validation_accuracy"] = metrics["validation"]["accuracy"]
        records.append(record)
    pd.DataFrame(records).to_csv(target / "model_comparison.csv", index=False)
    (target / "model_comparison.json").write_text(json.dumps(records, indent=2))
    if records:
        selected = min(records, key=lambda r: r["validation_loss"])
        (target / "selection.json").write_text(json.dumps({"model": selected["model"], "criterion": "minimum validation loss", "test_used_for_selection": False, "checkpoint_sha256": selected["checkpoint_sha256"]}, indent=2))
    print(pd.DataFrame(records)[["model", "accuracy", "f1_macro", "n_images"]].to_string(index=False))


if __name__ == "__main__":
    main()
