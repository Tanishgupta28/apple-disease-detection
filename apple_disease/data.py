"""Audit original Train/Test folders, deduplicate, group related images, then split."""
import argparse
import hashlib
import json
from pathlib import Path
from collections import Counter
import imagehash
import numpy as np
import pandas as pd
from PIL import Image
from . import CLASSES
from .preprocessing import read_image

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def perceptual_hashes(array):
    """Eight rotations/reflections; Hamming matches are candidates, not proof."""
    return [int(str(imagehash.phash(Image.fromarray(b))), 16)
            for a in [array, np.fliplr(array)] for b in [np.rot90(a, k) for k in range(4)]]


def prepare(root, output, seed=42, threshold=6, validation_fraction=.2):
    root, output = Path(root), Path(output)
    if (output / "manifest.csv").exists():
        raise FileExistsError("Manifest already exists; use a new output directory for a new split")
    if not 0 < validation_fraction < .5 or not 0 <= threshold <= 16:
        raise ValueError("Invalid validation fraction or duplicate threshold")
    rows, corrupt, ignored = [], [], []
    for partition in ("Train", "Test"):
        base = root / partition
        if not base.is_dir():
            raise ValueError(f"Missing {partition} directory")
        unexpected = {p.name for p in base.iterdir() if p.is_dir()} - {c + "_Apple" for c in CLASSES}
        if unexpected:
            raise ValueError(f"Unknown class directories: {unexpected}")
        for label, name in enumerate(CLASSES):
            folder = base / (name + "_Apple")
            if not folder.is_dir():
                raise ValueError(f"Missing class directory {folder}")
            for p in sorted(folder.rglob("*")):
                if not p.is_file():
                    continue
                relative = p.relative_to(root).as_posix()
                if p.suffix.lower() not in EXTENSIONS:
                    ignored.append(relative)
                    continue
                try:
                    array = read_image(p)
                except ValueError as exc:
                    corrupt.append({"path": relative, "error": str(exc)})
                    continue
                pixel_hash = hashlib.sha256(str(array.shape).encode() + array.tobytes()).hexdigest()
                rows.append(dict(path=relative, label=label, class_name=name,
                                 original_split=partition, sha256=sha256(p), pixel_hash=pixel_hash,
                                 width=array.shape[1], height=array.shape[0],
                                 hashes=perceptual_hashes(array), status="included"))
    if not rows:
        raise ValueError("No valid images found")
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    near_pairs = []
    for i in range(len(rows)):
        for j in range(i):
            exact = rows[i]["pixel_hash"] == rows[j]["pixel_hash"]
            distance = min((h ^ rows[j]["hashes"][0]).bit_count() for h in rows[i]["hashes"])
            if exact or distance <= threshold:
                parent[find(i)] = find(j)
                near_pairs.append({"path_a": rows[j]["path"], "path_b": rows[i]["path"],
                                   "hamming": distance, "exact_pixels": exact,
                                   "cross_label": rows[i]["label"] != rows[j]["label"]})
    groups = {}
    for i, row in enumerate(rows):
        groups.setdefault(find(i), []).append(row)
    for group in groups.values():
        group_id = min(r["sha256"] for r in group)[:16]
        conflict = len({r["label"] for r in group}) > 1
        has_test = any(r["original_split"] == "Test" for r in group)
        seen = set()
        for row in sorted(group, key=lambda r: (r["original_split"] != "Test", r["path"])):
            row["group"] = group_id
            row["split"] = "excluded"
            if conflict:
                row["status"] = "conflicting_labels"
            elif row["pixel_hash"] in seen:
                row["status"] = "exact_duplicate"
            elif has_test and row["original_split"] == "Train":
                row["status"] = "related_to_original_test"
            else:
                row["split"] = "test" if has_test else "train"
            seen.add(row["pixel_hash"])
    rng = np.random.default_rng(seed)
    # Stratification at the group level keeps every related-image group together.
    for label in range(4):
        candidates = [g for g in groups.values() if any(r["label"] == label and r["split"] == "train" for r in g)]
        if len(candidates) < 2:
            raise ValueError(f"Need at least two independent training groups for {CLASSES[label]}")
        rng.shuffle(candidates)
        target = sum(sum(r["split"] == "train" for r in g) for g in candidates) * validation_fraction
        count = 0
        for group in candidates[:-1]:
            if count >= target:
                break
            for row in group:
                if row["split"] == "train":
                    row["split"] = "validation"
                    count += 1
    frame = pd.DataFrame([{k: v for k, v in r.items() if k != "hashes"} for r in rows])
    valid = frame[frame.status == "included"]
    validate_manifest(valid)
    report = {"source": "Recovered local appledisease Train/Test folders; original publisher/license unverified",
              "seed": seed, "classes": list(CLASSES), "valid_image_files": len(rows),
              "unique_byte_hashes": frame.sha256.nunique(), "unique_pixel_hashes": frame.pixel_hash.nunique(),
              "corrupt_files": corrupt, "ignored_files": ignored,
              "near_duplicate_hamming": threshold, "near_pair_count": len(near_pairs),
              "status_counts": dict(Counter(frame.status)),
              "raw_class_counts": dict(Counter(frame.class_name)),
              "included_class_counts": dict(Counter(valid.class_name)),
              "split_counts": valid.groupby(["split", "class_name"]).size().unstack(fill_value=0).to_dict(orient="index"),
              "validation_fraction_of_remaining_original_train": validation_fraction,
              "test_policy": "Retain original Test images after deduplication/conflict quarantine; exclude related Train images"}
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "audit_manifest.csv", index=False)
    valid.to_csv(output / "manifest.csv", index=False)
    pd.DataFrame(near_pairs, columns=["path_a", "path_b", "hamming", "exact_pixels", "cross_label"]).to_csv(output / "near_duplicates.csv", index=False)
    (output / "dataset_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    table = valid.groupby(["class_name", "split"]).size().unstack(fill_value=0).reindex(CLASSES)
    ax = table[["train", "validation", "test"]].plot.bar(rot=0, figsize=(8, 4))
    ax.set(ylabel="Unique included images", title="Recovered four-class fruit dataset")
    plt.tight_layout(); plt.savefig(output / "class_distribution.png", dpi=150); plt.close()
    return report


def validate_manifest(frame):
    required = {"path", "label", "class_name", "split", "group", "pixel_hash", "sha256"}
    if not required.issubset(frame.columns) or frame.empty:
        raise ValueError("Missing manifest columns or empty manifest")
    if set(frame["split"]) != {"train", "validation", "test"}:
        raise ValueError("Manifest must contain train, validation and test")
    if frame.path.duplicated().any() or frame.pixel_hash.duplicated().any():
        raise ValueError("Duplicate paths or pixels in manifest")
    if frame.groupby("group")["split"].nunique().max() > 1:
        raise ValueError("Related-image group leaks between splits")
    for split, subset in frame.groupby("split"):
        if set(subset.label) != set(range(4)):
            raise ValueError(f"All four classes required in {split}")
    for row in frame.itertuples():
        if row.label not in range(4) or CLASSES[row.label] != row.class_name:
            raise ValueError("Invalid class mapping")
        if Path(row.path).is_absolute() or ".." in Path(row.path).parts:
            raise ValueError("Manifest image paths must stay inside the dataset root")
    return frame


def load_manifest(path):
    return validate_manifest(pd.read_csv(path))


def load_arrays(frame, root, size):
    root = Path(root).resolve()
    arrays = []
    for row in frame.itertuples():
        path = (root / row.path).resolve()
        if not path.is_relative_to(root) or sha256(path) != row.sha256:
            raise ValueError(f"Dataset content changed: {row.path}")
        arrays.append(read_image(path, size))
    return np.stack(arrays), frame.label.to_numpy(dtype=np.int32)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="apple_disease_classification")
    parser.add_argument("--output", default="results/four_class")
    parser.add_argument("--config", default="configs/four_class.json")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    print(json.dumps(prepare(args.root, args.output, config["seed"], config["near_duplicate_hamming"], config["validation_fraction"]), indent=2))
