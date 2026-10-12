from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from PIL import Image
from apple_disease import CLASSES
from apple_disease.data import prepare, load_manifest, load_arrays, validate_manifest


def make_dataset(root):
    rng = np.random.default_rng(100)
    for split, n in (("Train", 5), ("Test", 2)):
        for name in CLASSES:
            folder = root / split / (name + "_Apple")
            folder.mkdir(parents=True)
            for i in range(n):
                Image.fromarray(rng.integers(0, 256, (20, 20, 3), dtype=np.uint8)).save(folder / f"{i}.png")


def test_audit_dedup_split_reproducibility_and_integrity(tmp_path):
    root = tmp_path / "raw"
    make_dataset(root)
    test_image = root / "Test/Blotch_Apple/0.png"
    duplicate = root / "Train/Blotch_Apple/duplicate.png"
    duplicate.write_bytes(test_image.read_bytes())
    (root / "Train/Rot_Apple/corrupt.jpg").write_bytes(b"broken")
    with Image.open(root / "Train/Scab_Apple/0.png") as im:
        im.transpose(Image.Transpose.ROTATE_90).save(root / "Train/Scab_Apple/rotated.png")
    report = prepare(root, tmp_path / "first", threshold=0)
    assert len(report["corrupt_files"]) == 1
    assert report["status_counts"]["exact_duplicate"] == 1
    prepare(root, tmp_path / "second", threshold=0)
    assert (tmp_path / "first/manifest.csv").read_bytes() == (tmp_path / "second/manifest.csv").read_bytes()
    frame = load_manifest(tmp_path / "first/manifest.csv")
    assert frame.groupby("group")["split"].nunique().max() == 1
    images, labels = load_arrays(frame[frame.split == "test"], root, 16)
    assert images.shape == (8, 16, 16, 3) and labels.shape == (8,)
    test_image.write_bytes(b"modified")
    with pytest.raises(ValueError, match="content changed"):
        load_arrays(frame[frame.split == "test"], root, 16)
    with pytest.raises(FileExistsError):
        prepare(root, tmp_path / "first")


def test_manifest_rejects_leakage_and_mapping(tmp_path):
    make_dataset(tmp_path / "raw")
    prepare(tmp_path / "raw", tmp_path / "out", threshold=0)
    frame = load_manifest(tmp_path / "out/manifest.csv")
    bad = frame.copy()
    bad.loc[bad.split == "train", "group"] = "shared"
    bad.loc[bad.split == "test", "group"] = "shared"
    with pytest.raises(ValueError, match="leaks"):
        validate_manifest(bad)
    bad = frame.copy(); bad.loc[0, "class_name"] = "Rust"
    with pytest.raises(ValueError, match="class mapping"):
        validate_manifest(bad)
