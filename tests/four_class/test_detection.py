import zipfile
import numpy as np
import pytest
import yaml
from PIL import Image
from apple_disease.detection import validate_yolo
from apple_disease.roboflow_import import import_zip, download_version


def fixture_dataset(root):
    rng = np.random.default_rng(20)
    for split in ("train", "val", "test"):
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir()
        Image.fromarray(rng.integers(0, 256, (24, 24, 3), dtype=np.uint8)).save(root / split / "images/a.png")
        (root / split / "labels/a.txt").write_text("0 0.5 0.5 0.4 0.6\n")
    config = root / "data.yaml"
    config.write_text(yaml.safe_dump({"names": ["apple"], "nc": 1, "train": "train/images", "val": "val/images", "test": "test/images"}))
    return config


def test_valid_real_box_format_and_mapping(tmp_path):
    config = fixture_dataset(tmp_path)
    result = validate_yolo(config, expected_names=["apple"])
    assert result["total_boxes"] == 3
    assert result["counts"]["test"]["images"] == 1
    with pytest.raises(ValueError, match="class mapping"):
        validate_yolo(config, expected_names=["disease"])


@pytest.mark.parametrize("label", ["4 0.5 0.5 0.4 0.6", "0 nan 0.5 0.4 0.6", "0 0.9 0.5 0.8 0.6", "0 0.5 0.5 -1 0.6", "0 0.5 0.5", "", "hello", "0.5 0.5 0.5 0.4 0.6"])
def test_invalid_boxes_rejected(tmp_path, label):
    config = fixture_dataset(tmp_path)
    (tmp_path / "train/labels/a.txt").write_text(label)
    with pytest.raises(ValueError):
        validate_yolo(config)


def test_fake_full_frame_labels_rejected(tmp_path):
    config = fixture_dataset(tmp_path)
    for split in ("train", "val", "test"):
        (tmp_path / split / "labels/a.txt").write_text("0 0.5 0.5 1 1")
    with pytest.raises(ValueError, match="full image"):
        validate_yolo(config)


def test_missing_labels_and_duplicate_leakage(tmp_path):
    config = fixture_dataset(tmp_path)
    target = tmp_path / "val/images/a.png"
    target.write_bytes((tmp_path / "train/images/a.png").read_bytes())
    with pytest.raises(ValueError, match="duplicate"):
        validate_yolo(config)
    (tmp_path / "train/labels/a.txt").unlink()
    with pytest.raises(ValueError, match="Missing label"):
        validate_yolo(config)


def test_split_overlap_and_script_yaml_rejected(tmp_path):
    config = fixture_dataset(tmp_path)
    value = yaml.safe_load(config.read_text()); value["test"] = "train/images"
    config.write_text(yaml.safe_dump(value))
    with pytest.raises(ValueError, match="overlaps"):
        validate_yolo(config)
    value["download"] = "do not execute"; config.write_text(yaml.safe_dump(value))
    with pytest.raises(ValueError, match="script"):
        validate_yolo(config)


def test_zip_import_and_path_traversal(tmp_path):
    source = tmp_path / "source"; fixture_dataset(source)
    archive = tmp_path / "export.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        for p in source.rglob("*"):
            if p.is_file(): handle.write(p, p.relative_to(source))
    config = import_zip(archive, tmp_path / "imported")
    assert validate_yolo(config)["total_boxes"] == 3
    with zipfile.ZipFile(tmp_path / "unsafe.zip", "w") as handle:
        handle.writestr("../escape.txt", "bad")
    with pytest.raises(ValueError, match="Unsafe"):
        import_zip(tmp_path / "unsafe.zip", tmp_path / "unsafe")


def test_roboflow_missing_credentials(monkeypatch, tmp_path):
    monkeypatch.delenv("ROBOFLOW_API_KEY", raising=False)
    with pytest.raises(ValueError, match="ROBOFLOW_API_KEY"):
        download_version("workspace", "project", 1, tmp_path / "download")


def test_roboflow_relative_paths(tmp_path):
    from apple_disease.roboflow_import import normalize_export_paths
    config = fixture_dataset(tmp_path)
    value = yaml.safe_load(config.read_text())
    for split in ("train", "val", "test"):
        value[split] = "../" + value[split]
    config.write_text(yaml.safe_dump(value))
    normalize_export_paths(config)
    assert (tmp_path / "data.original.yaml").exists()
    assert validate_yolo(config)["total_boxes"] == 3


def test_training_wrapper_uses_best_checkpoint_and_test_split(tmp_path, monkeypatch):
    # An API contract test with a fake backend, explicitly NOT measured YOLO training.
    import sys
    from types import SimpleNamespace
    from apple_disease.detection import train_yolo
    config = fixture_dataset(tmp_path / "data")
    calls = []
    class FakeYOLO:
        task = "detect"
        def __init__(self, weights): calls.append(("load", weights))
        def train(self, **kwargs):
            calls.append(("train", kwargs))
            best = tmp_path / "best.pt"; best.write_bytes(b"test-only mock")
            self.trainer = SimpleNamespace(best=str(best))
        def val(self, **kwargs):
            calls.append(("val", kwargs))
            return SimpleNamespace(box=SimpleNamespace(mp=.1, mr=.2, map50=.3, map=.4))
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=FakeYOLO))
    train_yolo(config, tmp_path / "run", epochs=1)
    assert calls[2] == ("load", str(tmp_path / "best.pt"))
    assert calls[3][1]["split"] == "test"
    assert calls[1][1]["seed"] == 42
