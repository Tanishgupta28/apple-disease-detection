import json
import numpy as np
import pytest
from apple_disease import CLASSES
from apple_disease.features import FEATURE_NAMES
from apple_disease.models import ann, compile_model
from apple_disease.data import sha256
from apple_disease.inference import available_models, available_detector, load_classifier, predict_classifier


def test_missing_checkpoint_graceful(tmp_path):
    assert available_models(tmp_path) == {}
    assert available_detector(tmp_path / "absent.json") is None
    with pytest.raises(ValueError, match="missing"):
        load_classifier(tmp_path)


def test_saved_ann_inference_and_tamper_rejection(tmp_path):
    folder = tmp_path / "ann"; folder.mkdir()
    rng = np.random.default_rng(42)
    model = compile_model(ann(rng.normal(size=(16, 10)).astype("float32")))
    model.save(folder / "best.keras")
    info = {"model": "ann", "status": "trained", "classes": list(CLASSES),
            "feature_names": list(FEATURE_NAMES), "image_size": 32,
            "glcm": {"levels": 32, "distances": [1, 2, 4], "angles": [0, .785398, 1.570796, 2.356194]},
            "checkpoint_sha256": sha256(folder / "best.keras")}
    (folder / "metadata.json").write_text(json.dumps(info))
    assert list(available_models(tmp_path)) == ["ann"]
    loaded, saved_info = load_classifier(folder)
    result = predict_classifier(loaded, saved_info, rng.integers(0, 256, (40, 40, 3), dtype=np.uint8))
    assert result["class"] in CLASSES
    assert sum(result["probabilities"].values()) == pytest.approx(1, abs=1e-6)
    with (folder / "best.keras").open("ab") as handle: handle.write(b"tamper")
    with pytest.raises(ValueError, match="checksum"):
        load_classifier(folder)


def test_generic_detector_not_listed_as_disease_model(tmp_path):
    registry = tmp_path / "detector.json"
    registry.write_text(json.dumps({"status": "trained", "task": "coco_detection"}))
    assert available_detector(registry) is None
