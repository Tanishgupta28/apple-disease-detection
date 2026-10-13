"""Verified checkpoint discovery and shared Streamlit/CLI classification inference."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from . import CLASSES
from .data import sha256
from .preprocessing import validate_rgb
from .features import glcm_features, FEATURE_NAMES
from .evaluation import validate_probabilities


def metadata(directory):
    directory = Path(directory)
    if not (directory / "metadata.json").is_file() or not (directory / "best.keras").is_file():
        raise ValueError("Checkpoint or metadata is missing; train this model first")
    try:
        info = json.loads((directory / "metadata.json").read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Cannot read checkpoint metadata") from exc
    if info.get("status") != "trained" or info.get("classes") != list(CLASSES):
        raise ValueError("Checkpoint is not a trained four-class model with the required mapping")
    if info.get("model") not in ("ann", "cnn", "resnet50"):
        raise ValueError("Unsupported classifier")
    if not isinstance(info.get("image_size"), int) or not 8 <= info["image_size"] <= 1024:
        raise ValueError("Invalid checkpoint image size")
    if info["model"] == "ann" and info.get("feature_names") != list(FEATURE_NAMES):
        raise ValueError("ANN feature order is incompatible")
    return info


def available_models(root="artifacts"):
    available = {}
    for name in ("ann", "cnn", "resnet50"):
        directory = Path(root) / name
        try:
            info = metadata(directory)
            if info["model"] == name:
                available[name] = directory
        except (ValueError, OSError):
            continue
    return available


def load_classifier(directory):
    from tensorflow import keras
    directory = Path(directory)
    info = metadata(directory)
    if sha256(directory / "best.keras") != info.get("checkpoint_sha256"):
        raise ValueError("Checkpoint checksum mismatch; restore the evaluated model")
    try:
        model = keras.models.load_model(directory / "best.keras", compile=False, safe_mode=True)
    except (OSError, ValueError) as exc:
        raise ValueError("Cannot load the saved classifier") from exc
    expected = (10,) if info["model"] == "ann" else (info["image_size"], info["image_size"], 3)
    if tuple(model.input_shape[1:]) != expected or model.output_shape[-1] != 4:
        raise ValueError("Checkpoint tensor shapes disagree with metadata")
    return model, info


def predict_classifier(model, info, image):
    image = validate_rgb(image)
    resized = np.asarray(Image.fromarray(image).resize((info["image_size"], info["image_size"]), Image.Resampling.BILINEAR))
    values = glcm_features(resized, **info["glcm"]) if info["model"] == "ann" else resized
    probabilities = validate_probabilities(model(values[None].astype(np.float32), training=False).numpy())[0]
    index = int(probabilities.argmax())
    return {"class": CLASSES[index], "confidence": float(probabilities[index]),
            "probabilities": dict(zip(CLASSES, probabilities.astype(float))),
            "low_confidence": bool(probabilities[index] < .6)}


def available_detector(registry="models/detection.json"):
    """Generic COCO weights cannot silently become a custom disease model in the UI."""
    path = Path(registry)
    if not path.is_file():
        return None
    try:
        info = json.loads(path.read_text())
        weights = Path(info["weights"])
        if info.get("status") != "trained" or info.get("task") != "apple_disease_detection":
            return None
        if not info.get("names") or not weights.is_file() or sha256(weights) != info.get("sha256"):
            return None
        return info
    except (KeyError, ValueError, OSError):
        return None
