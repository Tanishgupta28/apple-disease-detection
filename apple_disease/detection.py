"""YOLO detection workflow for real box annotations, independent of classification."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import yaml
from .preprocessing import read_image, validate_rgb
from .data import EXTENSIONS, sha256


def class_names(config):
    names = config.get("names")
    if isinstance(names, dict):
        if set(names) != set(range(len(names))):
            raise ValueError("YOLO class IDs must be contiguous integers starting at zero")
        names = [names[i] for i in range(len(names))]
    if not isinstance(names, list) or not names or any(not isinstance(n, str) or not n.strip() for n in names):
        raise ValueError("YOLO names must be a nonempty list or integer-keyed mapping")
    if len(set(names)) != len(names) or config.get("nc", len(names)) != len(names):
        raise ValueError("Duplicate class names or nc mismatch")
    return names


def resolve_images(spec, root):
    specs = spec if isinstance(spec, list) else [spec]
    images = []
    for item in specs:
        if not isinstance(item, str):
            raise ValueError("Split path must be a string or list of strings")
        path = Path(item)
        path = path if path.is_absolute() else root / path
        path = path.resolve()
        if path.is_dir():
            images.extend(sorted(p.resolve() for p in path.rglob("*") if p.suffix.lower() in EXTENSIONS))
        elif path.is_file() and path.suffix == ".txt":
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                image = Path(line.strip())
                # This project's portable convention: list entries relative to YAML root.
                images.append((image if image.is_absolute() else root / image).resolve())
        else:
            raise ValueError(f"Missing split directory/list: {path}")
    if not images or len(images) != len(set(images)):
        raise ValueError("Empty split or repeated image path")
    return images


def label_path(image):
    parts = list(image.parts)
    if "images" not in parts:
        raise ValueError("YOLO layout must contain an images directory with a parallel labels directory")
    # Replace the final images component so dataset parents may also be named images.
    index = len(parts) - 1 - parts[::-1].index("images")
    parts[index] = "labels"
    return Path(*parts).with_suffix(".txt")


def validate_yolo(yaml_path, expected_names=None, allow_background=False):
    yaml_path = Path(yaml_path).resolve()
    config = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Dataset YAML must be a mapping")
    if "download" in config:
        raise ValueError("Automatic download/script entries are not accepted; import the dataset locally")
    names = class_names(config)
    if expected_names is not None and names != list(expected_names):
        raise ValueError("Detection class mapping does not match the requested class order")
    root = Path(config.get("path", "."))
    root = (root if root.is_absolute() else yaml_path.parent / root).resolve()
    rows, seen_paths, seen_pixels, counts = [], {}, {}, {}
    total_boxes, full_frame = 0, 0
    for split in ("train", "val", "test"):
        if split not in config:
            raise ValueError(f"Dataset needs an explicit {split} split")
        images = resolve_images(config[split], root)
        box_count, class_counts = 0, [0] * len(names)
        for image in images:
            if image in seen_paths:
                raise ValueError(f"Image path overlaps splits: {image.name}")
            seen_paths[image] = split
            pixels = read_image(image)
            digest = hashlib.sha256(str(pixels.shape).encode() + pixels.tobytes()).hexdigest()
            if digest in seen_pixels:
                raise ValueError(f"Exact duplicate images in detection dataset: {image.name}")
            seen_pixels[digest] = split
            label = label_path(image)
            if not label.is_file():
                raise ValueError(f"Missing label file: {label.name}; backgrounds require an explicit empty file")
            lines = [line for line in label.read_text().splitlines() if line.strip()]
            if not lines and not allow_background:
                raise ValueError("Empty annotation: pass allow_background only for reviewed negative images")
            for line in lines:
                try:
                    values = np.array([float(v) for v in line.split()])
                except ValueError as exc:
                    raise ValueError(f"Nonnumeric bounding box in {label.name}") from exc
                if values.shape != (5,) or not np.isfinite(values).all():
                    raise ValueError("Each detection row must contain five finite numbers")
                cls, cx, cy, width, height = values
                if cls != int(cls) or not 0 <= cls < len(names):
                    raise ValueError("Bounding-box class ID is out of range")
                if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < width <= 1 and 0 < height <= 1):
                    raise ValueError("Bounding-box coordinates must be normalized and sizes positive")
                if min(cx - width / 2, cy - height / 2) < -1e-6 or max(cx + width / 2, cy + height / 2) > 1 + 1e-6:
                    raise ValueError("Bounding box extends outside the image")
                full_frame += int(width >= .98 and height >= .98)
                box_count += 1; total_boxes += 1; class_counts[int(cls)] += 1
            rows.append({"split": split, "path": str(image), "label": str(label), "sha256": sha256(image)})
        if not box_count:
            raise ValueError(f"No positive box annotations in {split}")
        counts[split] = {"images": len(images), "boxes": box_count, "class_box_counts": dict(zip(names, class_counts))}
    if total_boxes and full_frame / total_boxes > .9:
        raise ValueError("Over 90% of boxes cover the full image: annotation provenance needs review; classification labels are not box ground truth")
    return {"names": names, "counts": counts, "total_boxes": total_boxes,
            "full_frame_boxes": full_frame, "yaml_sha256": sha256(yaml_path),
            "limitations": ["Format checks cannot prove biological labels or manual annotation provenance", "Near-duplicate and source-group review is required before a defensible training benchmark"],
            "images": rows}


def train_yolo(data, output, weights="yolo11n.pt", epochs=30, image_size=416, device="cpu", allow_background=False):
    report = validate_yolo(data, allow_background=allow_background)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Detection output already exists")
    output.mkdir(parents=True)
    # Generate an explicit, sanitized dataset configuration from the validated paths.
    sanitized = {"names": report["names"], "nc": len(report["names"])}
    for split in ("train", "val", "test"):
        listing = output / f"{split}.txt"
        listing.write_text("\n".join(r["path"] for r in report["images"] if r["split"] == split), encoding="utf-8")
        sanitized[split] = str(listing)
    local_yaml = output / "validated_data.yaml"
    local_yaml.write_text(yaml.safe_dump(sanitized), encoding="utf-8")
    from ultralytics import YOLO
    model = YOLO(weights)
    if model.task != "detect":
        raise ValueError("A detection model is required")
    model.train(data=str(local_yaml), epochs=epochs, imgsz=image_size, batch=8,
                device=device, workers=0, seed=42, deterministic=True,
                project=str(output), name="train", exist_ok=False, patience=8)
    best = Path(model.trainer.best)
    metrics = YOLO(str(best)).val(data=str(local_yaml), split="test", device=device, imgsz=image_size, workers=0,
                                 project=str(output), name="test")
    result = {"precision": float(metrics.box.mp), "recall": float(metrics.box.mr),
              "mAP50": float(metrics.box.map50), "mAP50_95": float(metrics.box.map),
              "split": "test", "checkpoint_sha256": sha256(best), "names": report["names"]}
    (output / "detection_metrics.json").write_text(json.dumps(result, indent=2))
    (output / "dataset_validation.json").write_text(json.dumps(report, indent=2))
    return result


def predict_yolo(weights, image, confidence=.25, expected_names=None):
    image = validate_rgb(image)
    if not 0 < confidence <= 1:
        raise ValueError("Confidence threshold must be in (0, 1]")
    from ultralytics import YOLO
    from PIL import Image
    model = YOLO(str(weights))
    if model.task != "detect":
        raise ValueError("Checkpoint is not an object detector")
    if expected_names is not None and list(model.names.values()) != list(expected_names):
        raise ValueError("Detector checkpoint class names disagree with the registry")
    # PIL keeps the RGB convention explicit; Ultralytics numpy inputs expect BGR.
    prediction = model.predict(Image.fromarray(image), conf=confidence, device="cpu", verbose=False)[0]
    detections = [{"class": prediction.names[int(box.cls.item())], "confidence": float(box.conf.item()),
                   "xyxy": box.xyxy[0].tolist()} for box in prediction.boxes]
    return prediction.plot()[..., ::-1].copy(), detections  # plot() returns BGR.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate"); validate.add_argument("data")
    validate.add_argument("--allow-background", action="store_true")
    train = sub.add_parser("train"); train.add_argument("data"); train.add_argument("--output", required=True)
    train.add_argument("--weights", default="yolo11n.pt"); train.add_argument("--epochs", type=int, default=30)
    train.add_argument("--device", default="cpu"); train.add_argument("--allow-background", action="store_true")
    predict = sub.add_parser("predict"); predict.add_argument("image"); predict.add_argument("--weights", required=True)
    predict.add_argument("--output", default="results/local_examples/yolo_prediction.png")
    args = parser.parse_args()
    if args.command == "validate":
        report = validate_yolo(args.data, allow_background=args.allow_background)
        print(json.dumps({k: v for k, v in report.items() if k != "images"}, indent=2))
    elif args.command == "train":
        print(train_yolo(args.data, args.output, args.weights, args.epochs, device=args.device, allow_background=args.allow_background))
    else:
        from PIL import Image
        rendered, boxes = predict_yolo(args.weights, read_image(args.image))
        output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rendered).save(output)
        output.with_suffix(".json").write_text(json.dumps({"weights": args.weights, "detections": boxes, "evaluation_metrics": None}, indent=2))
        print(f"Saved {len(boxes)} detections to {output}")


if __name__ == "__main__":
    main()
