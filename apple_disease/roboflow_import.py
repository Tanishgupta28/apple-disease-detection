"""Import an annotated local export or download a pinned Roboflow dataset version."""
import argparse
import contextlib
import io
import os
from pathlib import Path
import zipfile
import yaml
from .detection import validate_yolo


def import_zip(archive, destination):
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError("Import destination already exists")
    with zipfile.ZipFile(archive) as zipped:
        total = 0
        for entry in zipped.infolist():
            target = (destination / entry.filename).resolve()
            if not target.is_relative_to(destination) or (entry.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Unsafe archive path or symlink")
            total += entry.file_size
        if total > 10 * 1024**3:
            raise ValueError("Archive expands beyond 10 GiB")
        destination.mkdir(parents=True)
        zipped.extractall(destination)
    configs = list(destination.rglob("data.yaml"))
    if len(configs) != 1:
        raise ValueError("Export must contain exactly one data.yaml")
    config = configs[0]
    normalize_export_paths(config)
    return config


def normalize_export_paths(config):
    """Repair Roboflow's ../train/images convention when the images are siblings of YAML."""
    config = Path(config)
    value = yaml.safe_load(config.read_text())
    if not isinstance(value, dict):
        raise ValueError("Export YAML must be a mapping")
    changed = False
    if "path" not in value:
        for split in ("train", "val", "test"):
            spec = value.get(split)
            if isinstance(spec, str) and spec.startswith("../"):
                local = config.parent / spec[3:]
                if not (config.parent / spec).exists() and local.is_dir():
                    value[split] = spec[3:]; changed = True
    if changed:
        config.with_name("data.original.yaml").write_bytes(config.read_bytes())
        config.write_text(yaml.safe_dump(value), encoding="utf-8")


def download_version(workspace, project, version, destination):
    key = os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        raise ValueError("Set ROBOFLOW_API_KEY in the environment, or import a local export")
    if version < 1:
        raise ValueError("Choose an explicit positive dataset version")
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError("Download destination already exists")
    from roboflow import Roboflow
    # SDK status messages can include signed download URLs. Do not expose them.
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        try:
            dataset = Roboflow(api_key=key).workspace(workspace).project(project).version(version).download("yolov8", location=str(destination), overwrite=False)
        except Exception:
            raise RuntimeError("Roboflow download failed; verify credentials, project access and version") from None
    config = Path(dataset.location) / "data.yaml"
    normalize_export_paths(config)
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    local = sub.add_parser("local"); local.add_argument("archive"); local.add_argument("--output", required=True)
    remote = sub.add_parser("download")
    for name in ("workspace", "project", "output"):
        remote.add_argument("--" + name, required=True)
    remote.add_argument("--version", type=int, required=True)
    args = parser.parse_args()
    config = import_zip(args.archive, args.output) if args.command == "local" else download_version(args.workspace, args.project, args.version, args.output)
    report = validate_yolo(config)
    print(f"Imported {config}: {report['counts']}")


if __name__ == "__main__":
    main()
