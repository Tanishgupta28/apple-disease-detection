"""Train one predeclared experiment; checkpoint selection uses validation loss."""
import argparse
import json
import os
import time
from pathlib import Path
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "4")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from . import CLASSES
from .data import load_manifest, load_arrays, sha256
from .features import glcm_features, FEATURE_NAMES
from .models import ann, cnn, compile_model
from .evaluation import save_evaluation


def dataset(x, y, batch_size, training=False, seed=42):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if training:
        ds = ds.shuffle(len(y), seed=seed, reshuffle_each_iteration=True)
    options = tf.data.Options()
    options.threading.private_threadpool_size = 2
    return ds.batch(batch_size).with_options(options).prefetch(1)


def callbacks(directory, patience):
    return [keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
            keras.callbacks.ModelCheckpoint(str(directory / "best.keras"), monitor="val_loss", save_best_only=True),
            keras.callbacks.CSVLogger(str(directory / "history.csv")),
            keras.callbacks.TerminateOnNaN()]


def train(model_name, root, manifest, config_path, output):
    config = json.loads(Path(config_path).read_text())
    if config["classes"] != list(CLASSES):
        raise ValueError("Configuration class order is incompatible")
    output = Path(output)
    if output.exists():
        raise FileExistsError("Run already exists; use a new output path to protect experiment evidence")
    keras.utils.set_random_seed(config["seed"])
    tf.config.experimental.enable_op_determinism()
    frame = load_manifest(manifest)
    # Test images are decoded only after the complete training/selection procedure.
    partitions = {}
    for split in ("train", "validation"):
        subset = frame[frame.split == split]
        x, y = load_arrays(subset, root, config["image_size"])
        if model_name == "ann":
            x = np.stack([glcm_features(im, **config["glcm"]) for im in x])
        partitions[split] = (x.astype(np.float32), y)
    x_train, y_train = partitions["train"]
    x_val, y_val = partitions["validation"]
    weights = {i: len(y_train) / (4 * int(np.sum(y_train == i))) for i in range(4)}
    if model_name == "ann":
        model = ann(x_train)
    elif model_name == "cnn":
        model = cnn(config["image_size"], config["seed"])
    elif model_name == "resnet50":
        from .resnet import resnet50
        model = resnet50(config["image_size"], config["seed"])
    else:
        raise ValueError("Unknown model")
    output.mkdir(parents=True)
    (output / "config.json").write_text(json.dumps(config, indent=2))
    model.summary(print_fn=lambda line: None)
    with (output / "architecture.txt").open("w", encoding="utf-8") as handle:
        model.summary(print_fn=lambda line: handle.write(line + "\n"))
    if model_name == "ann":
        for split, (x, y) in partitions.items():
            features = pd.DataFrame(x, columns=FEATURE_NAMES)
            features.insert(0, "path", frame[frame.split == split].path.to_numpy())
            features["label"] = y
            features.to_csv(output / f"{split}_features.csv", index=False)
        norm = model.get_layer("training_feature_scaler")
        (output / "scaler.json").write_text(json.dumps({"mean": norm.mean.numpy().tolist(), "variance": norm.variance.numpy().tolist(), "fit_split": "train", "features": FEATURE_NAMES}, indent=2))
    params = config[model_name]
    compile_model(model, params["learning_rate"])
    start = time.perf_counter()
    train_ds = dataset(x_train, y_train, config["batch_size"], True, config["seed"])
    val_ds = dataset(x_val, y_val, config["batch_size"])
    history = model.fit(train_ds, validation_data=val_ds,
                        epochs=params.get("epochs", params.get("head_epochs")),
                        class_weight=weights, callbacks=callbacks(output, params["patience"]), verbose=2)
    histories = {"head" if model_name == "resnet50" else "training": history.history}
    selected_phase = "head" if model_name == "resnet50" else "training"
    if model_name == "resnet50":
        from .resnet import fine_tune
        # Resume from best head checkpoint, not the last epoch.
        model = keras.models.load_model(output / "best.keras")
        fine_tune(model, params["fine_tune_learning_rate"])
        fine_dir = output / "fine_tune"
        fine_dir.mkdir()
        fine_history = model.fit(train_ds, validation_data=val_ds,
                                 epochs=params["fine_tune_epochs"], class_weight=weights,
                                 callbacks=callbacks(fine_dir, params["patience"]), verbose=2)
        histories["fine_tune"] = fine_history.history
        if min(fine_history.history["val_loss"]) < min(history.history["val_loss"]):
            import shutil
            shutil.copy2(fine_dir / "best.keras", output / "best.keras")
            selected_phase = "fine_tune"
    elapsed = time.perf_counter() - start
    model = keras.models.load_model(output / "best.keras")
    validation = model.evaluate(val_ds, verbose=0, return_dict=True)
    metadata = {"model": model_name, "classes": list(CLASSES), "image_size": config["image_size"],
                "glcm": config["glcm"], "feature_names": list(FEATURE_NAMES) if model_name == "ann" else None,
                "manifest_sha256": sha256(manifest), "checkpoint_sha256": sha256(output / "best.keras"),
                "selected_phase": selected_phase, "selection_metric": "validation_loss", "validation": validation,
                "class_weights": weights, "training_seconds": elapsed, "parameters": model.count_params(),
                "tensorflow_version": tf.__version__, "seed": config["seed"],
                "gpu_devices": [d.name for d in tf.config.list_physical_devices("GPU")],
                "status": "trained", "data_scope": "recovered four-class apple fruit dataset"}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (output / "history.json").write_text(json.dumps(histories, indent=2))
    test = frame[frame.split == "test"]
    x_test, y_test = load_arrays(test, root, config["image_size"])
    if model_name == "ann":
        x_test = np.stack([glcm_features(im, **config["glcm"]) for im in x_test])
        features = pd.DataFrame(x_test, columns=FEATURE_NAMES)
        features.insert(0, "path", test.path.to_numpy()); features["label"] = y_test
        features.to_csv(output / "test_features.csv", index=False)
    probabilities = model.predict(dataset(x_test.astype(np.float32), y_test, config["batch_size"]), verbose=0)
    metrics = save_evaluation(output, test, probabilities, metadata)
    plot_history(histories, output)
    print(json.dumps({k: metrics[k] for k in ("model", "accuracy", "f1_macro", "n_images", "training_seconds")}, indent=2))
    return metrics


def plot_history(histories, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    offset = 0
    for phase, hist in histories.items():
        epochs = np.arange(1, len(hist["loss"]) + 1) + offset
        for ax, metric in zip(axes, ("loss", "accuracy")):
            ax.plot(epochs, hist[metric], label=f"{phase} train")
            ax.plot(epochs, hist["val_" + metric], label=f"{phase} validation")
            ax.set(xlabel="Epoch", ylabel=metric); ax.legend(fontsize=7)
        offset += len(epochs)
    fig.tight_layout(); fig.savefig(Path(output) / "learning_curves.png", dpi=140); plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=["ann", "cnn", "resnet50"])
    parser.add_argument("--root", default="apple_disease_classification")
    parser.add_argument("--manifest", default="results/four_class/manifest.csv")
    parser.add_argument("--config", default="configs/four_class.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    train(args.model, args.root, args.manifest, args.config, args.output)
