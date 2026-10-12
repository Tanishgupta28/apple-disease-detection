"""Generate real local-image demonstrations and a clearly labelled toy texture plot."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .preprocessing import read_image
from .segmentation import segment
from .features import glcm_features, FEATURE_NAMES


def main():
    output = Path("results/local_examples"); output.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv("results/four_class/manifest.csv")
    sample = manifest[manifest.split == "train"].iloc[0]
    image = read_image(Path("apple_disease_classification") / sample.path, 192)
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes[0, 0].imshow(image); axes[0, 0].set_title("Original fruit")
    axes[1, 0].axis("off")
    for column, k in enumerate((2, 3, 4), 1):
        segmented, labels, centers = segment(image, k)
        axes[0, column].imshow(segmented); axes[0, column].set_title(f"K={k} color reconstruction")
        axes[1, column].imshow(labels, cmap="tab10", vmin=0, vmax=9)
        axes[1, column].set_title("Cluster IDs (not lesion labels)")
    for ax in axes.flat:
        ax.axis("off")
    fig.tight_layout(); fig.savefig(output / "kmeans_example.png", dpi=150); plt.close(fig)
    (output / "example_source.txt").write_text(sample.path + "\nDerived from local image; source license unverified.")
    pairs = pd.read_csv("results/four_class/near_duplicates.csv")
    interesting = pairs[pairs.cross_label | (~pairs.exact_pixels)].head(12)
    fig, axes = plt.subplots(len(interesting), 2, figsize=(6, len(interesting) * 2))
    for row_axes, pair in zip(axes, interesting.itertuples()):
        for ax, path in zip(row_axes, (pair.path_a, pair.path_b)):
            ax.imshow(read_image(Path("apple_disease_classification") / path, 128))
            ax.set_title(f"{path}\nd={pair.hamming}, cross-label={pair.cross_label}", fontsize=7); ax.axis("off")
    fig.tight_layout(); fig.savefig(output / "duplicate_review.png", dpi=110); plt.close(fig)
    checker = ((np.indices((96, 96)).sum(0) % 2) * 255).astype(np.uint8)
    gradient = np.tile(np.linspace(0, 255, 96, dtype=np.uint8), (96, 1))
    examples = {"constant": np.full((96, 96), 100, np.uint8), "checkerboard": checker, "gradient": gradient}
    values = {name: glcm_features(np.repeat(im[..., None], 3, 2)) for name, im in examples.items()}
    pd.DataFrame(values, index=FEATURE_NAMES).T.to_csv("results/four_class/toy_texture_features.csv")
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    for ax, (name, im) in zip(axes, examples.items()):
        ax.imshow(im, cmap="gray", vmin=0, vmax=255); ax.set_title(f"Toy: {name}\ncontrast={values[name][0]:.3f}"); ax.axis("off")
    fig.tight_layout(); fig.savefig("results/four_class/toy_texture_examples.png", dpi=140); plt.close(fig)


if __name__ == "__main__":
    main()
