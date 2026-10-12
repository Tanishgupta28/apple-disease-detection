"""Seeded K-means in perceptual Lab color space."""
import cv2
import numpy as np
from sklearn.cluster import KMeans
from .preprocessing import validate_rgb


def segment(image, k=3, seed=42):
    image = validate_rgb(image)
    if not isinstance(k, int) or not 2 <= k <= 8:
        raise ValueError("k must be an integer from 2 to 8")
    lab = cv2.cvtColor(image.astype(np.float32) / 255, cv2.COLOR_RGB2LAB)
    pixels = lab.reshape(-1, 3)
    # Bound fitting cost; predict every pixel using the fitted color centroids.
    rng = np.random.default_rng(seed)
    sample = pixels[rng.choice(len(pixels), min(10000, len(pixels)), replace=False)]
    if len(np.unique(sample, axis=0)) < k:
        raise ValueError("Image has fewer distinct sampled colors than k")
    model = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(sample)
    labels = model.predict(pixels).reshape(image.shape[:2])
    centers = model.cluster_centers_.astype(np.float32)
    segmented = cv2.cvtColor(centers[labels], cv2.COLOR_LAB2RGB)
    return np.clip(segmented * 255, 0, 255).astype(np.uint8), labels, centers
