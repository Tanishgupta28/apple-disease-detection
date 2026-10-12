"""Quantized GLCM texture baseline; no biological lesion assumptions."""
import numpy as np
from skimage.color import rgb2gray
from skimage.feature import graycomatrix, graycoprops
from .preprocessing import validate_rgb

PROPERTIES = ("contrast", "homogeneity", "energy", "correlation", "dissimilarity")
FEATURE_NAMES = tuple(f"{p}_{stat}" for p in PROPERTIES for stat in ("mean", "std"))


def glcm_features(image, levels=32, distances=(1, 2, 4),
                  angles=(0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)):
    """Ten features: mean/std over 3 distances x 4 angles for each property."""
    image = validate_rgb(image)
    if not isinstance(levels, int) or not 2 <= levels <= 256:
        raise ValueError("levels must be an integer from 2 to 256")
    if not distances or any(int(d) != d or d < 1 or d >= min(image.shape[:2]) for d in distances):
        raise ValueError("Distances must be positive integers smaller than image dimensions")
    if len(angles) == 0 or not np.isfinite(angles).all():
        raise ValueError("Angles must be finite and nonempty")
    gray = np.minimum((rgb2gray(image) * levels).astype(np.uint16), levels - 1)
    matrix = graycomatrix(gray, distances=list(distances), angles=list(angles),
                         levels=levels, symmetric=True, normed=True)
    result = []
    for prop in PROPERTIES:
        values = graycoprops(matrix, prop)
        result.extend([float(values.mean()), float(values.std())])
    vector = np.asarray(result, dtype=np.float32)
    if not np.isfinite(vector).all():
        raise ValueError("Nonfinite texture features")
    return vector
