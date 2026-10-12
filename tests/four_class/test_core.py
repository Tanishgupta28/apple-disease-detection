import io
import numpy as np
import pytest
from PIL import Image
from apple_disease import CLASSES
from apple_disease.preprocessing import read_image
from apple_disease.features import glcm_features
from apple_disease.segmentation import segment
from apple_disease.evaluation import classification_metrics, validate_probabilities


def test_class_order():
    assert CLASSES == ("Blotch", "Rot", "Scab", "Normal")


def test_decoder_and_invalid_upload():
    stream = io.BytesIO()
    Image.new("L", (30, 40), 100).save(stream, format="PNG"); stream.seek(0)
    image = read_image(stream, 32)
    assert image.shape == (32, 32, 3) and image.dtype == np.uint8
    with pytest.raises(ValueError):
        read_image(io.BytesIO(b"not an image"))


def test_glcm_known_constant_and_texture():
    constant = np.full((16, 16, 3), 100, np.uint8)
    checker = np.repeat(((np.indices((16, 16)).sum(axis=0) % 2) * 255)[..., None], 3, axis=2).astype(np.uint8)
    features = glcm_features(constant)
    np.testing.assert_allclose(features, [0, 0, 1, 0, 1, 0, 1, 0, 0, 0], atol=1e-6)
    assert glcm_features(checker)[0] > features[0]
    with pytest.raises(ValueError):
        glcm_features(np.zeros((1, 1, 3), np.uint8))
    with pytest.raises(ValueError):
        glcm_features(constant.astype(float))


def test_segmentation_two_colors_and_reproducibility():
    image = np.zeros((16, 16, 3), np.uint8)
    image[:, :8] = (255, 0, 0); image[:, 8:] = (0, 255, 0)
    segmented, labels, centers = segment(image, 2)
    assert segmented.shape == image.shape and centers.shape == (2, 3)
    assert labels[0, 0] != labels[0, -1]
    np.testing.assert_array_equal(labels, segment(image, 2)[1])
    with pytest.raises(ValueError):
        segment(image, 9)
    with pytest.raises(ValueError):
        segment(np.zeros_like(image), 2)


def test_metrics_known_answer():
    metrics = classification_metrics(np.arange(4), np.eye(4))
    assert metrics["accuracy"] == metrics["f1_macro"] == 1
    with pytest.raises(ValueError):
        validate_probabilities(np.ones((2, 4)))
