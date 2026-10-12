import numpy as np
import tensorflow as tf
from apple_disease.models import ann, cnn, compile_model


def test_ann_scaler_training_and_reload(tmp_path):
    tf.keras.utils.set_random_seed(42)
    x = np.random.default_rng(42).normal(size=(12, 10)).astype("float32")
    model = compile_model(ann(x))
    np.testing.assert_allclose(model.get_layer("training_feature_scaler").mean.numpy()[0], x.mean(0), atol=1e-6)
    model.train_on_batch(x, np.arange(12) % 4)
    probabilities = model(x, training=False).numpy()
    assert probabilities.shape == (12, 4)
    np.testing.assert_allclose(probabilities.sum(1), 1, atol=1e-6)
    model.save(tmp_path / "ann.keras")
    restored = tf.keras.models.load_model(tmp_path / "ann.keras")
    np.testing.assert_allclose(restored(x, training=False), probabilities, atol=1e-6)


def test_cnn_train_predict_reload_and_eval_determinism(tmp_path):
    tf.keras.utils.set_random_seed(42)
    model = compile_model(cnn(32))
    x = np.random.default_rng(42).uniform(0, 255, (4, 32, 32, 3)).astype("float32")
    model.train_on_batch(x, np.arange(4))
    probabilities = model(x, training=False).numpy()
    assert probabilities.shape == (4, 4)
    np.testing.assert_allclose(probabilities.sum(1), 1, atol=1e-6)
    np.testing.assert_allclose(model(x, training=False), probabilities, atol=1e-6)
    model.save(tmp_path / "cnn.keras")
    restored = tf.keras.models.load_model(tmp_path / "cnn.keras")
    np.testing.assert_allclose(restored(x, training=False), probabilities, atol=1e-6)
