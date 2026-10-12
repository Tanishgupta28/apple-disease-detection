import numpy as np
import tensorflow as tf
from apple_disease.resnet import resnet50, fine_tune
from apple_disease.models import compile_model


def test_resnet_preprocess_freeze_finetune_and_reload(tmp_path):
    tf.keras.utils.set_random_seed(42)
    model = compile_model(resnet50(32, weights=None))  # Shape test needs no download.
    x = np.random.default_rng(42).uniform(0, 255, (2, 32, 32, 3)).astype("float32")
    preprocessing = tf.keras.Model(model.input, model.get_layer("imagenet_bgr_centering").output)
    np.testing.assert_allclose(preprocessing(x, training=False),
                               tf.keras.applications.resnet50.preprocess_input(x.copy()), atol=1e-5)
    backbone = model.get_layer("resnet50")
    assert not backbone.trainable_weights
    frozen_conv = backbone.get_layer("conv1_conv").kernel.numpy().copy()
    model.train_on_batch(x, np.array([0, 1]))
    np.testing.assert_array_equal(backbone.get_layer("conv1_conv").kernel, frozen_conv)
    fine_tune(model)
    assert backbone.trainable_weights
    assert all(not layer.trainable for layer in backbone.layers if isinstance(layer, tf.keras.layers.BatchNormalization))
    bn = backbone.get_layer("conv5_block3_3_bn")
    before_bn = bn.moving_mean.numpy().copy()
    before_deep = backbone.get_layer("conv5_block3_3_conv").kernel.numpy().copy()
    model.train_on_batch(x, np.array([2, 3]))
    np.testing.assert_array_equal(bn.moving_mean, before_bn)
    np.testing.assert_array_equal(backbone.get_layer("conv1_conv").kernel, frozen_conv)
    assert not np.array_equal(before_deep, backbone.get_layer("conv5_block3_3_conv").kernel.numpy())
    probabilities = model(x, training=False).numpy()
    assert probabilities.shape == (2, 4)
    np.testing.assert_allclose(probabilities.sum(1), 1, atol=1e-6)
    model.save(tmp_path / "resnet.keras")
    restored = tf.keras.models.load_model(tmp_path / "resnet.keras")
    np.testing.assert_allclose(restored(x, training=False), probabilities, atol=1e-5)
