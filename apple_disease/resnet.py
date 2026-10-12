"""Official Keras ResNet50 ImageNet transfer learning and cautious fine-tuning."""
from tensorflow import keras
from .models import augmentation, compile_model


def resnet50(size=96, seed=42, weights="imagenet"):
    inputs = keras.Input(shape=(size, size, 3), name="rgb_pixels_0_255")
    x = augmentation(seed)(inputs)
    # Keras ResNet V1 preprocessing: RGB -> BGR, then subtract ImageNet means.
    # Built-in operations/layers permit safe .keras loading without Lambda code.
    x = keras.ops.flip(x, axis=-1)
    x = keras.layers.Rescaling(1., offset=[-103.939, -116.779, -123.68], name="imagenet_bgr_centering")(x)
    backbone = keras.applications.ResNet50(include_top=False, weights=weights,
                                          input_shape=(size, size, 3))
    backbone.trainable = False
    x = backbone(x, training=False)  # Keep BN moving statistics fixed in both phases.
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(.3)(x)
    outputs = keras.layers.Dense(4, activation="softmax", name="fruit_probabilities")(x)
    return keras.Model(inputs, outputs, name="resnet50_classifier")


def fine_tune(model, learning_rate=1e-5):
    """Unfreeze the final residual block only; all BatchNormalization stays frozen."""
    backbone = model.get_layer("resnet50")
    backbone.trainable = True
    for layer in backbone.layers:
        layer.trainable = layer.name.startswith("conv5_block3") and not isinstance(layer, keras.layers.BatchNormalization)
    return compile_model(model, learning_rate)
