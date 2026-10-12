"""Small Keras baselines. Image models receive RGB float32 pixels in [0, 255]."""
import tensorflow as tf
from tensorflow import keras


def compile_model(model, learning_rate=1e-3):
    model.compile(optimizer=keras.optimizers.Adam(learning_rate),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def ann(training_features):
    """Adapt normalization ONLY on the caller's training partition; saved in model."""
    norm = keras.layers.Normalization(name="training_feature_scaler")
    norm.adapt(training_features)
    model = keras.Sequential([
        keras.Input(shape=(training_features.shape[1],)), norm,
        keras.layers.Dense(32, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-4)),
        keras.layers.Dropout(.25), keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(4, activation="softmax")], name="glcm_ann")
    return model


def augmentation(seed=42):
    return keras.Sequential([
        keras.layers.RandomFlip("horizontal", seed=seed),
        keras.layers.RandomRotation(.04, seed=seed + 1),
        keras.layers.RandomZoom(.1, seed=seed + 2)], name="training_augmentation")


def cnn(size=96, seed=42):
    inputs = keras.Input(shape=(size, size, 3))
    x = augmentation(seed)(inputs)
    x = keras.layers.Rescaling(1 / 255)(x)
    for filters in (16, 32, 64):
        x = keras.layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.MaxPooling2D()(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(.3)(x)
    return keras.Model(inputs, keras.layers.Dense(4, activation="softmax")(x), name="custom_cnn")
