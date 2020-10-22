"""training.py"""
import os
import time
import logging
import click
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout

SHAPE = (64, 64)


def make_generator(train=True):
    """Create an ImageDataGenerator object to read images from the
    directory and transform them on the fly"""
    params = {
        "rescale": 1.0 / 255,
    }
    if train:
        params = {
            **params,
            "featurewise_center": True,
            "featurewise_std_normalization": True,
            "rotation_range": 30,
            "shear_range": 0.2,
            "zoom_range": 0.2,
            "width_shift_range": 0.2,
            "height_shift_range": 0.2,
            "horizontal_flip": True,
        }
    gen = ImageDataGenerator(
        **params,
    )
    return gen


def make_model():
    """Build a CNN model using Keras Sequential API"""
    model = Sequential()
    model.add(
        Conv2D(
            filters=32,
            kernel_size=(3, 3),
            input_shape=SHAPE,
            activation="relu",
            padding="same",
        )
    )
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.2))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same"))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.2))

    model.add(
        Conv2D(filters=128, kernel_size=(3, 3), activation="relu", padding="same")
    )
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.2))

    model.add(Flatten())

    model.add(Dense(units=128, activation="relu"))
    model.add(Dropout(rate=0.5))
    model.add(Dense(units=1, activation="sigmoid"))

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


@click.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.argument("model_path", type=click.Path())
@click.argument("epochs", type=click.INT)
def train(input_path, model_path, epochs):
    """Train the CNN model on training data and save it."""
    logger = logging.getLogger(__name__)
    logger.info(
        "GPUs available for training: %s",
        tf.config.experimental.list_physical_devices("GPU"),
    )
    train_gen = make_generator(train=True)
    train = gen.flow_from_directory(
        input_path,
        class_mode="binary",
        batch_size=64,
        target_size=SHAPE,
        color_mode="grayscale",
    )
    model = make_model()
    train_start = time.time()
    model.fit_generator(train, epochs=epochs, steps_per_epoch=len(train))
    train_end = time.time()
    logger.info("Training completed in %.2f seconds", train_end - train_start)
    model.save(model_path)
    logger.info("Trained model saved to %s", model_path)
