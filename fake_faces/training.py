"""training.py"""
import time
import datetime
import logging
import click
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint

SHAPE = (128, 128, 1)
BATCH_SIZE = 128
CLASS_MODE = "binary"
COLOR_MODE = "grayscale"


def make_generator(train=True):
    """Create an ImageDataGenerator object to read images from the
    directory and transform them on the fly"""
    params = {
        "rescale": 1.0 / 255,
    }
    if train:
        params = {
            **params,
            # "featurewise_center": True,
            # "featurewise_std_normalization": True,
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


def make_model(weights_file=None):
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
    if weights_file:
        model.load_weights(weights_file)
    return model


def check_gpu():
    logger = logging.getLogger(__name__)
    gpus = tf.config.experimental.list_physical_devices("GPU")
    logger.info("GPUs available for training: %s", gpus)
    return len(gpus) > 0


def train_model(train_path, valid_path, epochs, weights_path=None):
    """Train the CNN model on training data and save it."""
    logger = logging.getLogger(__name__)
    check_gpu()
    train_gen = make_generator(train=True)
    flow_args = dict(
        class_mode=CLASS_MODE,
        batch_size=BATCH_SIZE,
        target_size=SHAPE[0:2],
        color_mode=COLOR_MODE,
    )
    train = train_gen.flow_from_directory(train_path, **flow_args)
    val_gen = make_generator(train=False)
    val = val_gen.flow_from_directory(valid_path, **flow_args)
    model_path = "models/model.{epoch:02d}-{val_loss:.2f}.hdf5"
    log_path = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_path, histogram_freq=1)
    checkpoint = ModelCheckpoint(
        filepath=model_path,
        save_weights_only=True,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    )
    model = make_model(weights_path)
    train_start = time.time()
    model.fit(
        train,
        validation_data=(val),
        epochs=epochs,
        # steps_per_epoch=len(train),
        # validation_steps=len(test),
        callbacks=[checkpoint, tensorboard],
    )
    train_end = time.time()
    logger.info("Training completed in %.2f seconds", train_end - train_start)
    logger.info("Checkpointed model saved to %s", model_path)
    return model


@click.command()
@click.argument("train_path", type=click.Path(exists=True))
@click.argument("valid_path", type=click.Path(exists=True))
@click.argument("epochs", type=click.INT)
@click.option(
    "--weights", type=click.Path(exists=True), help="Path to a saved weights file"
)
def train(train_path, valid_path, epochs, weights):
    """Train the model on images in TRAIN_PATH and validate on VALID_PATH for # EPOCHS"""
    train_model(train_path, valid_path, epochs, weights)
