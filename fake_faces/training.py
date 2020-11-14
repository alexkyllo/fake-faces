"""training.py
DONE: Try adding batch normalization
DONE: Try doubling up the Conv2D layers
DONE: Try training on RGB images instead of grayscale
TODO: Try Inception, Xception, ResNet, EfficientNet based architectures
"""
import os
import time
import datetime
import re
import logging
import click
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPool2D,
    Flatten,
    Dense,
    Dropout,
    BatchNormalization,
)
from tensorflow.keras.callbacks import ModelCheckpoint
from fake_faces import (
    SHAPE,
    BATCH_SIZE,
    CLASS_MODE,
    CHECKPOINT_FMT,
)
from fake_faces.models import MODELS


def check_gpu():
    logger = logging.getLogger(__name__)
    gpus = tf.config.experimental.list_physical_devices("GPU")
    logger.info("GPUs available for training: %s", gpus)
    return len(gpus) > 0


def train_model(
    model_name,
    train_path,
    valid_path,
    epochs,
    colors=1,
):
    """Train the CNN model on training data and save it."""
    logger = logging.getLogger(__name__)
    check_gpu()

    if model_name not in MODELS.keys():
        raise ValueError(f"model_name must be one of: {list(MODELS.keys())}")

    model = MODELS[model_name]("models/{model_name}", "logs/{model_name}")
    train_start = time.time()
    model.train(train_path, valid_path, epochs)
    train_end = time.time()
    logger.info("Training completed in %.2f seconds", train_end - train_start)
    logger.info("Checkpointed model saved to %s", model.path)
    return model


@click.command()
@click.argument("model_name", type=click.Choice(MODELS.keys()))
@click.argument("train_path", type=click.Path(exists=True))
@click.argument("valid_path", type=click.Path(exists=True))
@click.argument("epochs", type=click.INT)
@click.option("--colors", type=click.Choice([1, 3]), default=1)
def train(model_name, train_path, valid_path, epochs, colors):
    """Train the model on images in TRAIN_PATH and validate on VALID_PATH for # EPOCHS"""
    train_model(model_name, train_path, valid_path, epochs, colors)
