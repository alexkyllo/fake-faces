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
from fake_faces.experiments import EXPERIMENTS
import questionary


def check_gpu():
    logger = logging.getLogger(__name__)
    gpus = tf.config.experimental.list_physical_devices("GPU")
    tf.config.experimental.set_memory_growth(gpus[0], True)
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

    if model_name not in MODELS.keys():
        raise ValueError(f"model_name must be one of: {list(MODELS.keys())}")

    model = MODELS[model_name](
        f"models/{model_name}", f"logs/{model_name}", color_channels=colors
    )
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
@click.option("--rgb/--grayscale", default=False)
def train(model_name, train_path, valid_path, epochs, rgb):
    """Train MODEL_NAME on images in TRAIN_PATH and validate on VALID_PATH for # EPOCHS"""
    colors = 3 if rgb else 1
    train_model(model_name, train_path, valid_path, epochs, colors)


@click.command()
def exp():
    """Run the experiment EXP_NAME for # EPOCHS"""
    exp_name = questionary.rawselect(
        "Which experiment would you like to run?", list(EXPERIMENTS.keys())
    ).ask()
    exp = EXPERIMENTS[exp_name]
    click.echo(f"This experiment has been run for {exp.initial_epoch} epochs so far.")
    epochs = click.prompt(
        "How many total epochs would you like to run it for?", type=int
    )
    exp.run(epochs)
