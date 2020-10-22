"""training.py"""
import logging
import click
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout


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


@click.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path())
def train(input_path, model_path):
    """Train the CNN model on training data and save it."""
    logger = logging.getLogger(__name__)
    logger.info(
        "GPUs available: %s", tf.config.experimental.list_physical_devices("GPU")
    )
    gen = make_generator(train=True)
    # TODO
