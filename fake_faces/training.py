"""training.py"""
import click
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout


@click.command()
def train():
    """Train the CNN model on training data and save it."""
