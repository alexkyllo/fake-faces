"""densenet121.py"""
import os
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import DenseNet121 as DenseNet
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Dense,
    GlobalAveragePooling2D,
    BatchNormalization,
)
from fake_faces.models.model import Model
from fake_faces import SHAPE


class DenseNet121(Model):
    """A DenseNet121 pre-built model architecture from keras"""
    def build(self, shape=SHAPE, color_channels=1, optimizer=Adam()):
        """Build the model with the given hyperparameter values."""
        densenet = DenseNet(
            weights=None, include_top=False, input_shape=(*SHAPE, color_channels)
        )

        model = Sequential(
            [
                densenet,
                GlobalAveragePooling2D(),
                Dense(512, activation="relu"),
                BatchNormalization(),
                Dense(1, activation="sigmoid"),
            ]
        )
        model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
        self.model = model
        return self
