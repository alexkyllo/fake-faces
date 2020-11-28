"""baseline.py
A baseline CNN with 3 Conv2D layers"""
import os
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Dense,
    BatchNormalization,
    MaxPool2D,
    Flatten,
)
from tensorflow.keras.applications import VGG16 as VGG
from tensorflow.keras.optimizers import Adam
from fake_faces.models.model import Model
from fake_faces import SHAPE


class VGG16(Model):
    def build(self, shape=SHAPE, color_channels=1, pooling=None, optimizer=Adam()):
        """Build the model with the given hyperparameters."""
        model = Sequential()
        model.add(
            VGG(
                weights=None,
                include_top=False,
                pooling=pooling,
                input_shape=(*SHAPE, color_channels),
            )
        )
        model.add(Flatten())
        model.add(BatchNormalization())
        model.add(Dense(units=128, activation="relu"))
        model.add(BatchNormalization())
        model.add(Dense(units=128, activation="relu"))
        model.add(BatchNormalization())
        model.add(Dense(units=1, activation="sigmoid"))
        model.compile(
            optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"]
        )
        self.model = model
        return self
