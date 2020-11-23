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
    def __init__(self, path, log_path, color_channels=1, shape=SHAPE):
        """constructor"""
        super().__init__(path, log_path, color_channels, shape)

        os.makedirs(path, exist_ok=True)
        model = Sequential()
        model.add(
            VGG(
                weights=None,
                include_top=False,
                # pooling="max",
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
        opt = Adam()
        model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])
        if self.checkpoint:
            model.load_weights(self.checkpoint)
        self.model = model
