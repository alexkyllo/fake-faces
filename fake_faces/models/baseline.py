"""baseline.py
A baseline CNN with 3 Conv2D layers"""
import os
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPool2D,
    Flatten,
    Dense,
    Dropout,
    BatchNormalization,
)

from fake_faces.models.model import Model
from fake_faces.training import SHAPE, BATCH_SIZE, CLASS_MODE


class Baseline(Model):
    def __init__(self, path, log_path, color_channels=1):
        """constructor"""
        super().__init__(path, log_path, color_channels)

        os.makedirs(path, exist_ok=True)
        model = Sequential()
        model.add(
            Conv2D(
                filters=32,
                kernel_size=(3, 3),
                input_shape=(*SHAPE, color_channels),
                activation="relu",
                padding="same",
            )
        )
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(rate=0.2))

        model.add(
            Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same")
        )
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

        model.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
        )
        if self.checkpoint:
            model.load_weights(self.checkpoint)
        self.model = model
