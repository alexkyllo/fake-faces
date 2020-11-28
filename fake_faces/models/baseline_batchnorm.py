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
from tensorflow.keras.optimizers import Adam
from fake_faces.models.model import Model
from fake_faces import SHAPE, BATCH_SIZE, CLASS_MODE


class BaselineBatchNorm(Model):
    """A simple 3-layer CNN with batch normalization."""
    def build(self, shape=SHAPE, color_channels=1, momentum=0.99, normalize_fc=False, optimizer=Adam()):
        """Build the model with the given hyperparameter values."""
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
        model.add(BatchNormalization(momentum=momentum))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(
            Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same")
        )
        model.add(BatchNormalization(momentum=momentum))
        model.add(MaxPool2D(pool_size=(2, 2)))

        model.add(
            Conv2D(filters=128, kernel_size=(3, 3), activation="relu", padding="same")
        )
        model.add(BatchNormalization(momentum=momentum))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Flatten())
        if normalize_fc:
            model.add(BatchNormalization(momentum=momentum))
        model.add(Dense(units=128, activation="relu"))
        if normalize_fc:
            model.add(BatchNormalization(momentum=momentum))
        model.add(Dense(units=1, activation="sigmoid"))
        model.compile(
            optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"]
        )
        self.model = model
        return self
