"""baseline.py
A baseline CNN with 3 Conv2D layers"""
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPool2D,
    Flatten,
    Dense,
    Dropout,
)
from tensorflow.keras.optimizers import Adam
from fake_faces.models.model import Model
from fake_faces import SHAPE


class Baseline(Model):
    """A simple 3-layer CNN with dropout regularization to use as a baseline model"""

    def build(
        self,
        shape=SHAPE,
        color_channels=1,
        maxpool_dropout_rate=0.2,
        dense_dropout_rate=0.5,
        optimizer=Adam(),
    ):
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
        model.add(Dropout(maxpool_dropout_rate))

        model.add(
            Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same")
        )
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(maxpool_dropout_rate))

        model.add(
            Conv2D(filters=128, kernel_size=(3, 3), activation="relu", padding="same")
        )
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(maxpool_dropout_rate))

        model.add(Flatten())

        model.add(Dense(units=128, activation="relu"))
        model.add(Dropout(dense_dropout_rate))
        model.add(Dense(units=1, activation="sigmoid"))

        model.compile(
            optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"]
        )
        self.model = model
        return self
