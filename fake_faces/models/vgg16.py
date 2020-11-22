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
from tensorflow.keras.optimizers import SGD, Adam
from fake_faces.models.model import Model
from fake_faces import SHAPE

class VGG16(Model):
    def __init__(self, path, log_path, color_channels=1, shape=SHAPE):
        """constructor"""
        super().__init__(path, log_path, color_channels, shape)

        os.makedirs(path, exist_ok=True)
        model = Sequential()
        model.add(
            Conv2D(
                filters=64,
                kernel_size=(3, 3),
                input_shape=(*SHAPE, color_channels),
                activation="relu",
                padding="same",
            )
        )
        model.add(
            Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same")
        )
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(
            Conv2D(filters=128, kernel_size=(3, 3), activation="relu", padding="same")
        )
        model.add(
            Conv2D(filters=128, kernel_size=(3, 3), activation="relu", padding="same")
        )
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(
            Conv2D(filters=256, kernel_size=(3, 3), activation="relu", padding="same")
        )
        model.add(
            Conv2D(filters=256, kernel_size=(3, 3), activation="relu", padding="same")
        )
        model.add(
            Conv2D(filters=256, kernel_size=(3, 3), activation="relu", padding="same")
        )
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(
            Conv2D(filters=512, kernel_size=(3, 3), activation="relu", padding="same")
        )
        model.add(
            Conv2D(filters=512, kernel_size=(3, 3), activation="relu", padding="same")
        )
        model.add(
            Conv2D(filters=512, kernel_size=(3, 3), activation="relu", padding="same")
        )
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(
            Conv2D(filters=512, kernel_size=(3, 3), activation="relu", padding="same")
        )
        model.add(
            Conv2D(filters=512, kernel_size=(3, 3), activation="relu", padding="same")
        )
        model.add(
            Conv2D(filters=512, kernel_size=(3, 3), activation="relu", padding="same")
        )
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(BatchNormalization())
        model.add(Dense(units=128, activation="relu"))
        model.add(BatchNormalization())
        model.add(Dense(units=128, activation="relu"))
        model.add(BatchNormalization())
        model.add(Dense(units=1, activation="sigmoid"))
        #opt = SGD(lr=0.01)
        opt = Adam()
        model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])
        if self.checkpoint:
            model.load_weights(self.checkpoint)
        self.model = model
