"""baseline.py
A baseline CNN with 3 Conv2D layers"""
import os
from tensorflow.keras.optimizers import Adam  # , SGD
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
    def __init__(self, path, log_path, color_channels=1, shape=SHAPE):
        """constructor"""
        super().__init__(path, log_path, color_channels, shape)

        os.makedirs(path, exist_ok=True)
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
        # opt = SGD(lr=0.01)
        opt = Adam(lr=0.001)
        model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])
        if self.checkpoint:
            model.load_weights(self.checkpoint)
        self.model = model
