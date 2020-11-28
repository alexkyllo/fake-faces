"""densenet121.py"""
import os
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Dense,
    Flatten,
    BatchNormalization,
)
from fake_faces.models.model import Model
from fake_faces import SHAPE


class DenseNet121(Model):
    """A ResNet50V2 pre-built (but w/o weights) model architecture from keras"""
    def build(self, shape=SHAPE, color_channels=1, pooling=None, optimizer=Adam()):
        """Build the model with the given hyperparameter values."""
        resnet = ResNet50V2(
            weights=None, include_top=False, pooling=pooling, input_shape=(*SHAPE, color_channels)
        )

        model = Sequential(
            [
                resnet,
                Flatten(),
                Dense(512, activation="relu"),
                BatchNormalization(),
                Dense(1, activation="sigmoid"),
            ]
        )
        model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
        self.model = model
        return self
