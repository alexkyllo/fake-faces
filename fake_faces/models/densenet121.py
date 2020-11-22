"""baseline.py
A baseline CNN with 3 Conv2D layers"""
import os
from tensorflow.keras.optimizers import Adam#, SGD
from tensorflow.keras.applications import DenseNet121 as DenseNet
from fake_faces.models.model import Model
from fake_faces import SHAPE

class DenseNet121(Model):
    def __init__(self, path, log_path, color_channels=1, shape=SHAPE):
        """constructor"""
        super().__init__(path, log_path, color_channels, shape)

        os.makedirs(path, exist_ok=True)
        model = DenseNet(weights=None, input_shape=(*SHAPE, color_channels), classes=2)
        #opt = SGD(lr=0.01)
        opt = Adam()
        model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])
        if self.checkpoint:
            model.load_weights(self.checkpoint)
        self.model = model
