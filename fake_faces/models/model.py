"""model.py
Base class to provide a train() method to subclasses.
"""
import os
import re
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from fake_faces import SHAPE, BATCH_SIZE, CLASS_MODE, CHECKPOINT_FMT, EPOCH_PAT


def make_generator(train=True):
    """Create an ImageDataGenerator object to read images from the
    directory and transform them on the fly"""
    params = {
        "rescale": 1.0 / 255,
    }
    # Randomly shear, zoom, rotate, shift, and flip training images
    if train:
        params = {
            **params,
            "rotation_range": 30,
            "shear_range": 0.2,
            "zoom_range": 0.2,
            "width_shift_range": 0.2,
            "height_shift_range": 0.2,
            "horizontal_flip": True,
        }
    gen = ImageDataGenerator(
        **params,
    )
    return gen


class Model:
    """A base class for CNN models."""

    def __init__(self, path, log_path, color_channels=1):
        self.path = path
        self.log_path = log_path
        self.checkpoint = self.latest_checkpoint()
        self.model = None
        self.color_channels = color_channels
        self.color_mode = "rgb" if color_channels == 3 else "grayscale"

    def latest_checkpoint(self):
        return os.path.join(
            os.path.abspath(self.path),
            max(
                [f for f in os.scandir(self.path)], key=lambda x: x.stat().st_mtime
            ).name,
        )

    def train(self, train_path, valid_path, epochs):
        """Train the model on input ImageDataGenerator"""
        train_gen = make_generator(train=True)
        valid_gen = make_generator(train=False)
        flow_args = dict(
            class_mode=CLASS_MODE,
            batch_size=BATCH_SIZE,
            target_size=SHAPE,
            color_mode=self.color_mode,
        )
        train_flow = train_gen.flow_from_directory(train_path, **flow_args)
        valid_flow = valid_gen.flow_from_directory(valid_path, **flow_args)
        tensorboard = TensorBoard(log_dir=self.log_path, histogram_freq=1)
        checkpoint = ModelCheckpoint(
            filepath=os.path.join(self.path, CHECKPOINT_FMT),
            save_weights_only=False,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        )
        if self.checkpoint:
            initial_epoch = int(re.findall(EPOCH_PAT, self.checkpoint)[0])
        else:
            initial_epoch = 0
        # Fit the model
        history = self.model.fit(
            train_flow,
            validation_data=(valid_flow),
            epochs=epochs,
            initial_epoch=initial_epoch,
            callbacks=[checkpoint, tensorboard],
        )
        self.checkpoint = self.latest_checkpoint()
        return history
