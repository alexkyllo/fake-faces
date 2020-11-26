"""experiment.py
A base class for a repeatable, parameterized model training experiment"""

import os
import datetime
import re
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, CSVLogger
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from fake_faces import SHAPE, RESCALE, BATCH_SIZE, CLASS_MODE, CHECKPOINT_FMT, EPOCH_PAT

class Experiment:
    """A base class for a repeatable, parameterized model training experiment"""

    def __init__(self, name, model, path, color_channels=1, shape=SHAPE):
        self.name=name
        self.path = path
        self.log_path = os.path.join(self.path, "logs/")
        self.checkpoint_path = os.path.join(self.path, "checkpoints/")
        self.checkpoint = self.latest_checkpoint_file
        self.model = model
        self.color_channels = color_channels
        self.shape = shape
        self.callbacks = [self.__tensorboard(), self.__csvlogger(), self.__checkpoint()]
        self.train_flow = None
        self.valid_flow = None

    def configure_generator(self, train_path, valid_path, **augment_kwargs):
        """Configure ImageDataGenerator hyperparameters"""
        train_gen = ImageDataGenerator(rescale=RESCALE, **augment_kwargs)
        valid_gen = ImageDataGenerator(rescale=RESCALE)
        flow_kwargs = dict(
            class_mode=CLASS_MODE,
            batch_size=BATCH_SIZE,
            target_size=self.shape,
            color_mode=self.color_mode,
        )
        self.train_flow = train_gen.flow_from_directory(train_path, **flow_kwargs)
        self.valid_flow = valid_gen.flow_from_directory(valid_path, **flow_kwargs)

    @property
    def color_mode(self):
        """Get color mode string from # of color channels"""
        if self.color_channels == 3:
            return "rgb"
        return "grayscale"

    def __tensorboard(self):
        """Set up the TensorBoard callback."""
        board = TensorBoard(
            log_dir=os.path.join(
                self.log_path, datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            ),
            histogram_freq=1,
        )
        return board

    def __csvlogger(self):
        """Set up the CSV Logger callback."""
        csvpath = os.path.join(self.path, f"history_{self.name}.csv")
        return CSVLogger(
            csvpath, separator=",", append=True
        )

    def __checkpoint(self):
        """Set up the ModelCheckpoint callback"""
        checkpoint = ModelCheckpoint(
            filepath=os.path.join(self.checkpoint_path, CHECKPOINT_FMT),
            save_weights_only=False,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        )
        return checkpoint

    @property
    def latest_checkpoint_file(self):
        """Get the path to the most recent checkpoint file for this experiment."""
        files = os.scandir(self.path)
        if len(files) < 1:
            return None
        return os.path.join(
            os.path.abspath(self.path),
            max(files, key=lambda x: x.stat().st_mtime).name,
        )

    @property
    def initial_epoch(self):
        """Get the starting epoch # based on where any prior run left off."""
        if self.checkpoint:
            return int(re.findall(EPOCH_PAT, self.checkpoint)[0])
        return 0

    def ensure_paths(self):
        """Ensure that the model data directories exist before training."""
        os.makedirs(self.path, exist_ok=True)
        os.makedirs(self.log_path, exist_ok=True)
        os.makedirs(self.checkpoint_path, exist_ok=True)

    def train(self, epochs):
        """Train the model on input data for a given # of epochs."""
        # Fit the model
        self.ensure_paths()
        history = self.model.fit(
            self.train_flow,
            validation_data=(self.valid_flow),
            epochs=epochs,
            initial_epoch=self.initial_epoch,
            callbacks=self.callbacks,
        )
        self.checkpoint = self.latest_checkpoint_file
        return history
