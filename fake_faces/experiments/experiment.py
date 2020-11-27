"""experiment.py
A base class for a repeatable, parameterized model training experiment"""

import os
import datetime
import re
import logging
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, CSVLogger
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from fake_faces import SHAPE, RESCALE, BATCH_SIZE, CLASS_MODE, CHECKPOINT_FMT, EPOCH_PAT
from slugify import slugify

LOG = logging.getLogger(__name__)


class Experiment:
    """A base class for a repeatable, parameterized model training experiment"""

    def __init__(self, name, color_channels=1, shape=SHAPE):
        self.name = name
        self.slug = slugify(name, max_length=64)
        self.path = os.path.join("experiments/", self.slug)
        self.log_path = os.path.join(self.path, "logs/")
        self.checkpoint_path = os.path.join(self.path, "checkpoints/")
        self.checkpoint = self.latest_checkpoint_file
        self.color_channels = color_channels
        self.shape = shape
        self.callbacks = [self.__tensorboard(), self.__csvlogger(), self.__checkpoint()]
        self.train_path = None
        self.valid_path = None
        self.train_gen = None
        self.valid_gen = None
        self.flow_kwargs = {}
        self.model = None
        self.model_kwargs = {}

    def __str__(self):
        return f"Experiment: {self.name} (slug: {self.slug})"

    def set_pipeline(
        self, train_path, valid_path, batch_size=BATCH_SIZE, **augment_kwargs
    ):
        """Configure ImageDataGenerator hyperparameters"""
        self.train_path = train_path
        self.valid_path = valid_path
        self.train_gen = ImageDataGenerator(rescale=RESCALE, **augment_kwargs)
        self.valid_gen = ImageDataGenerator(rescale=RESCALE)
        self.flow_kwargs = dict(
            class_mode=CLASS_MODE,
            batch_size=batch_size,
            target_size=self.shape,
            color_mode=self.color_mode,
        )
        return self

    def set_model(self, model_class, **model_kwargs):
        """Configure model class and hyperparameters."""
        self.model = model_class()
        self.model_kwargs = model_kwargs
        return self

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
        csvpath = os.path.join(self.path, f"history_{self.slug}.csv")
        return CSVLogger(csvpath, separator=",", append=True)

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
        try:
            files = os.scandir(self.checkpoint_path)
        except FileNotFoundError:
            return None
        if len(list(files)) < 1:
            return None
        return os.path.join(
            os.path.abspath(self.checkpoint_path),
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

    def run(self, epochs):
        """Run the model experiment on input data for a given # of epochs."""
        # Fit the model
        self.ensure_paths()
        self.model = self.model.build(
            color_channels=self.color_channels, shape=self.shape, **self.model_kwargs
        )
        train_flow = self.train_gen.flow_from_directory(self.train_path, **self.flow_kwargs)
        valid_flow = self.valid_gen.flow_from_directory(self.valid_path, **self.flow_kwargs)
        if self.checkpoint:
            self.model.load_weights(self.checkpoint)
        history = self.model.train(
            train_flow,
            valid_flow,
            epochs=epochs,
            initial_epoch=self.initial_epoch,
            callbacks=self.callbacks,
        )
        self.checkpoint = self.latest_checkpoint_file
        return history
