"""model.py
Base class to provide a train() method to subclasses.
"""
from abc import ABC, abstractmethod
from fake_faces import SHAPE # default input image dimensions


class Model(ABC):
    """A base class for CNN models."""

    def __init__(self):
        self.model = None

    @abstractmethod
    def build(self, shape, color_channels, **kwargs):
        """Abstract method to build the model from hyperparameter kwargs"""

    def train(self, train_flow, valid_flow, initial_epoch, epochs, callbacks):
        """Train the model."""
        history = self.model.fit(
            train_flow,
            validation_data=(valid_flow),
            epochs=epochs,
            initial_epoch=initial_epoch,
            callbacks=callbacks,
        )
        return history
