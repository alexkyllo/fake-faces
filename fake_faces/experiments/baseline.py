"""Baseine model experiments"""
import os
from fake_faces.experiments.experiment import Experiment
from fake_faces.models import (Baseline, BaselineBatchNorm)
from tensorflow.keras.optimizers import Adam, SGD
import dotenv

dotenv.load_dotenv()
DATA_DIR = os.getenv("FAKE_FACES_DIR")

TRIALS = [
    # baseline cropped grayscale, no augmentation
    Experiment("baseline cropped grayscale noaug", color_channels=1)
    .set_pipeline(
        os.path.join(DATA_DIR, "cropped/train/"),
        os.path.join(DATA_DIR, "cropped/valid/"),
    )
    .set_model(
        Baseline,
        maxpool_dropout_rate=0.2,
        dense_dropout_rate=0.5,
        optimizer=Adam(learning_rate=0.001),
    ),
    # baseline cropped color, no augmentation
    Experiment("baseline cropped color", color_channels=3)
    .set_pipeline(
        os.path.join(DATA_DIR, "cropped/train/"),
        os.path.join(DATA_DIR, "cropped/valid/"),
    )
    .set_model(
        Baseline,
        maxpool_dropout_rate=0.2,
        dense_dropout_rate=0.5,
        optimizer=Adam(learning_rate=0.001),
    ),
    # baseline cropped grayscale, no augmentation, SGD
    Experiment("baseline sgd", color_channels=1)
    .set_pipeline(
        os.path.join(DATA_DIR, "cropped/train/"),
        os.path.join(DATA_DIR, "cropped/valid/"),
    )
    .set_model(
        Baseline,
        maxpool_dropout_rate=0.2,
        dense_dropout_rate=0.5,
        optimizer=SGD(learning_rate=0.001),
    ),
    # baseline cropped grayscale, batch normalization instead of dropout
    Experiment("baseline crop gray batchnorm", color_channels=1)
    .set_pipeline(
        os.path.join(DATA_DIR, "cropped/train/"),
        os.path.join(DATA_DIR, "cropped/valid/"),
    )
    .set_model(
        BaselineBatchNorm,
        optimizer=Adam(learning_rate=0.001),
    ),
    # baseline cropped grayscale, also batch normalize the fully connected layers
    Experiment("baseline crop gray batchnorm fc", color_channels=1)
    .set_pipeline(
        os.path.join(DATA_DIR, "cropped/train/"),
        os.path.join(DATA_DIR, "cropped/valid/"),
    )
    .set_model(
        BaselineBatchNorm,
        normalize_fc=True,
        optimizer=Adam(learning_rate=0.001),
    ),
    # baseline cropped grayscale, also batch normalize the fc layers, h. flip training images
    Experiment("baseline crop gray batchnorm fc hflip", color_channels=1)
    .set_pipeline(
        os.path.join(DATA_DIR, "cropped/train/"),
        os.path.join(DATA_DIR, "cropped/valid/"),
        horizontal_flip=True,
    )
    .set_model(
        BaselineBatchNorm,
        normalize_fc=True,
        optimizer=Adam(learning_rate=0.001),
    ),
]
