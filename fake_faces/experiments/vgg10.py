"""Baseine model experiments"""
import os
from fake_faces.experiments.experiment import Experiment
from fake_faces.models import VGG10
from tensorflow.keras.optimizers import Adam, SGD
import dotenv

dotenv.load_dotenv()
DATA_DIR = os.getenv("FAKE_FACES_DIR")
COMBINED_DIR = os.getenv("COMBINED_DIR") # fakefaces and fairface data combined
COMBINED_DIR_125 = os.getenv("COMBINED_DIR_125") # fakefaces and fairface data combined, with wider margin on fairface images

TRIALS = [
    # baseline cropped grayscale, no augmentation
    Experiment("vgg10 adam", color_channels=1)
    .set_pipeline(
        os.path.join(DATA_DIR, "cropped/train/"),
        os.path.join(DATA_DIR, "cropped/valid/"),
    )
    .set_model(
        VGG10,
        dense_dropout_rate=0.5,
        optimizer=Adam(learning_rate=0.001),
    ),
    Experiment("vgg10 sgd", color_channels=1)
    .set_pipeline(
        os.path.join(DATA_DIR, "cropped/train/"),
        os.path.join(DATA_DIR, "cropped/valid/"),
    )
    .set_model(
        VGG10,
        dense_dropout_rate=0.5,
        optimizer=SGD(learning_rate=0.001),
    ),
    Experiment("vgg10 dlib hflip", color_channels=1)
    .set_pipeline(
        os.path.join(DATA_DIR, "aligned/train/"),
        os.path.join(DATA_DIR, "aligned/valid/"),
        horizontal_flip=True,
    )
    .set_model(
        VGG10,
        dense_dropout_rate=0.5,
        optimizer=Adam(learning_rate=0.001),
    ),
    Experiment("vgg10 dlib hflip sgd", color_channels=1)
    .set_pipeline(
        os.path.join(DATA_DIR, "aligned/train/"),
        os.path.join(DATA_DIR, "aligned/valid/"),
        horizontal_flip=True,
    )
    .set_model(
        VGG10,
        dense_dropout_rate=0.5,
        optimizer=SGD(learning_rate=0.001),
    ),
    Experiment("vgg10 dlib hflip combined 0001", color_channels=1)
    .set_pipeline(
        os.path.join(COMBINED_DIR, "train/"),
        os.path.join(COMBINED_DIR, "valid/"),
        horizontal_flip=True,
    )
    .set_model(
        VGG10,
        dense_dropout_rate=0.5,
        optimizer=Adam(learning_rate=0.0001),
    ),
    Experiment("vgg10 dlib hflip combined 125 0001", color_channels=1)
    .set_pipeline(
        os.path.join(COMBINED_DIR_125, "train/"),
        os.path.join(COMBINED_DIR_125, "valid/"),
        horizontal_flip=True,
    )
    .set_model(
        VGG10,
        dense_dropout_rate=0.5,
        optimizer=Adam(learning_rate=0.0001),
    ),
]
