"""Baseine model experiments"""
import os
from fake_faces.experiments.experiment import Experiment
from fake_faces.models import Baseline
from tensorflow.keras.optimizers import Adam
import dotenv

dotenv.load_dotenv()
DATA_DIR = os.getenv("FAKE_FACES_DIR")

TRIALS = [
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
    Experiment("baseline cropped color noaug", color_channels=3)
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
]
