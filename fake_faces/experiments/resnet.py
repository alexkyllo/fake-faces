"""Baseine model experiments"""
import os
from fake_faces.experiments.experiment import Experiment
from fake_faces.models import ResNet50
from tensorflow.keras.optimizers import Adam, SGD
import dotenv

dotenv.load_dotenv()
DATA_DIR = os.getenv("FAKE_FACES_DIR")
COMBINED_DIR_125 = os.getenv("COMBINED_DIR_125")

TRIALS = [
    # ResNet50 experiment with Adam optimizer on grayscale images
    Experiment("resnet50 adam", color_channels=1)
    .set_pipeline(
        os.path.join(DATA_DIR, "cropped/train/"),
        os.path.join(DATA_DIR, "cropped/valid/"),
    )
    .set_model(
        ResNet50,
        optimizer=Adam(learning_rate=0.001),
    ),
    Experiment("resnet50 rgb hflip adam combined 0001", color_channels=3)
    .set_pipeline(
        os.path.join(COMBINED_DIR_125, "train/"),
        os.path.join(COMBINED_DIR_125, "valid/"),
        horizontal_flip=True,
    )
    .set_model(
        ResNet50,
        optimizer=Adam(learning_rate=0.0001),
    ),
]
