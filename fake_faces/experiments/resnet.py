"""Baseine model experiments"""
import os
from fake_faces.experiments.experiment import Experiment
from fake_faces.models import ResNet50
from tensorflow.keras.optimizers import Adam, SGD
import dotenv

dotenv.load_dotenv()
DATA_DIR = os.getenv("FAKE_FACES_DIR")

TRIALS = [
    # baseline cropped grayscale, no augmentation
    Experiment("resnet50 adam", color_channels=1)
    .set_pipeline(
        os.path.join(DATA_DIR, "cropped/train/"),
        os.path.join(DATA_DIR, "cropped/valid/"),
    )
    .set_model(
        ResNet50,
        optimizer=Adam(learning_rate=0.001),
    ),
]
