"""Baseine model experiments"""
import os
from fake_faces.experiments.experiment import Experiment
from fake_faces.models import Baseline
from tensorflow.keras.optimizers import Adam
import dotenv

dotenv.load_dotenv()
DATA = os.getenv("FAKE_FACES_DIR")

TRIALS = [
    Experiment("baseline cropped grayscale no augment", color_channels=1)
    .build_pipeline(
        os.path.join(DATA, "cropped/train/"),
        os.path.join(DATA, "cropped/valid/"),
    )
    .build_model(
        Baseline,
        maxpool_dropout_rate=0.2,
        dense_dropout_rate=0.5,
        optimizer=Adam(learning_rate=0.001),
    )
]

if __name__ == "__main__":
    for trial in TRIALS:
        trial.run(epochs=10)
