import os
from fake_faces.experiments.experiment import Experiment
from fake_faces.models import DenseNet121
from tensorflow.keras.optimizers import Adam
import dotenv

dotenv.load_dotenv()
DATA_DIR = os.getenv("FAKE_FACES_DIR")
COMBINED_DIR = os.getenv("COMBINED_DIR")  # fakefaces and fairface data combined
COMBINED_DIR_125 = os.getenv(
    "COMBINED_DIR_125"
)  # fakefaces and fairface data combined, with wider margin on fairface images

TRIALS = [
    Experiment("densenet121 dlib hflip rgb combined 125 0001", color_channels=3)
    .set_pipeline(
        os.path.join(COMBINED_DIR_125, "train/"),
        os.path.join(COMBINED_DIR_125, "valid/"),
        horizontal_flip=True,
    )
    .set_model(
        DenseNet121,
        optimizer=Adam(learning_rate=0.0001),
    ),
]
