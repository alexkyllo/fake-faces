"""testing.py
"""
from fake_faces import (
    CLASS_MODE,
    BATCH_SIZE,
    SHAPE,
)
from fake_faces.models.model import make_generator
from tensorflow.keras.models import load_model
import numpy as np
from sklearn import metrics


def make_confusion_matrix(weights_file, test_path, threshold=0.5, color_mode="grayscale"):
    """Calculate metrics on test data"""
    model = load_model(weights_file)
    test_gen = make_generator(train=False)
    flow_args = dict(
        class_mode=CLASS_MODE,
        batch_size=BATCH_SIZE,
        target_size=SHAPE[0:2],
        color_mode=color_mode,
        shuffle=False,
    )
    test = test_gen.flow_from_directory(test_path, **flow_args)
    test_steps_per_epoch = np.math.ceil(test.samples / test.batch_size)
    predictions = model.predict(test, steps=test_steps_per_epoch)
    predicted_classes = (predictions > threshold)
    true_classes = test.classes

    return metrics.confusion_matrix(true_classes, predicted_classes)
