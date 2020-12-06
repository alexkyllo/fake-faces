"""testing.py
"""
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd
from sklearn import metrics
from fake_faces import CLASS_MODE, BATCH_SIZE, SHAPE, RESCALE


def get_predictions(weights_file, test_path, threshold=0.5, color_mode="grayscale"):
    """Get model predictions for a directory of test data."""
    model = load_model(weights_file)
    test_gen = ImageDataGenerator(rescale=RESCALE)
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
    y_pred = predictions > threshold
    return (test.classes, y_pred.flatten(), test.filenames)


def make_confusion_matrix(
    weights_file, test_path, threshold=0.5, color_mode="grayscale"
):
    """Calculate confusion matrix on test data.
    Output is np.array([[tn, fp], [fn, tp]])"""
    y, y_pred, filenames = get_predictions(
        weights_file, test_path, threshold, color_mode
    )
    return metrics.confusion_matrix(y, y_pred)


def stratify(
    var, model_path, label_path, test_path, threshold=0.5, color_mode="grayscale"
):
    """Generate confusion matrix per each level of var.
    var options supported by FairFace are 'age', 'gender', and 'race'."""
    df_labels = pd.read_csv(label_path)
    df_labels["basename"] = df_labels.file.apply(os.path.basename)
    y, y_pred, filenames = get_predictions(model_path, test_path, threshold, color_mode)
    basenames = [os.path.basename(f) for f in filenames]
    df_results = pd.DataFrame({"y": y, "y_pred": y_pred, "basename": basenames})
    # TODO: join the dfs together and calculate confusion matrix
