"""testing.py
"""
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
    y_pred = (predictions > threshold).flatten().astype(int)
    return (test.classes, y_pred, test.filenames)


def make_confusion_matrix(
    weights_file, test_path, threshold=0.5, color_mode="grayscale"
):
    """Calculate confusion matrix on test data.
    Output is np.array([[tn, fp], [fn, tp]])"""
    y, y_pred, filenames = get_predictions(
        weights_file, test_path, threshold, color_mode
    )
    return metrics.confusion_matrix(y, y_pred)


def plot_learning_curves(history_file, loss=False):
    """Plot the train vs. validation learning curves from a history file."""
    df = pd.read_csv(history_file)
    fig, ax = plt.subplots()
    if loss:
        y = df.loss
        y_val = df.val_loss
        label = "Loss"
    else:
        y = df.accuracy
        y_val = df.val_accuracy
        label = "Accuracy"
    ax.plot(df.epoch, y, label=f"Training {label}")
    ax.plot(df.epoch, y_val, label=f"Validation {label}")
    ax.set_title("Learning Curve")
    ax.set_xlabel("Training Epochs")
    ax.set_ylabel(f"{label} Score")
    ax.legend(loc="best")
    fig.tight_layout()
    return ax


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
