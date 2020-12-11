"""testing.py
Functions for testing the model and reporting metrics
"""
import os
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(gpus[0], True)
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from fake_faces import CLASS_MODE, BATCH_SIZE, SHAPE, RESCALE

# TODO: write script to output LaTeX table of performance metrics using pd.to_latex()
# TODO: write script to output LaTeX table of fairness metrics using pd.to_latex()

def get_predictions(weights_file, test_path, threshold=0.5, color_mode="grayscale"):
    """Get a saved model's predictions on a directory of images, as 1s and 0s"""
    y, y_prob, filenames = get_probabilities(
        weights_file, test_path, threshold=threshold, color_mode=color_mode
    )
    y_pred = (y_prob > threshold).astype(int)
    return (y, y_prob, filenames)

def get_probabilities(weights_file, test_path, color_mode="grayscale"):
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
    predictions = model.predict(test, steps=test_steps_per_epoch).flatten()

    return (test.classes, predictions, test.filenames)

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
    var options supported by FairFace are 'age', 'gender', and 'race'.
    Returns a DataFrame indexed by the grouping var, with 4 columns:
    tn, fp, fn, tp"""
    df_labels = pd.read_csv(label_path)
    df_labels["basename"] = df_labels.file.apply(os.path.basename)
    y, y_pred, filenames = get_predictions(model_path, test_path, threshold, color_mode)
    basenames = [os.path.basename(f) for f in filenames]
    df_results = pd.DataFrame({"y": y, "y_pred": y_pred, "basename": basenames})
    df_results = df_results.merge(df_labels, how="left", on=["basename"])
    df_cm = (
        df_results.groupby("age")
        .apply(lambda x: metrics.confusion_matrix(x.y, x.y_pred).ravel())
        .reset_index()
    )
    df_cm["tn"] = df_cm[0].apply(lambda x: x[0])
    df_cm["fp"] = df_cm[0].apply(lambda x: x[1])
    df_cm["fn"] = df_cm[0].apply(lambda x: x[2])
    df_cm["tp"] = df_cm[0].apply(lambda x: x[3])
    df_cm = df_cm.drop(0, axis=1)
    df_cm = df_cm.set_index(var)
    return df_cm
