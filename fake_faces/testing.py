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
import click
from fake_faces import CLASS_MODE, BATCH_SIZE, SHAPE, RESCALE


def get_predictions(weights_file, test_path, threshold=0.5, color_mode="grayscale"):
    """Get a saved model's predictions on a directory of images, as 1s and 0s"""
    y, y_prob, filenames = get_probabilities(
        weights_file, test_path, color_mode=color_mode
    )
    y_pred = (y_prob > threshold).astype(int)
    return (y, y_pred, filenames)


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


def make_metrics_latex(
    weights_file, test_path, label, threshold=0.5, color_mode="grayscale"
):
    """Generate model metrics and output them to a LaTeX table."""
    y, y_pred, filenames = get_predictions(
        weights_file, test_path, threshold, color_mode
    )
    f1 = metrics.f1_score(y, y_pred)
    accuracy = metrics.accuracy_score(y, y_pred)
    precision = metrics.precision_score(y, y_pred)
    recall = metrics.recall_score(y, y_pred)

    df = pd.DataFrame(
        {"Accuracy": accuracy, "F1": f1, "Precision": precision, "Recall": recall},
        index=[1],
    )
    return df.to_latex(
        buf=None,
        index=False,
        caption="Model performance metrics",
        label=label,
    )


@click.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.argument("test_path", type=click.Path(exists=True))
@click.argument("label", type=click.STRING)
def make_metrics(model_path, test_path, label):
    """Score the model in MODEL_PATH on data in TEST_PATH and output LaTeX table."""
    click.echo(make_metrics_latex(model_path, test_path, label))


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


def get_labeled_predictions(
    model_path, label_path, test_path, threshold=0.5, color_mode="grayscale"
):
    """Get a DataFrame of y, y_pred, filename and demographic labels"""
    df_labels = pd.read_csv(label_path)
    df_labels["basename"] = df_labels.file.apply(os.path.basename)
    y, y_pred, filenames = get_predictions(model_path, test_path, threshold, color_mode)
    basenames = [os.path.basename(f) for f in filenames]
    df_results = pd.DataFrame({"y": y, "y_pred": y_pred, "basename": basenames})
    df_results = df_results.merge(df_labels, how="left", on=["basename"])
    df_results = df_results[~df_results.file.isna()]
    return df_results


def stratify(
    var, model_path, label_path, test_path, threshold=0.5, color_mode="grayscale"
):
    """Generate confusion matrix per each level of var and stack in a DataFrame.
    var options supported by FairFace are 'age', 'gender', and 'race'.
    Returns a DataFrame indexed by the grouping var, with 4 columns:
    tn, fp, fn, tp"""
    df_results = get_labeled_predictions(
        model_path, label_path, test_path, threshold, color_mode
    )
    df_cm = (
        df_results.groupby(var)
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


def stratify_cm(df, var):
    """Group the output of get_labeled_predictions into a confusion table"""
    df_cm = (
        df.groupby(var)
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


def group_privilege_var(df_cm, var, privilege_var, privilege_fn):
    """Group a confusion matrix df by privileged var, applying is_privileged_fn."""
    cm_priv = (
        df_cm.reset_index()
        .assign(
            var=var,
            privilege_var=privilege_var,
            is_privileged=privilege_fn,
        )
        .groupby(["var", "privilege_var", "is_privileged"])[["tn", "fp", "fn", "tp"]]
        .sum()
    )
    return cm_priv


def stratify_all(df):
    """Get confusion table stratified by all privilege vars."""
    cm_age = stratify_cm(df, "age")
    cm_gender = stratify_cm(df, "gender")
    cm_race = stratify_cm(df, "race")

    cm_male = group_privilege_var(
        cm_gender, "gender", "Male", lambda x: x.gender == "Male"
    )
    cm_white = group_privilege_var(
        cm_race, "race", "White", lambda x: x.race == "White"
    )
    cm_nonblack = group_privilege_var(
        cm_race, "race", "Non-Black", lambda x: x.race != "Black"
    )
    cm_nonchild = group_privilege_var(
        cm_age, "age", "Non-Child", lambda x: ~(x.age.isin(["0-2", "3-9"]))
    )
    cm_nonsenior = group_privilege_var(
        cm_age, "age", "Non-Senior", lambda x: x.age != "more than 70"
    )

    cm_all = pd.concat(
        [
            cm_male,
            cm_white,
            cm_nonblack,
            cm_nonchild,
            cm_nonsenior,
        ]
    )
    cm_all["fnr"] = cm_all["fn"] / (cm_all["fn"] + cm_all["tp"])
    cm_all["fpr"] = cm_all["fp"] / (cm_all["fp"] + cm_all["tn"])
    cm_all["tpr"] = cm_all["tp"] / (cm_all["tp"] + cm_all["fn"])
    cm_all["tnr"] = cm_all["tn"] / (cm_all["tn"] + cm_all["fp"])

    return cm_all


def disparate_impact_ratio(predictions, privileged):
    """P(yhat=1|unprivileged)/P(yhat=1|privileged)
    Expects a numpy array of binary predictions and an equal
    length numpy array of binary privilege indicators
    (where 1 = privileged class member).
    """
    ct11 = np.sum(predictions & privileged)
    ct10 = np.sum(predictions & np.logical_not(privileged))
    ct01 = np.sum(np.logical_not(predictions) & privileged)
    ct00 = np.sum(np.logical_not(predictions | privileged))
    ct = ct11 + ct10 + ct01 + ct00
    p1un = np.divide(ct10, ct10 + ct00)
    p1pr = np.divide(ct11, ct11 + ct01)
    return np.divide(p1un, p1pr)


def average_odds_difference(fpr_unpriv, fpr_priv, tpr_unpriv, tpr_priv):
    """Calculate the average odds difference fairness metric."""
    return ((fpr_unpriv - fpr_priv) + (tpr_unpriv - tpr_priv)) / 2


def fairness_metrics(df):
    """Report fairness metrics based on the DataFrame output from
    get_labeled_predictions()."""
    disparate_male = disparate_impact_ratio(df.y_pred, df.gender.eq("Male").astype(int))
    disparate_white = disparate_impact_ratio(df.y_pred, df.race.eq("White").astype(int))
    disparate_nonblack = disparate_impact_ratio(
        df.y_pred, df.race.eq("Black").astype(int)
    )
    disparate_nonchild = disparate_impact_ratio(
        df.y_pred, np.logical_not(df.age.isin(["0-2", "3-9"])).astype(int)
    )
    disparate_nonsenior = disparate_impact_ratio(
        df.y_pred, np.logical_not(df.age.eq("more than 70")).astype(int)
    )

    cm_all = stratify_all(df)

    return pd.DataFrame(
        {
            "Disparate Impact Ratio": [
                disparate_male,
                disparate_white,
                disparate_nonblack,
                disparate_nonchild,
                disparate_nonsenior,
            ],
            "Average Odds Difference": [
                average_odds_difference(
                    cm_all.loc["gender", "Male", False]["fpr"],
                    cm_all.loc["gender", "Male", True]["fpr"],
                    cm_all.loc["gender", "Male", False]["tpr"],
                    cm_all.loc["gender", "Male", True]["tpr"],
                ),
                average_odds_difference(
                    cm_all.loc["race", "White", False]["fpr"],
                    cm_all.loc["race", "White", True]["fpr"],
                    cm_all.loc["race", "White", False]["tpr"],
                    cm_all.loc["race", "White", True]["tpr"],
                ),
                average_odds_difference(
                    cm_all.loc["race", "Non-Black", False]["fpr"],
                    cm_all.loc["race", "Non-Black", True]["fpr"],
                    cm_all.loc["race", "Non-Black", False]["tpr"],
                    cm_all.loc["race", "Non-Black", True]["tpr"],
                ),
                average_odds_difference(
                    cm_all.loc["age", "Non-Child", False]["fpr"],
                    cm_all.loc["age", "Non-Child", True]["fpr"],
                    cm_all.loc["age", "Non-Child", False]["tpr"],
                    cm_all.loc["age", "Non-Child", True]["tpr"],
                ),
                average_odds_difference(
                    cm_all.loc["age", "Non-Senior", False]["fpr"],
                    cm_all.loc["age", "Non-Senior", True]["fpr"],
                    cm_all.loc["age", "Non-Senior", False]["tpr"],
                    cm_all.loc["age", "Non-Senior", True]["tpr"],
                ),
            ],
        },
        index=["Male", "White", "Non-Black", "Non-Child", "Non-Senior"],
    )


def make_fairness_metrics_latex(
    weights_file, label_path, test_path, threshold=0.5, color_mode="grayscale"
):
    """Generate model metrics and output them to a LaTeX table."""
    df_results = get_labeled_predictions(
        weights_file, label_path, test_path, threshold, color_mode
    )
    df_fairness = fairness_metrics(df_results)
    return df_fairness.to_latex(
        buf=None,
        index=True,
        caption="Model performance metrics",
        label="fairnessmetrics",
    )


@click.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.argument("label_path", type=click.Path(exists=True))
@click.argument("test_path", type=click.Path(exists=True))
@click.option("--rgb/--grayscale", default=False)
def make_fairness_metrics(model_path, label_path, test_path, rgb):
    """Calculate fairness metrics on model in MODEL_PATH on labels in LABEL_PATH
    and data in TEST_PATH and output LaTeX table."""
    color_mode = "rgb" if rgb else "grayscale"
    click.echo(
        make_fairness_metrics_latex(
            model_path, label_path, test_path, color_mode=color_mode
        )
    )
