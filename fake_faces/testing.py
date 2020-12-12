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

# TODO: write script to output LaTeX table of fairness metrics using pd.to_latex()


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
        cm_age, "race", "Non-Senior", lambda x: x.age != "more than 70"
    )


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


def fairness_metrics(df):
    """Report fairness metrics based on the DataFrame output from
    get_labeled_predictions()."""
    cm_age = stratify_cm(df, "age")
    cm_gender = stratify_cm(df, "gender")
    cm_race = stratify_cm(df, "race")

    disparate_male = disparate_impact_ratio(df.y_pred, df.gender.eq("Male").astype(int))
    disparate_white = disparate_impact_ratio(df.y_pred, df.race.eq("White").astype(int))
    disparate_nonblack = disparate_impact_ratio(
        df.y_pred, df.race.eq("Black").astype(int)
    )
    disparate_nonchild = disparate_impact_ratio(
        df.y_pred, ~df.age.isin(["0-2", "3-9"]).astype(int)
    )
    disparate_nonsenior = disparate_impact_ratio(
        df.y_pred, ~df.age.eq("more than 70").astype(int)
    )

    fn_male = cm_gender.loc["Male", "fn"]
    fnr_male = fn_male / (fn_male + cm_gender.loc["Male", "tp"])
    fn_nonmale = cm_gender[cm_gender.index != "Male", "fn"].sum()
    fnr_nonmale = (
        fn_nonmale / (fn_nonmale + cm_gender[cm_gender.index != "Male", "fn"]).sum()
    )

    fn_white = cm_race.loc["White", "fn"]
    fnr_white = fn_white / (fn_white + cm_race.loc["White", "tp"])
    fn_nonwhite = cm_race[cm_race.index != "White", "fn"]
    fnr_nonwhite = fn_nonwhite / (fn_nonwhite + cm_race[cm_race.index != "White", "tp"])

    fn_black = cm_race.loc["Black", "fn"]
    fnr_black = fn_black / (fn_black + cm_race.loc["Black", "tp"])
    fn_nonblack = cm_race[cm_race.index != "Black", "fn"]
    fnr_nonblack = fn_nonblack / (fn_nonblack + cm_race[cm_race.index != "Black", "tp"])

    fn_child = cm_age.loc[["0-2", "3-9"], "fn"]
    fnr_child = fn_child / (fn_child + cm_age.loc[["0-2", "3-9"], "tp"])
    fn_nonchild = cm_age[cm_age.index != "child", "fn"]
    fnr_nonchild = fn_nonchild / (
        fn_nonchild + cm_age[cm_age.index.isin(["0-2", "3-9"], "tp")]
    )

    fn_senior = cm_age.loc["more than 70", "fn"]
    fnr_senior = fn_senior / (fn_senior + cm_age.loc["more than 70", "tp"])
    fn_nonsenior = cm_age[cm_age.index != "senior", "fn"]
    fnr_nonsenior = fn_nonsenior / (
        fn_nonsenior + cm_age[cm_age.index != "senior", "tp"]
    )

    # TODO: FINISH THIS
    return pd.DataFrame(
        {
            "Disparate Impact": [
                disparate_male,
                disparate_white,
                disparate_nonblack,
                disparate_nonchild,
                disparate_nonsenior,
            ],
            "FNR": [
                fnr_male,
                fnr_nonmale,
                fnr_white,
                fnr_nonblack,
                fnr_nonchild,
                fnr_nonsenior,
            ],
            "FPR": [
                fpr_male,
                fpr_nonmale,
                fpr_white,
                fpr_nonblack,
                fpr_nonchild,
                fpr_nonsenior,
            ],
        },
        index=["Male", "White", "Non-Black", "Age > 9", "Age < 70"],
    )

    # TODO: finish this