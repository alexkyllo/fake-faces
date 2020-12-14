"""explaining.py
Visually explain model predictions
# NOTE: We were not able to get this code working due to an issue with the eli5 package.
"""
import eli5
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
from tensorflow.keras.models import load_model
from fake_faces.training import SHAPE
from tensorflow import keras
import tensorflow as tf
import click
gpus = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(gpus[0], True)

def explain_image(model, color_mode, image_path):
    """Plot on an image a visualization of the areas used to classify it."""
    tf.compat.v1.disable_eager_execution()
    im = load_img(image_path, color_mode=color_mode, target_size=SHAPE)  # -> PIL image
    pixels = np.expand_dims(img_to_array(im) / 255.0, axis=0)
    prediction = eli5.explain_prediction(model, pixels, image=im)
    return eli5.format_as_image(prediction)

@click.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.argument("image_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path())
@click.option("--rgb/--grayscale", default=False)
def explain(model_path, image_path, output_path, rgb):
    """Use MODEL_PATH to predict IMAGE_PATH with pixel activation map."""
    color_mode = "rgb" if rgb else "grayscale"
    model = load_model(model_path)
    explanation = explain_image(model, color_mode, image_path)
    plt.savefig(output_path)
    click.echo(f"Image pixel activation map saved to {output_path}")
