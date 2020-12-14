"""predicting.py
Make a prediction (inference) on a single image file."""
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
from tensorflow.keras.models import load_model
from fake_faces.training import SHAPE
import click

def predict_image(model, image_path, color_mode):
    """Get the prediction for a single image."""
    im = load_img(image_path, color_mode=color_mode, target_size=SHAPE)
    pixels = np.expand_dims(img_to_array(im), axis=0) / 255.0
    prediction = model.predict(pixels)
    return prediction.flatten()[0]

@click.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.argument("image_path", type=click.Path(exists=True))
@click.option("--rgb/--grayscale", default=False)
def predict(model_path, image_path, rgb):
    """Use MODEL_PATH to predict on IMAGE_PATH."""
    color_mode = "rgb" if rgb else "grayscale"
    model = load_model(model_path)
    prediction = np.round(predict_image(model, image_path, color_mode), 2)
    if prediction >= 0.5:
        result = "REAL"
        conf = prediction * 100
    else:
        result = "FAKE"
        conf = (1 - prediction) * 100
    click.echo(f"The model predicts this image is {result} with {conf}% confidence")
