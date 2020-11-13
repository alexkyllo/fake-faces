"""explaining.py
Visually explain model predictions
"""
import eli5
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from fake_faces.training import SHAPE


def explain(model, color_mode, image_path):
    """Plot on an image a visualization of the areas used to classify it."""
    # TODO: This has not been successfully tested yet.
    im = load_img(image_path, color_mode=color_mode, target_size=SHAPE) # -> PIL image
    pixels = np.expand_dims(img_to_array(im), axis=0)
    return eli5.format_as_image(eli5.explain_prediction(model, pixels))
