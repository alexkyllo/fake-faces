"""predict.py
Perform inference on a single image from a URL."""
import os  # os in order to grab the correct path for the model
# import sys
# import math
import logging
import urllib  # Used to convert from URL to an image file
from io import BytesIO, StringIO  # Used to convert from URL to an image file
import tensorflow as tf
import tensorflow.keras.preprocessing.image as image
from tensorflow.keras.models import load_model
from PIL import Image  # Used to convert from URL to an image file

# TODO: Refactor with a conditional based on type of input.

# Code from: https://stackoverflow.com/questions/55821612/whats-the-fastest-way-to-read-images-from-urls
def load_image(url):
    """Open image bytes from a URL and return as a numpy array."""
    # if url == "https://thispersondoesnotexist.com/image": url += ".jpg"
    logging.info("predict.py: Image URL received now: %s", url)

    with urllib.request.urlopen(url) as img_url:
        res = img_url.read()
    img = Image.open(BytesIO(res)).resize((128, 128))
    img = img.convert("L")  # convert the image to grayscale.
    return image.img_to_array(img)

def load_image_b(img_file):
    logging.info("predict.py: Image file: %s", img_file)
    # img = Image.open(img_file)
    # imgFile = BytesIO(img_file)
    
    img = Image.open(BytesIO(img_file))
    # img = Image.open(StringIO(img_file)).resize((128, 128))
    # TODO: Put size of image in as second param for frombytes, or solve "not enough image data" problem with BytesIO
    
    # img = Image.frombytes('RGBA', (180, 180), img_file, 'raw')
    img = img.resize((128,128))
    img = img.convert("L")
    return image.img_to_array(img)

def predict_image_from_url(image_url):
    """Read an image from a URL, """
    logging.info("predict.py: Image URL received: %s", image_url)
    ### Conversions
    # Convert from url to np.array
    img = load_image(image_url)

    # Convert from shape (128, 128, 3) to (1, 128, 128, 3)
    img = tf.expand_dims(img, axis=0)
    logging.info(img.shape.as_list())

    ### Load the model file
    # Code from: https://github.com/microsoft/vscode-azurefunctions/issues/1439
    function_dir = os.path.dirname(os.path.realpath(__file__))
    model_path = os.path.join(function_dir, "best-model.hdf5")
    model = load_model(model_path)

    ### Send the image through the model
    # gen = image.ImageDataGenerator({"rescale": 1.0 / 255})
    # test = gen.flow(img, batch_size=1)
    prediction = model.predict(img)

    return prediction

def predict_image_from_file(image_file):
    """Read an image from a bytestream, """
    ### Conversions
    # Convert from bytes to np.array
    img = load_image_b(image_file)

    # Convert from shape (128, 128, 3) to (1, 128, 128, 3)
    img = tf.expand_dims(img, axis=0)
    print(img.shape.as_list())
    ### Load the model file
    # Code from: https://github.com/microsoft/vscode-azurefunctions/issues/1439
    function_dir = os.path.dirname(os.path.realpath(__file__))
    model_path = os.path.join(function_dir, "best-model.hdf5")
    model = load_model(model_path)

    ### Send the image through the model
    # gen = image.ImageDataGenerator({"rescale": 1.0 / 255})
    # test = gen.flow(img, batch_size=1)
    prediction = model.predict(img)

    return prediction
