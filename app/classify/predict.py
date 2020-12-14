"""predict.py
Perform inference on a single image from a URL."""
import os  # os in order to grab the correct path for the model
import logging
import urllib  # Used to convert from URL to an image file
from io import BytesIO  # Used to convert from URL to an image file
import tensorflow as tf
import tensorflow.keras.preprocessing.image as image
from tensorflow.keras.models import load_model
from PIL import Image  # Used to convert from URL to an image file


def load_image(img_src):
    """Open image bytes from a URL/file and return as a numpy array."""
    logging.info("predict.py: Inside load_image")

    isStr = isinstance(img_src, str)
    res = img_src

    if isStr:
        with urllib.request.urlopen(img_src) as img_url:
            res = img_url.read()
    
    img = Image.open(BytesIO(res)).resize((128, 128))

    return image.img_to_array(img)

def predict_image(image_file):
    """Read an image from a bytestream, """
    isStr = isinstance(image_file, str)
    if isStr:
        logging.info("predict.py: Transferred a URL: %s", isStr)
    else:
        logging.info("predict.py: Transferred a file")
    
    ### Conversions
    # Convert from bytes to np.array
    img = load_image(image_file)

    # Rescaling pixels
    img = img / 255.0

    # Convert from shape (128, 128, 3) to (1, 128, 128, 3)
    img = tf.expand_dims(img, axis=0)
    print(img.shape.as_list())

    ### Load the model file
    function_dir = os.path.dirname(os.path.realpath(__file__))
    model_path = os.path.join(function_dir, "best-model.hdf5")
    model = load_model(model_path)

    ### Send the image through the model
    prediction = model.predict(img)

    return prediction