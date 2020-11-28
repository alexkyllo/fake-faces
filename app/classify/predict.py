import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.preprocessing.image as image
import logging
from tensorflow.keras.models import load_model

import h5py             # h5py for loading the hdf5 file holding our model
import os               # os in order to grab the correct path for the model
import urllib           # Used to convert from URL to an image file
from io import BytesIO  # Used to convert from URL to an image file
from PIL import Image   # Used to convert from URL to an image file

# Code from: https://stackoverflow.com/questions/55821612/whats-the-fastest-way-to-read-images-from-urls
def loadImage(URL):
    with urllib.request.urlopen(URL) as url:
        res = url.read()
    img = Image.open(BytesIO(res)).resize((128,128))
    return image.img_to_array(img)

def predict_image_from_url(image_url):
    logging.info('predict.py: Image URL received: ' + image_url)
    ### Conversions
    # Convert from url to np.array
    img = loadImage(image_url)

    # Convert from shape (128, 128, 3) to (1, 128, 128, 3)
    img = tf.expand_dims(img, axis=0)


    ### Load the model file
    # Code from: https://github.com/microsoft/vscode-azurefunctions/issues/1439
    function_dir = os.path.dirname(os.path.realpath(__file__))
    model_path = os.path.join(function_dir, 'best-model.hdf5')

    # Idea for h5py from: https://www.machinecurve.com/index.php/2020/04/13/how-to-use-h5py-and-keras-to-train-with-data-from-hdf5-files/
    f = h5py.File(model_path,'r')
    model = load_model(f)
    f.close()

    ### Send the image through the model
    gen = image.ImageDataGenerator({"rescale": 1.0 / 255})
    test = gen.flow(img, batch_size=1)
    prediction = model.predict(test)

    return prediction