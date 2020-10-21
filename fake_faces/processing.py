"""processing.py"""

import logging
import click
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mtcnn.mtcnn import MTCNN
import cv2


def detect_face_coords(path):
    """Detect a face and return its xy coordinates"""
    data = plt.imread(path)
    mtcnn = MTCNN()
    faces = mtcnn.detect_faces(data)
    if len(faces) < 1:
        return (0, 0, 0, 0)
    # Only interested in one face per image
    x, y, width, height = face[0]["box"]
    return (x, y, width, height)


def crop_first_face(path):
    """Crop an image to show only the face portion."""
    img = cv2.imread(path)
    (x, y, width, height) = detect_face_coords(path)
    # handle case of no face detected
    if x == (0, 0, 0, 0):
        return None
    crop_img = img[y : y + height, x : x + width]
    return crop_img


@click.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path())
def cropface(input_path, output_path):
    """Detect and crop a face in INPUT_PATH image and save to OUTPUT_PATH."""
    logger = logging.getLogger(__name__)
    fig = crop_first_face(input_path)
    if fig:
        cv2.imwrite(output_path, fig)
    else:
        logger.warn("No human face detected in %s. No output written.", input_path)
