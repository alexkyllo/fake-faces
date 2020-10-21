"""plotting.py"""
import click
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mtcnn.mtcnn import MTCNN
import cv2


def detect_face_coords(path):
    """Detect a face and return its xy coordinates"""
    data = plt.imread(path)
    mtcnn = MTCNN()
    # Only interested in one face per image
    face = mtcnn.detect_faces(data)[0]
    x, y, width, height = face["box"]
    return (x, y, width, height)


def draw_box(path, x, y, width, height):
    """Detect face in image and draw a box around it."""
    data = plt.imread(path)
    plt.imshow(data)
    ax = plt.gca()
    rect = Rectangle((x, y), width, height, fill=False, color="green")
    ax.add_patch(rect)
    return ax


def crop_first_face(path):
    """Crop an image to show only the face portion."""
    img = cv2.imread(path)
    (x, y, width, height) = detect_face_coords(path)
    crop_img = img[y : y + height, x : x + width]
    return crop_img


@click.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path())
def cropface(input_path, output_path):
    """Detect and crop a face in INPUT_PATH image and save to OUTPUT_PATH."""
    fig = crop_first_face(input_path)
    cv2.imwrite(output_path, fig)
