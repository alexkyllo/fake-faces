"""processing.py"""
import os
import logging
import click
import matplotlib.pyplot as plt
from mtcnn.mtcnn import MTCNN
import cv2

mtcnn = MTCNN()


def detect_face_coords(path):
    """Detect a face and return its xy coordinates"""
    data = plt.imread(path)
    faces = mtcnn.detect_faces(data)
    if len(faces) < 1:
        return (0, 0, 0, 0)
    # Only interested in one face per image
    # Find the face with the largest area
    face_areas = [face["box"][2] * face["box"][3] for face in faces]
    biggest_face = faces[face_areas.index(max(face_areas))]
    x, y, width, height = biggest_face["box"]
    # x and y can fall outside the image border.
    # If so, set them to 0
    if x < 0:
        width += x
        x = 0
    if y < 0:
        height += y
        y = 0
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


def list_excess_files(input_path, output_path):
    """List all files in a dir (non-recursively)"""
    files = [
        os.path.join(input_path, f)
        for f in os.listdir(input_path)
        if os.path.isfile(os.path.join(input_path, f))
        and f not in os.listdir(output_path)
    ]
    return files


def crop_faces(input_path, output_path):
    """Detect and crop the first face in an image or dir of images and save to output_path."""
    logger = logging.getLogger(__name__)
    output_files = []
    if os.path.isdir(input_path):
        if os.path.isfile(output_path):
            raise ValueError(
                """If input_path is a directory,
 output_path must also be a directory."""
            )
        else:
            os.makedirs(output_path, exist_ok=True)
        # Get all files in input_path but not in output_path
        files = list_excess_files(input_path, output_path)
        num_files = len(files)
        logger.info("Found %s files to process in %s", num_files, input_path)
    else:
        files = [input_path]
        num_files = len(files)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    for i, f in enumerate(files):
        logger.info(
            "Processing image %s of %s, progress %.2f%%",
            i + 1,
            num_files,
            (i + 1) / num_files * 100,
        )
        fig = crop_first_face(f)
        if fig is not None:
            if os.path.isdir(output_path):
                output_file = os.path.join(output_path, os.path.basename(f))
            else:
                output_file = output_path
            logger.info("Writing cropped face to %s", output_file)
            cv2.imwrite(output_file, fig)
            output_files.append(output_file)
        else:
            logger.warning(
                "No human face detected in %s. No output written.", input_path
            )
    return output_files


@click.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path())
def cropface(input_path, output_path):
    """Detect and crop a face in INPUT_PATH image(s) and save to OUTPUT_PATH."""
    if os.path.isdir(input_path):
        if os.path.isfile(output_path):
            raise click.BadParameter(
                """If input_path is a directory,
 output_path must also be a directory."""
            )
    crop_faces(input_path, output_path)
