"""plotting.py"""
import click
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mtcnn.mtcnn import MTCNN
import cv2

def draw_box(path, x, y, width, height, color="green"):
    """Draw a rectangle around given immage coordinates."""
    data = plt.imread(path)
    plt.imshow(data)
    ax = plt.gca()
    rect = Rectangle((x, y), width, height, fill=False, color=color)
    ax.add_patch(rect)
    return ax
