import os
import shutil
from fake_faces import __version__
from fake_faces import processing

def test_version():
    assert __version__ == '0.1.0'

def test_cropface_dir():
    input_path = "tests/input"
    output_path = "tests/output"
    processing.crop_faces(input_path, output_path)
    result = [f for f in os.listdir(output_path) if os.path.isfile(os.path.join(output_path, f))]
    assert sorted(os.listdir(input_path)) == sorted(result)
    shutil.rmtree(output_path)

def test_cropface_file():
    input_path = "tests/input/00A0WLZE5X.jpg"
    output_path = "tests/output/single/00A0WLZE5X.jpg"
    processing.crop_faces(input_path, output_path)
    assert os.path.isfile(output_path)
    shutil.rmtree(os.path.dirname(output_path))
