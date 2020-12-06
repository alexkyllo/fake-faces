# fake-faces

Fake Face Detection Project for CSS 581 Machine Learning at UW Bothell.

## Installing

Install [Poetry](https://python-poetry.org/) and then type `poetry install` in this
directory. This will create a virtual environment and install a lot of packages.
Then type `poetry shell` to activate the environment.

OR, install [miniconda3](https://docs.conda.io/en/latest/miniconda.html) and then type
`conda env create -f environment.yaml` in this directory.
Then, once the packages are installed, type `conda activate fake-faces` to activate
the environment.

OR, use the Dockerfile to install the project in a Docker container.

First, install
[NVIDIA Docker support](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)

Then build and run the container:

``` shell
sudo docker build --tag fake-faces:latest .
sudo docker run --gpus all -v path/to/fakefaces/real_vs_fake:path/to/fakefaces/real_vs_fake -it fake-faces:latest bash
fake-faces
```
### Build Tools

Install `make`, `cmake` and `ninja`: on Ubuntu, `sudo apt install make cmake ninja-build -y`

Install a LaTeX distribution, such as `TeXLive` or `MiKTeX`, that includes `pdflatex`
to compile the PDF report.

### GPU Support

If you use the `conda` method, the GPU support should be automatic because
`tensorflow-gpu` is a dependency.

If you use the `poetry` method, you will need to install CUDA libraries on your system.

For Tensorflow GPU Support, follow the instructions at
[GPU Support|Tensorflow](https://www.tensorflow.org/install/gpu)

Additionally, as TF complained that `libcublas.so.10` was missing,
I had to `sudo apt install libcublas10` and add the following to `~/.bashrc`:

``` shell
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda-10.2/targets/x86_64-linux/lib:/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH
```

Download the fake faces dataset (4 GB zipped) from
https://www.kaggle.com/xhlulu/140k-real-and-fake-faces, decompress it on a drive
and create a `.env` file in this directory with the following content, to tell the
python package where the dataset is located:

``` shell
FAKE_FACES_DIR=path/to/fakefaces/real_vs_fake
```

## Cropping input images

We have generally seen better results on pre-cropped face images. To detect and crop
faces in a directory of images, use:

``` shell
fake-faces cropface INPUT_PATH OUTPUT_PATH
```

A pre-trained MTCNN model will be used to detect the largest face in each image, crop to it,
and output the cropped image in OUTPUT_PATH with the same filename. If no face is detected,
the file will be skipped.

## Running Experiments

An Experiment is a set of trials for a given model, with varying hyperparameters.

To define a new model, add a .py file in [fake_faces/models/](fake_faces/models/) and write a new class
that inherits from `fake_faces.models.model.Model` and implements a `build` method
(see the existing classes in [fake_faces/models/](fake_faces/models/) for examples). Then add the model
to the dictionary of models in [fake_faces/models/__init__.py](fake_faces/models/__init__.py).

To define a new experiment, add a .py file in `fake_faces/experiments/`,
create an array of one or more trials, like this example from
[fake_faces/experiments/baseline.py](fake_faces/experiments/baseline.py):

``` python
TRIALS = [
    Experiment("baseline cropped grayscale no", color_channels=1)
    .set_pipeline(
        os.path.join(DATA_DIR, "cropped/train/"),
        os.path.join(DATA_DIR, "cropped/valid/"),
    )
    .set_model(
        Baseline,
        maxpool_dropout_rate=0.2,
        dense_dropout_rate=0.5,
        optimizer=Adam(learning_rate=0.001),
    ),
]
```
Then add these trials to the experiments dictionary in
[fake_faces/experiments/__init__.py](fake_faces/experiments/__init__.py).

Run experiments from the command line with `fake-faces exp`
(if you've installed fake-faces as a package) or
`python fake_faces/cli.py exp` (as a script). You will be prompted to select
the experiment you wish to run from a menu and enter a number of epochs.

Experiment results will be logged to a folder in [experiments/](experiments/) including a CSV
file of epoch training and validation scores for plotting learning curves, saved
model .hdf5 files for resuming training and inference, and TensorBoard logs.

## Fairness Assessment

We utilize the [FairFace dataset](https://github.com/joojs/fairface) (follow links to download
from Google Drive) and the [Pixel2Style2Pixel](https://github.com/eladrich/pixel2style2pixel)
project (as a git submodule in [pixel2style2pixel/](pixel2style2pixel/).

### Aligning input images

First, we need to align and crop the input FairFace images. The `fake-faces` application
includes a command to do this, e.g.:

``` shell
fake-faces align-all /path/to/fairface/train /path/to/fairface/aligned/train/real --num_threads 4
fake-faces align-all /path/to/fairface/val /path/to/fairface/aligned/val/real --num_threads 4
```

This process uses the pretrained model `shape_predictor_68_face_landmarks.dat`
(about 99.7 MB, stored in Git LFS).

Many of the FairFace face images are not front-facing, so the model will fail to crop
them and omit them from the batch.

### Generating fake FairFace images

We use the pixel2style2pixel pretrained model `psp_ffhq_encode.pt` (about 1.2GB, stored in Git LFS)
to convert the FairFace images to fake versions for scoring.

The `fake-faces` cli includes a `falsify` command to run this process on a folder, e.g.:

``` shell
fake-faces falsify psp_ffhq_encode.pt /path/to/fairface/aligned/train/real /path/to/fairface/aligned/train/fake
fake-faces falsify psp_ffhq_encode.pt /path/to/fairface/aligned/val/real /path/to/fairface/aligned/val/fake
```

## Testing

`pytest`
