FROM tensorflow/tensorflow:latest-gpu-jupyter
WORKDIR /fake-faces/
COPY setup.py /fake-faces/setup.py
COPY .env /fake-faces/.env
COPY fake_faces/ /fake-faces/fake_faces/
RUN apt-get update
RUN apt-get install libgl1-mesa-glx ffmpeg libsm6 libxext6 -y
RUN apt-get install build-essential cmake pkg-config
RUN apt-get install libx11-dev libatlas-base-dev
RUN apt-get install libgtk-3-dev libboost-python-dev
RUN apt-get install python-dev python3-dev
RUN pip install -e .
