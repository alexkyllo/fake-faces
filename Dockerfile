FROM tensorflow/tensorflow:latest-gpu-jupyter
WORKDIR /fake-faces/
COPY setup.py /fake-faces/setup.py
COPY .env /fake-faces/.env
COPY fake_faces/ /fake-faces/fake_faces/
RUN apt-get update
RUN apt-get install libgl1-mesa-glx ffmpeg libsm6 libxext6 -y
RUN pip install -e .
