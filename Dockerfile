FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel
LABEL Service="SparseInstanceActivation"

ENV TZ=Europe/Moscow
ENV DETECTRON_TAG=v0.3
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-key del 7fa2af80 && \
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub && \
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub
RUN apt update && apt install vim git g++ python3-tk ffmpeg libsm6 libxext6 -y

RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir opencv-python opencv-contrib-python scipy

WORKDIR /workspace
RUN git clone https://github.com/facebookresearch/detectron2.git && \
    cd detectron2/ && git checkout tags/${DETECTRON_TAG} && python3 setup.py build develop

RUN python3 -m pip uninstall -y iopath fvcore portalocker yacs && \
    python3 -m pip install --no-cache-dir iopath fvcore portalocker yacs timm pyyaml==5.1 shapely

RUN git clone https://github.com/hustvl/SparseInst
WORKDIR /workspace/SparseInst
RUN ln -s /usr/bin/python3 /usr/bin/python

ENTRYPOINT bash
