#FROM ubuntu:20.04
#FROM nvidia/cuda:11.1.1-cudnn8-runtime-ubuntu20.04
#FROM nvidia/cuda:11.7.0-cudnn8-runtime-ubuntu20.04

FROM nvidia/cuda:10.2-runtime-ubuntu18.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt update \
    && apt install -y htop python3.8 wget git ffmpeg libsm6 libxext6

RUN apt install -y python3-setuptools python3-pip python3-venv

RUN python3 -m pip install --upgrade pip
RUN pip3 install virtualenv
RUN virtualenv -p python3 ml

COPY . src/
RUN mkdir -p /data/exvivo

RUN /bin/bash -c "cd src \
    && source /ml/bin/activate \
    && pip3 install -r requirements.txt"

