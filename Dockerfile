FROM ubuntu

ARG DEBIAN_FRONTEND="noninteractive"
RUN apt update
RUN apt upgrade -y
RUN apt install -y \
    software-properties-common

RUN add-apt-repository ppa:deadsnakes/ppa && apt update

RUN apt install -y \
    python3.10 \
    python3-pip \
    python3-venv

WORKDIR /autoencoder-ids

COPY requirements.txt /autoencoder-ids/requirements.txt
RUN python3.10 -m pip install -r requirements.txt

RUN echo 'alias autoencoder-ids="python3.10 /autoencoder-ids/src/main.py"' >> /root/.bashrc

COPY src /autoencoder-ids/src
