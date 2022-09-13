FROM nvidia/cuda:11.2.1-base

ARG DEBIAN_FRONTEND=noninteractive
#nvidia key migration
RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64/7fa2af80.pub

RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests -y apt-utils curl git cmake build-essential unzip python3-pip  wget iproute2 software-properties-common

RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update
RUN apt-get install python3 python3-dev -y
RUN python3 -m pip install --upgrade pip

# add Bittensor code to docker image

WORKDIR /app
RUN pip install --upgrade numpy pandas setuptools "tqdm>=4.27,<4.50.0" wheel
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY ./bittensor /app/bittensor
RUN pip install -e ./bittensor

EXPOSE 8091
