# Use nvidia/cuda image
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

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN apt-get update

RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 
RUN conda --version

WORKDIR /app

# Install dependencies:

COPY environment.yaml .
RUN conda env create -f ./environment.yaml

RUN echo "source activate bittensor" > ~/.bashrc
ENV PATH /opt/conda/envs/bittensor/bin:$PATH
RUN /bin/bash -c "source activate bittensor"

# Install dependencies:
COPY requirements.txt .
RUN /bin/bash -c "pip install -r requirements.txt"

ADD ./bittensor /app/bittensor
RUN /bin/bash -c "pip install -e /app/bittensor"
RUN /bin/bash -c "pip install hub"

COPY ./cubit /app/cubit
RUN /bin/bash -c "pip install -e /app/cubit"


