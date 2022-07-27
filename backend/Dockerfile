# Use nvidia/cuda image
FROM python:3.8 

ENV PYTHONUNBUFFERED 1
ENV PYTHONDONTWRITEBYTECODE 1
ENV TZ=Europe/London
ARG DEBIAN_FRONTEND=noninteractive

# #nvidia key migration
# RUN apt-key del 7fa2af80
# RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
# RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64/7fa2af80.pub

# RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests -y apt-utils curl git cmake build-essential unzip python3-pip  wget iproute2 software-properties-common
# RUN add-apt-repository ppa:deadsnakes/ppa
# RUN apt-get install python3 python3-dev -y
# RUN python3 -m pip install --upgrade pip

# # Install linux dependencies
# RUN apt-get install -y libssl-dev
# RUN apt-get install -y npm
# RUN apt-get install -y gcc

WORKDIR /app

# RUNNING THESE IN THE EXECUTE COMMAND 

# add BITTENSOR commune fork

# TEMPORARY UNTIL MERGE WITH FSSPEC add ipfsspec

# RUN pip install bittensor

# ADD ./brownie/ /app/brownie/
# RUN rm -r ./ipfsspec
# RUN pip install bittensor
# ADD ./brownie/ /app/brownie/
# RUN pip install -e ./brownie
## -- Enter Virtual Env Installs our requirements into the Docker image
COPY ./requirements.txt .
RUN pip install -r requirements.txt

ADD ./bittensor/ /app/bittensor/
RUN python -m pip install -e bittensor/

# RUN pip install eth-brownie






COPY ./scripts scripts
RUN chmod +x ./scripts/*

# set working directory



# RUN npm install -g ganache-cli
# RUN pip install bittensor
