# syntax=docker/dockerfile:1
FROM opentensorfdn/bittensor:3.3.3-torch1.12.0-cuda11.3-pm2
# Install cubit

# Install cubit
COPY ./cubit /cubit

RUN cd /cubit && python3 -m pip install -r requirements.dev.txt
RUN cd /cubit && python3 setup.py build_ext --inplace

ENV PYTHONPATH "${PYTHONPATH}:/cubit"

WORKDIR app
COPY ./bittensor /app/bittensor
RUN python3 -m pip install -e /app/bittensor

RUN python3 -m pip install -U "ray[default]"
RUN python3 -m pip install -U streamlit
RUN python3 -m pip install -U gradio
RUN python3 -m pip install -U plotly
RUN python3 -m pip install -U sentence-transformers
RUN python3 -m pip install -U datasets

COPY ./scripts .
