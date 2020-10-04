FROM python:3.7-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        gnupg2 curl ca-certificates gcc libc-dev unzip libgtk2.0-dev libgl1-mesa-dev \
    && curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub | apt-key add - \
    && echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda.list \ 
    && echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list 


ENV CUDA_VERSION 10.2.89
ENV CUDA_PKG_VERSION 10-2=$CUDA_VERSION-1
ENV NVIDIA_VISIBLE_DEVICES all
ENV MYPYPATH /srv/stubs
ENV PATH /usr/local/cuda/bin:/usr/local/nvidia/bin:/root/.poetry/bin:${PATH}

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        cuda-cudart-$CUDA_PKG_VERSION \
        cuda-compat-10-2 \
    && ln -s cuda-10.2 /usr/local/cuda \
    && rm -rf /var/lib/apt/lists/* \
    && curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python \
    && poetry config virtualenvs.create false

WORKDIR /srv
COPY . .
RUN poetry install
