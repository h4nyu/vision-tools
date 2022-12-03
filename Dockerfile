FROM python:3.7.12-buster


ENV NVIDIA_VISIBLE_DEVICES=all \
    PATH=/usr/local/cuda/bin:/usr/local/nvidia/bin:/root/.local/bin:${PATH} \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    NVIDIA_REQUIRE_CUDA="cuda>=11.3" \
    CUDA_VERSION=11.3.0

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        gnupg2 \
        libc-dev \
        libjpeg-dev \
        libpng-dev \
        zlib1g-dev \
        curl \
        ca-certificates \
        cmake \
        unzip \
        graphviz \
        graphviz-dev \
        git \
        zip \
        unzip \
    && wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb \
    && dpkg -i cuda-keyring_1.0-1_all.deb \
    && rm cuda-keyring_1.0-1_all.deb \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        cuda-cudart-11-3=11.3.109-1 \
        cuda-compat-11-3 \
    && ln -s cuda-11.3 /usr/local/cuda \
    && rm -rf /var/lib/apt/lists/* \
    && pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

WORKDIR /app
COPY . .
RUN pip install -e vision_tools[dev] \
    && pip install -e coco_annotator[dev] \
    && pip install -e kuzushiji_bench \
    && pip install -e cots_bench \
    && pip install -e hwad_bench \
    && pip install -e tanacho_bench \
    && pip install -e rbcd \
    &&  CC="cc -mavx2" pip install -U --force-reinstall pillow-simd
