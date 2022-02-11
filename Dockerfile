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
        zlib1g-dev \ 
        curl \ 
        ca-certificates \ 
        unzip \ 
        graphviz \ 
        git \
    && curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub | apt-key add - \
    && echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64 /" > /etc/apt/sources.list.d/cuda.list \
    && echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        cuda-cudart-11-3=11.3.109-1 \
        cuda-compat-11-3 \
    && ln -s cuda-11.3 /usr/local/cuda \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html

WORKDIR /app
COPY . .
RUN pip install -e vision_tools[dev] \
    && pip install -e kuzushiji_bench \
    && pip install -e cots_bench
