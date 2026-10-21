FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.12 python3.12-dev python3.12-venv python3-pip \
    git ffmpeg libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/*

RUN python3.12 -m ensurepip --upgrade && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    python3.12 -m pip install --upgrade pip

RUN python3.12 -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu126

WORKDIR /opt/lerobot
COPY vendor/lerobot/ .
RUN python3.12 -m pip install --no-cache-dir -e ".[groot]"

# SageMaker L40S (sm_89) NVRTC workaround: integer reductions have no precompiled
# kernel, triggering NVRTC JIT which fails due to missing libnvrtc-builtins.so.13.0.
# This module monkey-patches ALL integer reduction ops to use CPU instead.
COPY docker/groot-sagemaker/nvrtc_workaround.py /opt/nvrtc_workaround.py
RUN SITE=$(python3.12 -c "import site; print(site.getsitepackages()[0])") && \
    cp /opt/nvrtc_workaround.py "$SITE/nvrtc_workaround.py" && \
    echo "import nvrtc_workaround" > "$SITE/nvrtc_workaround.pth"

COPY docker/groot-sagemaker/train.sh /opt/train.sh
RUN chmod +x /opt/train.sh

ENTRYPOINT ["/opt/train.sh"]
