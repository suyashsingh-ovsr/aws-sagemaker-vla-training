FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      software-properties-common git ffmpeg libsm6 libxext6 && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
      python3.12 python3.12-dev python3.12-venv python3-pip && \
    rm -rf /var/lib/apt/lists/*

RUN python3.12 -m ensurepip --upgrade && \
    python3.12 -m pip install --upgrade pip

WORKDIR /opt/lerobot

COPY vendor/lerobot/ .

RUN python3.12 -m pip install -e ".[smolvla,dataset]"

COPY docker/smolvla/train.sh /opt/train.sh
RUN chmod +x /opt/train.sh

ENTRYPOINT ["/opt/train.sh"]
