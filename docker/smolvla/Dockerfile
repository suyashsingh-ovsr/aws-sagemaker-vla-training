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
      python3.12 python3.12-dev && \
    rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /opt/training
COPY docker/smolvla/pyproject.toml .
RUN uv sync --no-install-project --python 3.12

ENV PATH="/opt/training/.venv/bin:$PATH"

COPY docker/smolvla/train.sh /opt/train.sh
RUN chmod +x /opt/train.sh

ENTRYPOINT ["/opt/train.sh"]
