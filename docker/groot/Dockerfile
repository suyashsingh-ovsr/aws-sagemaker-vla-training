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
COPY docker/groot/pyproject.toml .
RUN uv sync --no-install-project --python 3.12

ENV PATH="/opt/training/.venv/bin:$PATH"

# NVRTC workaround for SageMaker L40S (sm_89): patches integer reduction ops to CPU.
COPY docker/groot/nvrtc_workaround.py /opt/nvrtc_workaround.py
RUN SITE=$(python -c "import site; print(site.getsitepackages()[0])") && \
    cp /opt/nvrtc_workaround.py "$SITE/nvrtc_workaround.py" && \
    echo "import nvrtc_workaround" > "$SITE/nvrtc_workaround.pth"

COPY docker/groot/train.sh /opt/train.sh
RUN chmod +x /opt/train.sh

ENTRYPOINT ["/opt/train.sh"]
