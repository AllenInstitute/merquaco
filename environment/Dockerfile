# hash:sha256:4b7401ec3bee6e53a57122521e5c84959555bcbdb7f51013ff9a58146f8ed344
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:24.5.0-0-python3.12.4-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1-mesa-glx=23.0.4-0ubuntu1~22.04.1 \
    && rm -rf /var/lib/apt/lists/*

RUN conda install -y \
        jupyterlab==4.3.4 \
    && conda clean -ya

RUN pip install -U --no-cache-dir \
    imagecodecs==2024.1.1 \
    matplotlib==3.9.0 \
    numpy==1.26.4 \
    opencv-python==4.9.0.80 \
    pandas==2.2.2 \
    scikit-image==0.23.2 \
    sphinx-rtd-theme==2.0.0 \
    tifffile==2024.5.10
