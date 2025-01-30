# hash:sha256:0f35ebf4ef55d33814299a4996fb6faa2393d971b9a945272598b97e7b79d267
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:24.5.0-0-python3.12.4-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN conda install -y \
        jupyterlab==4.3.4 \
    && conda clean -ya

RUN pip install -U --no-cache-dir \
    imagecodecs==2024.12.30 \
    matplotlib==3.10.0 \
    numpy==1.26.4 \
    opencv-python-headless==4.11.0.86 \
    pandas==2.2.3 \
    scikit-image==0.25.0 \
    sphinx-rtd-theme==3.0.2 \
    tifffile==2025.1.10
