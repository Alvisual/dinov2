# syntax=docker/dockerfile:1
# https://docs.docker.com/engine/reference/builder/
# https://docs.docker.com/get-started/09_image_best/
# https://docs.docker.com/develop/dev-best-practices/
# https://docs.docker.com/develop/develop-images/dockerfile_best-practices/

# https://docs.docker.com/language/python/build-images/
FROM python:3.9.19-slim-bullseye

ARG USERNAME=USERNAME
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
RUN pip install --timeout=600 --no-cache-dir -r requirements.txt