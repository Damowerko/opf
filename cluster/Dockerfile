FROM python:3.11-slim
ARG USERNAME=default
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --gid $USER_GID $USERNAME && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME -s /bin/bash

USER $USERNAME
ENV PATH="/home/$USERNAME/.local/bin:${PATH}"
WORKDIR /home/$USERNAME/opf

ENV VIRTUAL_ENV=/home/$USERNAME/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# install torch to avoid rebuilding the image every time
RUN pip install torch==2.0.0

# install poetry
RUN curl -sSL https://install.python-poetry.org | python3 - \
    && poetry config virtualenvs.create false \
    && poetry config installer.max-workers 10

# install requirements
COPY poetry.lock pyproject.toml README.md ./
RUN poetry install --no-root

# Copy whole repo, .dockerignore should exclude everything unnecessary
RUN git config --global --add safe.directory /home/$USERNAME/opf
COPY . .
# install opf package in editable mode
RUN poetry install --only-root