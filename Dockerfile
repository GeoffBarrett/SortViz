FROM python:3.10.5-slim-bullseye@sha256:ca78039cbd3772addb9179953bbf8fe71b50d4824b192e901d312720f5902b22

ENV POETRY_HOME="/opt/poetry"
ENV POETRY_VERSION 1.1.13
ENV HOMEDIR /home/geba/app
ENV PATH="$POETRY_HOME/bin:$PATH"

# Install needed packages and setup non-root user. Use a separate RUN statement to add your own dependencies.
ARG USERNAME=geba
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME

RUN apt-get update && apt-get install -y \
    vim \
    dos2unix \
    && rm -rf /var/lib/apt/lists/*

RUN set -ex; pip install --no-cache-dir poetry==$POETRY_VERSION;

USER ${USERNAME}
WORKDIR ${HOMEDIR}

COPY --chown=$USERNAME poetry.lock pyproject.toml ./
RUN PIP_NO_CACHE_DIR=true POETRY_VIRTUALENVS_CREATE=false poetry install
