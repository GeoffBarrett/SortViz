FROM python:3.10.5-slim-bullseye as python-base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    # poetry 
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_PATH="/opt/python/venv"

ENV HOMEDIR=/home/geba/app
ENV PACKAGE_NAME=sortviz

# Add Poetry to PATH
ENV PATH="$POETRY_HOME/bin:$POETRY_VIRTUALENVS_PATH/bin:$PATH"

# Install needed packages and setup non-root user. Use a separate RUN statement to add your own dependencies.
ARG USERNAME=geba
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME

# builder-base is used to build dependencies
FROM python-base as builder-base

RUN apt-get update && apt-get install -y \
    # poetry
    curl \
    build-essential \
    dos2unix \
    && rm -rf /var/lib/apt/lists/*

# Bootstrap poetry, installs `POETRY_VERSION` at `POETRY_HOME`
ENV POETRY_VERSION=1.3.2 \
    POETRY_VIRTUALENVS_IN_PROJECT=false \
    POETRY_VIRTUALENVS_CREATE=true \
    POETRY_NO_INTERACTION=1

RUN curl -sSL https://install.python-poetry.org | python - -y

RUN mkdir -p ${POETRY_VIRTUALENVS_PATH} && \
    chown -R ${USER_GID}:${USER_UID} ${POETRY_VIRTUALENVS_PATH}

USER ${USERNAME}
WORKDIR ${HOMEDIR}

# Install Python
COPY --chown=$USERNAME poetry.lock pyproject.toml README.md CHANGELOG.md ./
COPY --chown=$USERNAME ${PACKAGE_NAME} ./${PACKAGE_NAME}
RUN poetry install --without dev

FROM python-base as development

RUN apt-get update && apt-get install -y \
    dos2unix \
    && rm -rf /var/lib/apt/lists/*

USER ${USERNAME}
WORKDIR ${HOMEDIR}

# Copying poetry and venv into image
COPY --from=builder-base --chown=$USERNAME ${POETRY_HOME} ${POETRY_HOME}
COPY --from=builder-base --chown=$USERNAME ${POETRY_VIRTUALENVS_PATH} ${POETRY_VIRTUALENVS_PATH}
COPY --from=builder-base --chown=$USERNAME ${HOMEDIR} ${HOMEDIR}

# venv already has runtime deps installed we get a quicker install
RUN poetry install
