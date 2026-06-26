FROM python:3.11-slim-bookworm

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
  astrometry.net \
  source-extractor \
  netcat-openbsd \
  git \
  wget \
  build-essential \
  zlib1g-dev \
  libbz2-dev \
  && rm -rf /var/lib/apt/lists/*

RUN --mount=type=cache,target=/root/.cache/pip \
  pip install -U pip setuptools wheel

COPY requirements.local.txt /app/

RUN --mount=type=cache,target=/root/.cache/pip \
  pip install -r requirements.local.txt

COPY pyproject.toml setup.py MANIFEST.in README.md /app/
COPY catch_analysis_tools /app/catch_analysis_tools
COPY scripts /app/scripts
COPY docker /app/docker

ARG PACKAGE_VERSION=0.0.0
ENV SETUPTOOLS_SCM_PRETEND_VERSION_FOR_CATCH_ANALYSIS_TOOLS=${PACKAGE_VERSION}

RUN --mount=type=cache,target=/root/.cache/pip \
  pip install --no-deps /app

RUN chmod +x /app/docker/entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["/app/docker/entrypoint.sh"]
