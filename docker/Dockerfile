FROM python:3.14-slim-trixie

# Set working directory
WORKDIR /app

### Install system dependencies
RUN apt-get update && apt-get install -y \
  netcat-openbsd \
  git \
  wget \
  build-essential

# Copy python code, but this is only used for running locally
COPY \
  setup.py \
  pyproject.toml \
  requirements.prod.txt \
  requirements.local.txt \
  catch_analysis_tools \
  .

# Install dependencies
ARG CAT_DEPLOYMENT
RUN pip install -q -U pip setuptools wheel
RUN if [ "$CAT_DEPLOYMENT" = "prod" ]; then \
  pip install -r requirements.prod.txt; \
else \
  pip install .; \
  pip install -r requirements.local.txt; \
fi

