FROM python:3.14-slim-trixie

# Set working directory
WORKDIR /app

### Install system dependencies
RUN apt-get update && apt-get install -y \
  netcat-openbsd \
  git \
  wget \
  build-essential

RUN pip install -U pip setuptools wheel

COPY dist /app/dist
COPY requirements.local.txt .
RUN pip install -r requirements.local.txt
#RUN pip install --find-links dist catch_analysis_tools


# Checkout code
# RUN --mount=type=bind,source=./,target=/app/src/
# COPY \
#   requirements.local.txt \
#   setup.py \
#   pyproject.toml \
#   .
# COPY catch_analysis_tools /app/catch_analysis_tools

# # ARG CAT_DEPLOYMENT
# # RUN if [ "$CAT_DEPLOYMENT" = "prod" ]; then \
# #   git clone git+https://github.com
# #   pip install -r src/requirements.prod.txt; \
# # else \
# # fi

# # Install dependencies
# #RUN pip install -r requirements.local.txt;

# # Install code
# RUN pip install 
