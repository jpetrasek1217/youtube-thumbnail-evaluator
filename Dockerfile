FROM python:3.12.12 AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/model_cache

WORKDIR /app

# Create venv
RUN python -m venv .venv

# Install dependencies
COPY requirements.txt ./
RUN .venv/bin/pip install --upgrade pip
RUN .venv/bin/pip install -r requirements.txt

# Pre-download model - now it will save to /app/model_cache automatically
RUN .venv/bin/python -c "from huggingface_hub import hf_hub_download; \
    hf_hub_download(repo_id='josephpetrasek/youtube-video-evaluator', filename='model.pth')"

# ---- Final image ----
FROM python:3.12.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/model_cache

WORKDIR /app

# System deps for Pillow / PyTorch
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libjpeg-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the model cache from the builder stage
COPY --from=builder /app/model_cache /app/model_cache

# Copy venv
COPY --from=builder /app/.venv .venv/

# Copy app code
COPY . .

CMD [".venv/bin/uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]