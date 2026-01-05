FROM python:3.12.12 AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/model_cache

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --upgrade pip
RUN python -m pip install -r requirements.txt

# Create model cache and pre-download
RUN mkdir -p /app/model_cache
RUN python -c "from huggingface_hub import hf_hub_download; \
    hf_hub_download(repo_id='josephpetrasek/youtube-video-evaluator', filename='model.pth')"

# ---- Final image ----
FROM python:3.12.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/model_cache \
    PATH="/install/bin:$PATH"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libjpeg-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages
COPY --from=builder /install /install

# Copy pre-downloaded model
COPY --from=builder /app/model_cache /app/model_cache

# Copy app code
COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
