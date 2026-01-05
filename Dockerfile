FROM python:3.12.12 AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Create venv
RUN python -m venv .venv

# Install dependencies
COPY requirements.txt ./
RUN .venv/bin/pip install --upgrade pip
RUN .venv/bin/pip install -r requirements.txt

# Optional: pre-download model to reduce startup time
RUN .venv/bin/python -c "from huggingface_hub import hf_hub_download; \
hf_hub_download('josephpetrasek/youtube-video-evaluator','model.pth', cache_dir='/app/model_cache')"

# ---- Final image ----
FROM python:3.12.12-slim

WORKDIR /app

# Copy the model cache from the builder stage into the final image
COPY --from=builder /app/model_cache /app/model_cache

# System deps for Pillow / PyTorch
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libjpeg-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy venv
COPY --from=builder /app/.venv .venv/

# Copy app code
COPY . .

# Run using venv
CMD [".venv/bin/uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
