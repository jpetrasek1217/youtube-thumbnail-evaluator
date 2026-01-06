# -------- Builder --------
FROM python:3.12-slim AS builder

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# System deps needed for torch + pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libjpeg-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch + deps
COPY requirements.txt .
RUN python -m pip install --upgrade pip && \
    python -m pip install \
      torch==2.9.1+cpu \
      torchvision==0.24.1+cpu \
      --index-url https://download.pytorch.org/whl/cpu && \
    python -m pip install -r requirements.txt

# -------- Runtime --------
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Runtime deps only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy only what we need
COPY --from=builder /usr/local /usr/local
COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
