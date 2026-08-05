# Multi-stage Dockerfile for EquiMind Financial Research Orchestration Engine
FROM python:3.12-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt uvicorn fastapi gunicorn

# Production Image Stage
FROM python:3.12-slim

WORKDIR /app

# Copy installed dependencies from builder
COPY --from=builder /install /usr/local

# Copy application source code
COPY equimind/ ./equimind/
COPY web/ ./web/
COPY pyproject.toml README.md ./

# Create non-root system user for security
RUN useradd -m -u 1000 equimind && chown -R equimind:equimind /app
USER equimind

EXPOSE 8000

ENV PYTHONUNBUFFERED=1
ENV PORT=8000

CMD ["python", "-m", "uvicorn", "equimind.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
