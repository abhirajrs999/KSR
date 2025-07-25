FROM python:3.11-slim as builder

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy and install requirements step by step
COPY requirements.txt .

# Install packages with better error handling
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --timeout 1000 -r requirements.txt

# Production stage
FROM python:3.11-slim

# Install only runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Create data directories with proper permissions
RUN mkdir -p /data/raw_pdfs /data/processed/parsed_docs /data/processed/chunks \
             /data/processed/metadata /data/vector_db && \
    chmod -R 755 /data

# Copy minimal application files for initial deployment
COPY main-minimal.py ./main.py

# Temporarily disable health check for debugging
# HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
#   CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

# Expose port 8000 since Railway domain is configured for it
EXPOSE 8000

# Use Railway's PORT env var if available, otherwise default to 8000
CMD python -m uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
