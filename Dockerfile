# Multi-stage Dockerfile for ACCT445-Showcase
# Target: <700 MB (down from 2.07 GB)

# ============================================================
# Stage 1: Builder (install dependencies, compile wheels)
# ============================================================
FROM python:3.11-slim AS builder

# Metadata
LABEL maintainer="Nirvan Chitnis"
LABEL description="ACCT445 Bank Disclosure Opacity Trading System"
LABEL version="1.0"

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy dependency files
COPY requirements.txt pyproject.toml ./

# Install Python dependencies in a virtual environment
RUN pip install --upgrade pip setuptools wheel && \
    python -m venv /build/.venv && \
    /build/.venv/bin/pip install --no-cache-dir -r requirements.txt

# Install package (editable mode setup files)
COPY src/ ./src/
RUN /build/.venv/bin/pip install --no-cache-dir -e .

# ============================================================
# Stage 2: Runtime (minimal final image)
# ============================================================
FROM python:3.11-slim

# Install runtime dependencies only (git for DVC)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app && \
    chown -R appuser:appuser /app

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder --chown=appuser:appuser /build/.venv /app/.venv

# Copy application code
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser config/ ./config/
COPY --chown=appuser:appuser notebooks/ ./notebooks/
COPY --chown=appuser:appuser pyproject.toml ./

# Create data directories
RUN mkdir -p data/cache data/factors data/prices results logs && \
    chown -R appuser:appuser data results logs

# Switch to non-root user
USER appuser

# Add virtual environment to PATH
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Expose ports
EXPOSE 8501  # Streamlit dashboard
EXPOSE 8000  # Optional API (future)

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=5 \
    CMD python -c "import sys; sys.exit(0)" || exit 1

# Default command (dashboard)
CMD ["streamlit", "run", "src/dashboard/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
