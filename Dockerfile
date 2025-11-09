# ACCT445-Showcase Production Container

FROM python:3.11-slim

# Metadata
LABEL maintainer="Nirvan Chitnis"
LABEL description="ACCT445 Bank Disclosure Opacity Trading System"
LABEL version="1.0"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
COPY pyproject.toml .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir streamlit plotly sphinx

# Copy application code
COPY src/ ./src/
COPY config/ ./config/
COPY notebooks/ ./notebooks/
COPY results/ ./results/
COPY data/ ./data/

# Install package
RUN pip install -e .

# Create directories
RUN mkdir -p data/cache results logs

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV LOG_LEVEL=INFO

# Expose ports
EXPOSE 8501
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import src; print('OK')" || exit 1

# Default command: Run dashboard
CMD ["streamlit", "run", "src/dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
