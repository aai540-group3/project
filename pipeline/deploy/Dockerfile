FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY pipeline/ pipeline/
COPY conf/ conf/
COPY models/ models/

# Set environment variables
ENV PYTHONPATH=/app
ENV MODEL_PATH=/app/models/model.pkl

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["uvicorn", "pipeline.deploy.app:app", "--host", "0.0.0.0", "--port", "8000"]