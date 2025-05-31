FROM python:3.11-slim

# Set pip configuration
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# First upgrade pip
RUN python -m pip install --upgrade pip

# Copy requirements first for caching
COPY requirements.txt .

# Install packages without hash checking (remove --no-deps for production)
RUN pip install -r requirements.txt

# Copy application
COPY . .

# Create directories
RUN mkdir -p /app/data/processed \
    && mkdir -p /app/data/features \
    && mkdir -p /app/models \
    && mkdir -p /app/static \
    && mkdir -p /app/mlruns

# Set environment variables
ENV MLFLOW_TRACKING_URI=sqlite:////app/mlflow.db \
    MLFLOW_REGISTRY_URI=sqlite:////app/mlflow.db

EXPOSE 8000
CMD ["uvicorn", "src.app.api:app", "--host", "0.0.0.0", "--port", "8000"]
