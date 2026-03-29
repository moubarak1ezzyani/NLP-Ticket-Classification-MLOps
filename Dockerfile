# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies (build-essential needed for some ML libraries)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first (for caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download required SpaCy model
RUN python -m spacy download en_core_web_sm

# Copy the rest of the application code
COPY src/ src/
COPY config/ config/
COPY data/ data/

# Set environment variable for Python path (same fix as main.py but in container)
ENV PYTHONPATH=/app

# Default command to run the pipeline
CMD ["python", "-u", "src/main.py"]
