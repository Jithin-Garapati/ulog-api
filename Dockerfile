FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p uploads data_cache csv_topics

# Set environment variables
ENV FLASK_APP=api.py
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1

# Railway injects the PORT environment variable
EXPOSE ${PORT:-5000}

# Start command for Railway deployment
CMD gunicorn --bind 0.0.0.0:${PORT:-5000} --workers=2 --threads=4 --timeout=120 api:app
