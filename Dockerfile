FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements_essential.txt /app/requirements_essential.txt

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements_essential.txt

# Copy Flask app
COPY flask_app/ /app/

# Copy models directory
COPY models/ /app/models/

# Verify model files exist
RUN ls -la /app/models/ && \
    test -f /app/models/model.pkl && echo "✅ Model file found" && \
    test -f /app/models/vectorizer.pkl && echo "✅ Vectorizer file found"

# Create necessary directories
RUN mkdir -p /app/data /app/logs

# Set environment variables for production
ENV FLASK_ENV=production
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=/app/models

EXPOSE 5000

# Add health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Use gunicorn for production
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "120", "--log-level", "info", "app:app"]