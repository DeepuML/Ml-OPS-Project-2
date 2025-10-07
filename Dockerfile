FROM python:3.9-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements_essential.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy Flask app
COPY flask_app/ /app/

# Copy models directory
COPY models/ /app/models/

# Create necessary directories
RUN mkdir -p /app/data /app/logs

# Set environment variables for production
ENV FLASK_ENV=production
ENV PYTHONPATH=/app

EXPOSE 5000

# Use gunicorn for production
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "120", "app:app"]m 