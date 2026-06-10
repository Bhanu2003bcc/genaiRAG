FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (curl for health check)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

# Create a non-privileged user to run the app securely
RUN adduser --disabled-password --gecos "" appuser && chown -R appuser:appuser /app
USER appuser

# Expose port 8080 to match original Java backend setup
EXPOSE 8080

# Health check — extended start-period to allow gunicorn to fork all workers
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:8080/api/actuator/health || exit 1

# Run via Gunicorn + UvicornWorker (multi-process, production-grade)
# Worker count and all tuning is in gunicorn.conf.py
# Override workers at runtime: docker run -e GUNICORN_WORKERS=2 ...
CMD ["gunicorn", "app.main:app", "--config", "gunicorn.conf.py"]
