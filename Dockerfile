FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000 8501 5000

# Default command (will be overridden in docker-compose.yml)
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
