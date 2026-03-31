# Base Python Image
FROM python:3.13-slim

# System setup
RUN apt-get update && apt-get install -y \
    git \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install poetry
RUN pip install poetry

# App directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml .

# Install dependencies
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --no-root

# Copy code
COPY . .

# Final Install
RUN poetry install --no-interaction --no-ansi

# Expose port
EXPOSE 8081

# Run (from Procfile)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8081"]
