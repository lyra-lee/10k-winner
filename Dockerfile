# Use official Python runtime as a parent image
FROM mdock.daumkakao.io/python:3.11-slim

# Prevent python from writing .pyc files to disc and enable unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install system dependencies (gcc for some python packages)
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first for better layer caching
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose the port Flask listens on
EXPOSE 8000

# Default environment variables (override as needed)
ENV FLASK_ENV=production 

# Start the Flask application
CMD ["python", "hackerthon_agent.py"]
