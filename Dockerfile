# Base Dockerfile for Intelligent E-commerce Monitoring System

# Use the official Python 3.11 image as the base image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Add a non-root user and switch to it
RUN useradd -m appuser \
    && chown -R appuser /app
USER appuser

# Copy the rest of the application code
COPY . /app

# Default command
CMD ["streamlit", "run", "frontend/app.py"]