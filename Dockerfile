# CUDA base image
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the application
COPY src/server.py .

# Expose the port
EXPOSE 8000

# Start the server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]