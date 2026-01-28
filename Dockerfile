# Use official Python slim image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV ENV_TYPE=CLOUD
ENV EASYOCR_MODEL_DIR=/app/easyocr_models
ENV PORT=10000
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV PYTORCH_NO_CUDA_MEMORY_CACHING=1

# Install system dependencies (Full compatibility requirement)
RUN apt-get update && apt-get install -y \
    build-essential \
    g++ \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements (we will create this next)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download EasyOCR models (Tamil, English) to avoid runtime delay
# We use a dummy script to trigger the download during build
COPY setup_models.py .
RUN python setup_models.py

# Copy the rest of the app
COPY . .

# Expose port
EXPOSE 10000

# Command to run the server
CMD ["python", "api_server.py"]
