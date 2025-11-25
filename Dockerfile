FROM python:3.10-slim

WORKDIR /app

# Ensure Python output is sent directly to terminal (helps debug crashes)
ENV PYTHONUNBUFFERED=1

# Install system dependencies
# build-essential might be needed for some python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first to avoid downloading huge CUDA binaries
# This reduces image size from ~8GB to ~1GB
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Copy the application code
COPY rainwords/ rainwords/

# Preload models during build to speed up startup and prevent timeouts
RUN python rainwords/preload_models.py

# Explicitly tell Railway we are using port 8080
EXPOSE 8080

# Command to run the application using the PORT environment variable
# We use /bin/sh -c to ensure the variable expansion works correctly
CMD ["/bin/sh", "-c", "uvicorn rainwords.main:app --host 0.0.0.0 --port ${PORT:-8080} --proxy-headers"]
