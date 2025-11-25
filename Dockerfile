FROM python:3.10-slim

WORKDIR /app

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

# Download NLTK data during build to avoid doing it at runtime
RUN python -m nltk.downloader punkt averaged_perceptron_tagger averaged_perceptron_tagger_eng stopwords

# Copy the application code
COPY rainwords/ rainwords/

# Expose the port
EXPOSE 8000

# Command to run the application
# We use --proxy-headers because it will likely be behind a reverse proxy (Nginx/Cloudflare/Railway LB)
CMD ["uvicorn", "rainwords.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
