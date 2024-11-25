# Use Python base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    gzip \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install flask fasttext scipy

# Download and unzip the FastText embeddings
RUN curl -o cc.en.300.bin.gz https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz && \
    gunzip cc.en.300.bin.gz

# Copy the server script into the container
COPY server.py server.py

# Expose the Flask port
EXPOSE 5555

# Command to run the server
CMD ["python", "server.py"]
