# Use Python base image
FROM python:3

# Set working directory
WORKDIR /usr/src/app

# Install Python dependencies
RUN pip install flask fasttext

# Download and unzip the FastText embeddings
RUN curl -o cc.en.300.bin.gz https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz && \
    gunzip cc.en.300.bin.gz

COPY . .

# Expose the Flask port
EXPOSE 5555

# Command to run the server
CMD ["python", "server.py"]
