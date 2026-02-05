FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install genarena[web]

# Download parquet benchmark data from HuggingFace
# This dataset contains the prompt/benchmark data (not arena battle results)
ENV HF_DATA_REPO="rhli/genarena"
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('${HF_DATA_REPO}', repo_type='dataset', local_dir='/app/data')"

# Expose port (HuggingFace Spaces default)
EXPOSE 7860

# Start the application
CMD ["python", "-m", "genarena.deploy.app"]