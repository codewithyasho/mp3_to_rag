# Use official lightweight Python image
FROM python:3.12-slim

# Install system dependencies (ffmpeg & ffprobe required for pydub audio processing)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user required by Hugging Face Spaces (UID 1000)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:${PATH}"
WORKDIR /home/user/app

# Copy requirements file and install dependencies
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application codebase
COPY --chown=user . .

# Hugging Face Spaces expects containers to serve on port 7860
EXPOSE 7860

# Run Streamlit on port 7860 listening on all interfaces
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
