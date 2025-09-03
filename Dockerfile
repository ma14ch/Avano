FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Minsk
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-get update -y
RUN apt-get -y install build-essential
RUN apt install software-properties-common -y
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update -y
RUN apt install python3.10 python3-pip -y

# Set environment variable to indicate we're in Docker
ENV RUNNING_IN_DOCKER=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy app code
COPY src /app/src
COPY ui /app/ui

# Make models dir available
RUN mkdir -p /app/models

# Ensure Python can import from /app and /app/src
ENV PYTHONPATH="/app:/app/src"

# Shared persistent Hugging Face cache (mounted via docker-compose)
ENV HF_HOME=/app/hf-cache
ENV TRANSFORMERS_CACHE=/app/hf-cache/transformers
RUN mkdir -p $TRANSFORMERS_CACHE

# Optionally pre-download models at build time: --build-arg PRELOAD_MODELS=true
ARG PRELOAD_MODELS=false
RUN if [ "$PRELOAD_MODELS" = "true" ]; then \
  python3 -c "from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor; m='vhdm/whisper-large-fa-v1'; AutoProcessor.from_pretrained(m, cache_dir='$TRANSFORMERS_CACHE'); AutoModelForSpeechSeq2Seq.from_pretrained(m, cache_dir='$TRANSFORMERS_CACHE'); print('Preloaded:', m)"; \
fi

# Expose API and UI ports
EXPOSE 5016
EXPOSE 7860

# Command to run the application 
CMD ["python3", "/app/src/main.py"]
# Expose API and UI ports
EXPOSE 5016
EXPOSE 7860

# Command to run the application 
CMD ["python3", "/app/src/main.py"]
