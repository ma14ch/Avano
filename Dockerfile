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

# Copy src directory to /app/src in the container
COPY src /app/src

# Create models directory and copy models if they exist
RUN mkdir -p /app/models

# Expose port
EXPOSE 5016

# Command to run the application 
CMD ["python3", "/app/src/main.py"]
