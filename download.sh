#!/bin/bash

# Download script for Whisper Large V3 Turbo model
# This script downloads the model locally to avoid downloading it every time

set -e  # Exit on any error

PROJECT_DIR="$(dirname "$(readlink -f "$0")")"
MODELS_DIR="$PROJECT_DIR/localmodels"
WHISPER_MODEL_DIR="$MODELS_DIR/whisper-large-v3-turbo"

echo "Starting Whisper model download..."
echo "Project directory: $PROJECT_DIR"
echo "Models directory: $MODELS_DIR"

# Create models directory if it doesn't exist
mkdir -p "$MODELS_DIR"

# Change to models directory
cd "$MODELS_DIR"

# Check if git-lfs is installed
if ! command -v git-lfs &> /dev/null; then
    echo "Error: git-lfs is not installed. Please install it first:"
    echo "https://git-lfs.com"
    exit 1
fi

# Initialize git-lfs
echo "Initializing git-lfs..."
git lfs install

# Check if model directory already exists
if [ -d "$WHISPER_MODEL_DIR" ]; then
    echo "Whisper model directory already exists at: $WHISPER_MODEL_DIR"
    read -p "Do you want to re-download? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing model directory..."
        rm -rf "$WHISPER_MODEL_DIR"
    else
        echo "Skipping download. Using existing model."
        exit 0
    fi
fi

# Clone the Whisper model repository
echo "Downloading Whisper Large V3 Turbo model..."
echo "This may take several minutes depending on your internet connection..."

git clone https://huggingface.co/openai/whisper-large-v3-turbo

# Verify the download
if [ -d "$WHISPER_MODEL_DIR" ]; then
    echo "✅ Whisper model downloaded successfully to: $WHISPER_MODEL_DIR"
    
    # Show directory size
    du -sh "$WHISPER_MODEL_DIR" 2>/dev/null || echo "Model directory created"
    
    # List key files
    echo "Key model files:"
    ls -la "$WHISPER_MODEL_DIR"/ | grep -E '\.(json|bin|safetensors|txt)$' || true
else
    echo "❌ Error: Failed to download Whisper model"
    exit 1
fi

echo "Download completed successfully!"
echo "The model will now be loaded locally from: $WHISPER_MODEL_DIR"
