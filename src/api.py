import os
import tempfile
import logging
from typing import Optional
from fastapi import APIRouter, File, Form, UploadFile, HTTPException
from pydantic import BaseModel

from processor import process_voice_file
from models import check_models_loaded

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

class TranscriptionResponse(BaseModel):
    segments: list

@router.post("/api/inference/", response_model=TranscriptionResponse)
async def api_inference(
    audio_file: UploadFile = File(...),
    num_speakers: Optional[int] = Form(None)
):
    """
    API endpoint for voice transcription.
    Accepts an audio file and an optional number of speakers.
    Returns the transcription with speaker diarization.
    """
    if not audio_file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    logger.info(f"Processing file: {audio_file.filename}")
    
    # Save uploaded file to a temporary location
    temp_path = os.path.join(tempfile.gettempdir(), audio_file.filename)
    with open(temp_path, "wb") as f:
        content = await audio_file.read()
        f.write(content)
    
    try:
        # Process the audio file
        logger.info(f"Starting processing of file {temp_path}")
        result = process_voice_file(temp_path, num_speakers=num_speakers)
        logger.info("File processing completed successfully")
        return result
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)

@router.get("/")
async def index():
    """
    Simple root endpoint
    """
    return {
        "message": "Speech-to-Text API is running",
        "usage": "POST /api/inference/ with an audio_file",
        "version": "1.0.0"
    }

@router.get("/debug/models")
async def debug_models():
    """
    Debug endpoint to check if models are loaded correctly
    """
    logger.info("Checking model status")
    try:
        status = check_models_loaded()
        return {
            "status": "ok",
            "models": status
        }
    except Exception as e:
        logger.error(f"Error checking models: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "error": str(e)
        }
