import uvicorn
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api import router
from models import get_whisper_model, get_diarization_pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Speech-to-Text API",
    description="API for transcribing audio files using Whisper and speaker diarization",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the API router
app.include_router(router)

@app.on_event("startup")
async def startup_event():
    """
    Verify models load correctly on startup
    """
    logger.info("Application starting up - verifying model loading")
    try:
        # Pre-load models during startup
        logger.info("Pre-loading Whisper model...")
        get_whisper_model()
        logger.info("Pre-loading diarization pipeline...")
        get_diarization_pipeline()
        logger.info("All models loaded successfully")
    except Exception as e:
        logger.error(f"Error loading models during startup: {str(e)}", exc_info=True)
        # We don't raise the exception here to allow the app to start even if models fail
        # This allows debugging endpoints to still be accessible

if __name__ == "__main__":
    # Run the FastAPI application with uvicorn
    logger.info("Starting Speech-to-Text API server")
    uvicorn.run("main:app", host="0.0.0.0", port=5016, reload=True)
