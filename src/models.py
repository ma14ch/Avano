import os
import torch
import logging
from pathlib import Path
from pyannote.audio import Pipeline
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables to store loaded models
whisper_processor = None
whisper_model = None
diarization_pipeline = None

def load_whisper_model():
    """Load Whisper model and processor"""
    global whisper_processor, whisper_model
    
    logger.info("Loading Whisper model...")
    try:
        whisper_processor = AutoProcessor.from_pretrained("vhdm/whisper-large-fa-v1")
        whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained("vhdm/whisper-large-fa-v1")
        
        # Set device to CUDA if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        whisper_model.to(device)
        
        # Verify the model is loaded
        if whisper_model is None or whisper_processor is None:
            logger.error("Failed to load Whisper model or processor")
            raise RuntimeError("Whisper model loading failed")
            
        logger.info("Whisper model loaded successfully")
        return whisper_processor, whisper_model
    except Exception as e:
        logger.error(f"Error loading Whisper model: {str(e)}")
        raise

def load_diarization_pipeline(path_to_config: str | Path = None) -> Pipeline:
    """Load pyannote diarization pipeline"""
    global diarization_pipeline
    
    if diarization_pipeline is not None:
        logger.info("Using already loaded diarization pipeline")
        return diarization_pipeline
    
    if path_to_config is None:
        path_to_config = "/app/src/models/pyannote_diarization_config.yaml"
    
    path_to_config = Path(path_to_config)
    logger.info(f"Loading pyannote pipeline from {path_to_config}...")
    
    try:
        cwd = Path.cwd().resolve()
        cd_to = path_to_config.parent.parent.resolve()
        
        logger.debug(f"Changing directory from {cwd} to {cd_to}")
        os.chdir(cd_to)
        
        if not path_to_config.exists():
            logger.error(f"Config file not found at {path_to_config}")
            raise FileNotFoundError(f"Diarization config not found: {path_to_config}")
            
        diarization_pipeline = Pipeline.from_pretrained(path_to_config)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Moving diarization pipeline to device: {device}")
        diarization_pipeline.to(device)
        os.chdir(cwd)
        
        logger.info("Diarization pipeline loaded successfully")
        return diarization_pipeline
    except Exception as e:
        logger.error(f"Error loading diarization pipeline: {str(e)}")
        if 'cwd' in locals() and os.path.exists(cwd):
            os.chdir(cwd)
        raise

def get_whisper_model():
    """Get or initialize Whisper model"""
    global whisper_processor, whisper_model
    if whisper_processor is None or whisper_model is None:
        logger.info("Initializing Whisper model (first request)")
        whisper_processor, whisper_model = load_whisper_model()
    return whisper_processor, whisper_model

def get_diarization_pipeline():
    """Get or initialize diarization pipeline"""
    global diarization_pipeline
    if diarization_pipeline is None:
        logger.info("Initializing diarization pipeline (first request)")
        diarization_pipeline = load_diarization_pipeline()
    return diarization_pipeline

def check_models_loaded():
    """Debug function to check if models are loaded"""
    result = {
        "whisper_model": whisper_model is not None,
        "whisper_processor": whisper_processor is not None,
        "diarization_pipeline": diarization_pipeline is not None,
        "cuda_available": torch.cuda.is_available(),
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }
    
    if torch.cuda.is_available():
        result["cuda_device_count"] = torch.cuda.device_count()
        result["cuda_device_name"] = torch.cuda.get_device_name(0)
        
    return result
