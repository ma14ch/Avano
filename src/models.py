import os
import torch
import logging
from pathlib import Path
from pyannote.audio import Pipeline
from huggingface_hub import snapshot_download
from pyannote.audio.core.task import Problem, Resolution, Specifications
from transformers import (
    AutoModelForSpeechSeq2Seq,
    WhisperFeatureExtractor,
    WhisperProcessor,
    WhisperTokenizerFast,
)

# Available Whisper models. Select the active one with the WHISPER_MODEL env var
# (one of the keys below). Both can be pre-downloaded locally so switching
# between them does not require a new download.
WHISPER_MODELS = {
    "v3": "openai/whisper-large-v3",
    "persian-v4": "nezamisafa/whisper-persian-v4",
    "persian-bf16": "AmirMohseni/whisper-large-v3-persian-bf16",
}
WHISPER_MODEL_KEY = os.getenv("WHISPER_MODEL", "persian-v4")
WHISPER_MODEL_ID = WHISPER_MODELS.get(WHISPER_MODEL_KEY, WHISPER_MODELS["persian-v4"])
WHISPER_MODEL_FILES = (
    "added_tokens.json",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "model.safetensors",
    "model-*-of-*.safetensors",
    "model.safetensors.index.json",
    "normalizer.json",
    "preprocessor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)

# PyTorch 2.6+ loads checkpoints in weights-only mode by default. The bundled
# diarization checkpoints are trusted project assets and serialize TorchVersion.
try:
    torch.serialization.add_safe_globals(
        [
            torch.torch_version.TorchVersion,
            Problem,
            Resolution,
            Specifications,
        ]
    )
except AttributeError:
    pass

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

def download_whisper_model(model_id: str) -> str:
    """Download (or reuse cached) snapshot for a given Whisper model id."""
    cache_dir = os.getenv("TRANSFORMERS_CACHE") or os.path.join(os.getenv("HF_HOME", "/app/hf-cache"), "transformers")
    logger.info("Downloading Whisper model: %s", model_id)
    return snapshot_download(
        model_id,
        cache_dir=cache_dir,
        allow_patterns=WHISPER_MODEL_FILES,
    )


def download_all_whisper_models() -> None:
    """Pre-download every model in WHISPER_MODELS so switching is instant."""
    for key, model_id in WHISPER_MODELS.items():
        logger.info("Pre-downloading model '%s' (%s)", key, model_id)
        download_whisper_model(model_id)


def load_whisper_model():
    """Load the Whisper model selected via WHISPER_MODEL and its processor."""
    global whisper_processor, whisper_model
    
    logger.info("Loading Whisper model: %s (WHISPER_MODEL=%s)", WHISPER_MODEL_ID, WHISPER_MODEL_KEY)
    try:
        # Use shared cache directory (set by Docker ENV)
        cache_dir = os.getenv("TRANSFORMERS_CACHE") or os.path.join(os.getenv("HF_HOME", "/app/hf-cache"), "transformers")

        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        # Fetch only files present in the Whisper repository, then load strictly
        # from that local snapshot. This avoids requests for optional chat files
        # that are not part of a speech-recognition model.
        model_path = snapshot_download(
            WHISPER_MODEL_ID,
            cache_dir=cache_dir,
            allow_patterns=WHISPER_MODEL_FILES,
        )
        feature_extractor = WhisperFeatureExtractor.from_pretrained(
            model_path,
            local_files_only=True,
        )
        tokenizer = WhisperTokenizerFast.from_pretrained(
            model_path,
            local_files_only=True,
        )
        whisper_processor = WhisperProcessor(feature_extractor, tokenizer)
        whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_path,
            local_files_only=True,
            dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        
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
