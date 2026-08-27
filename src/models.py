import os
import torch
import logging
from pathlib import Path
from pyannote.audio import Pipeline
from huggingface_hub import snapshot_download
from pyannote.audio.core.task import Problem, Resolution, Specifications
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    WhisperFeatureExtractor,
    WhisperProcessor,
    WhisperTokenizerFast,
)

# Available ASR models. Select the active one with the ASR_MODEL env var (one
# of the keys below). All of them can be pre-downloaded locally so switching
# between them does not require a new download.
ASR_MODEL_REPOS = {
    "v3": "openai/whisper-large-v3",
    "persian-v4": "nezamisafa/whisper-persian-v4",
    "persian-bf16": "AmirMohseni/whisper-large-v3-persian-bf16",
    "qwen3-asr-1.7b": "Qwen/Qwen3-ASR-1.7B-hf",
}

# Which inference code path a given ASR_MODEL_REPOS key uses:
#  - "whisper": encoder-decoder Whisper checkpoints, loaded with
#    WhisperProcessor + AutoModelForSpeechSeq2Seq.
#  - "qwen3_asr": Qwen3-ASR Transformers-native chat/audio models, loaded with
#    AutoProcessor + AutoModelForMultimodalLM. Requires transformers>=5.13.0.
ASR_MODEL_FAMILIES = {
    "v3": "whisper",
    "persian-v4": "whisper",
    "persian-bf16": "whisper",
    "qwen3-asr-1.7b": "qwen3_asr",
}

# Backward-compatible alias for older tooling that still refers to
# WHISPER_MODELS.
WHISPER_MODELS = ASR_MODEL_REPOS

ASR_MODEL_KEY = os.getenv("ASR_MODEL", os.getenv("WHISPER_MODEL", "persian-v4"))
if ASR_MODEL_KEY not in ASR_MODEL_REPOS:
    ASR_MODEL_KEY = "persian-v4"
ASR_MODEL_ID = ASR_MODEL_REPOS[ASR_MODEL_KEY]
ASR_MODEL_FAMILY = ASR_MODEL_FAMILIES.get(ASR_MODEL_KEY, "whisper")

# Backward-compatible aliases.
WHISPER_MODEL_KEY = ASR_MODEL_KEY
WHISPER_MODEL_ID = ASR_MODEL_ID
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
qwen3_processor = None
qwen3_model = None
diarization_pipeline = None


def _cache_dir() -> str:
    return os.getenv("TRANSFORMERS_CACHE") or os.path.join(os.getenv("HF_HOME", "/app/hf-cache"), "transformers")


def download_whisper_model(model_id: str) -> str:
    """Download (or reuse cached) snapshot for a given Whisper model id."""
    logger.info("Downloading Whisper model: %s", model_id)
    return snapshot_download(
        model_id,
        cache_dir=_cache_dir(),
        allow_patterns=WHISPER_MODEL_FILES,
    )


def download_asr_model(model_key: str) -> str:
    """Download (or reuse cached) snapshot for a given ASR_MODEL_REPOS key."""
    repo = ASR_MODEL_REPOS.get(model_key)
    if repo is None:
        raise ValueError(f"Unknown ASR model key: {model_key}")
    family = ASR_MODEL_FAMILIES.get(model_key, "whisper")
    logger.info("Downloading ASR model '%s' (%s, family=%s)", model_key, repo, family)
    if family == "whisper":
        return snapshot_download(repo, cache_dir=_cache_dir(), allow_patterns=WHISPER_MODEL_FILES)
    # Other families (e.g. qwen3_asr) have a different file layout, so fetch
    # the full snapshot instead of a Whisper-specific allow-list.
    return snapshot_download(repo, cache_dir=_cache_dir())


def download_all_asr_models() -> None:
    """Pre-download every model in ASR_MODEL_REPOS so switching is instant."""
    for key in ASR_MODEL_REPOS:
        download_asr_model(key)


# Backward-compatible alias.
download_all_whisper_models = download_all_asr_models


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


def load_qwen3_asr_model():
    """Load the Qwen3-ASR model selected via ASR_MODEL and its processor.

    Qwen3-ASR is a Transformers-native chat/audio model (not a Whisper
    encoder-decoder), so it uses AutoProcessor + AutoModelForMultimodalLM
    instead of WhisperProcessor + AutoModelForSpeechSeq2Seq. This requires
    transformers>=5.13.0.
    """
    global qwen3_processor, qwen3_model

    logger.info("Loading Qwen3-ASR model: %s (ASR_MODEL=%s)", ASR_MODEL_ID, ASR_MODEL_KEY)
    try:
        try:
            from transformers import AutoModelForMultimodalLM
        except ImportError as exc:
            raise RuntimeError(
                "Qwen3-ASR requires transformers>=5.13.0 (AutoModelForMultimodalLM "
                "is not available in the installed version). Upgrade the "
                "'transformers' package to use this model."
            ) from exc

        model_path = snapshot_download(ASR_MODEL_ID, cache_dir=_cache_dir())
        qwen3_processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)

        torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        qwen3_model = AutoModelForMultimodalLM.from_pretrained(
            model_path,
            local_files_only=True,
            dtype=torch_dtype,
            low_cpu_mem_usage=True,
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        qwen3_model.to(device)

        if qwen3_model is None or qwen3_processor is None:
            logger.error("Failed to load Qwen3-ASR model or processor")
            raise RuntimeError("Qwen3-ASR model loading failed")

        logger.info("Qwen3-ASR model loaded successfully")
        return qwen3_processor, qwen3_model
    except Exception as e:
        logger.error(f"Error loading Qwen3-ASR model: {str(e)}")
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
    """Get or initialize Whisper model. Kept as a backward-compatible alias;
    prefer get_asr_model() which also supports non-Whisper families."""
    global whisper_processor, whisper_model
    if whisper_processor is None or whisper_model is None:
        logger.info("Initializing Whisper model (first request)")
        whisper_processor, whisper_model = load_whisper_model()
    return whisper_processor, whisper_model


def get_asr_model():
    """Get or initialize the active ASR model according to ASR_MODEL_FAMILY.
    Returns (processor, model, family)."""
    global whisper_processor, whisper_model, qwen3_processor, qwen3_model

    if ASR_MODEL_FAMILY == "qwen3_asr":
        if qwen3_processor is None or qwen3_model is None:
            logger.info("Initializing Qwen3-ASR model (first request)")
            qwen3_processor, qwen3_model = load_qwen3_asr_model()
        return qwen3_processor, qwen3_model, "qwen3_asr"

    if whisper_processor is None or whisper_model is None:
        logger.info("Initializing Whisper model (first request)")
        whisper_processor, whisper_model = load_whisper_model()
    return whisper_processor, whisper_model, "whisper"

def get_diarization_pipeline():
    """Get or initialize diarization pipeline"""
    global diarization_pipeline
    if diarization_pipeline is None:
        logger.info("Initializing diarization pipeline (first request)")
        diarization_pipeline = load_diarization_pipeline()
    return diarization_pipeline

def check_models_loaded():
    """Debug function to check if models are loaded"""
    if ASR_MODEL_FAMILY == "qwen3_asr":
        active_processor_loaded = qwen3_processor is not None
        active_model_loaded = qwen3_model is not None
    else:
        active_processor_loaded = whisper_processor is not None
        active_model_loaded = whisper_model is not None

    result = {
        # Backward-compatible keys: reflect whichever model family is active.
        "whisper_model": active_model_loaded,
        "whisper_processor": active_processor_loaded,
        "asr_model_key": ASR_MODEL_KEY,
        "asr_model_family": ASR_MODEL_FAMILY,
        "diarization_pipeline": diarization_pipeline is not None,
        "cuda_available": torch.cuda.is_available(),
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }
    
    if torch.cuda.is_available():
        result["cuda_device_count"] = torch.cuda.device_count()
        result["cuda_device_name"] = torch.cuda.get_device_name(0)
        
    return result
