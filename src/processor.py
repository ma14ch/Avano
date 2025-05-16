import os
import tempfile
import uuid
import librosa
import torch
import logging
from pydub import AudioSegment
from pathlib import Path

from models import get_whisper_model, get_diarization_pipeline

# Configure logging
logger = logging.getLogger(__name__)

def convert_voice_to_text(audio_data: bytes) -> str:
    """
    Converts raw audio bytes to text using the Whisper model.
    """
    logger.info("Converting voice to text")
    processor, model = get_whisper_model()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio_file:
        temp_audio_file.write(audio_data)
        audio_path = temp_audio_file.name

    try:
        # Load the audio file (forcing a 16kHz sample rate)
        logger.debug(f"Loading audio file from {audio_path}")
        audio_input, sample_rate = librosa.load(audio_path, sr=16000)
        logger.debug(f"Audio loaded, sample rate: {sample_rate}Hz, duration: {len(audio_input)/sample_rate:.2f}s")
        
        inputs = processor(
            audio_input, sampling_rate=sample_rate, return_tensors="pt")
        input_features = inputs["input_features"].to(device)
        
        logger.debug("Running Whisper inference")
        with torch.no_grad():
            generated_ids = model.generate(
                input_features, num_beams=1, language="persian")
        transcription = processor.batch_decode(
            generated_ids, skip_special_tokens=True)[0]
        logger.info(f"Transcription completed, length: {len(transcription)} characters")
    except Exception as e:
        logger.error(f"Error during transcription: {str(e)}", exc_info=True)
        transcription = f"Error during transcription: {str(e)}"
    finally:
        os.remove(audio_path)
    return transcription

def diarize_audio(audio_file_path: str):
    """
    Performs speaker diarization on the given audio file.
    Returns a list of tuples: (speaker_label, segment_start, segment_end).
    """
    diarization_pipeline = get_diarization_pipeline()
    diarization = diarization_pipeline(audio_file_path)
    speaker_segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        speaker_segments.append((speaker, turn.start, turn.end))
    return speaker_segments

def segment_audio_by_speaker(audio_file_path: str, speaker_segments: list):
    """
    Splits the audio file into segments based on speaker segments.
    Returns a list of tuples: (speaker_label, segment_start, segment_end, segment_audio_path).
    """
    audio = AudioSegment.from_file(audio_file_path)
    segmented_audios = []

    for speaker, start, end in speaker_segments:
        start_ms = int(start * 1000)
        end_ms = int(end * 1000)
        segment = audio[start_ms:end_ms]
        segment_filename = f"{uuid.uuid4()}_{speaker}.wav"
        temp_dir = tempfile.gettempdir()
        segment_path = os.path.join(temp_dir, segment_filename)
        segment.export(segment_path, format="wav")
        segmented_audios.append((speaker, start, end, segment_path))

    return segmented_audios

def filter_speakers(speaker_segments: list, num_speakers: int) -> list:
    """
    If the diarization produces more speakers than desired,
    filter the segments to include only the top `num_speakers` (by total speaking duration).
    """
    if not num_speakers:
        return speaker_segments

    # Calculate total speaking time per speaker
    speaker_duration = {}
    for speaker, start, end in speaker_segments:
        duration = end - start
        speaker_duration[speaker] = speaker_duration.get(speaker, 0) + duration

    # Select the speakers with the most speaking time
    sorted_speakers = sorted(
        speaker_duration, key=speaker_duration.get, reverse=True)
    allowed_speakers = set(sorted_speakers[:num_speakers])
    return [seg for seg in speaker_segments if seg[0] in allowed_speakers]

def process_voice_file(audio_file_path: str, num_speakers: int = None) -> dict:
    """
    Processes the given audio file:
     - Performs diarization to get speaker segments.
     - Optionally filters segments to the desired number of speakers.
     - Splits the audio and transcribes each segment.
    Returns a dictionary suitable for JSON output.
    """
    speaker_segments = diarize_audio(audio_file_path)
    if num_speakers:
        speaker_segments = filter_speakers(speaker_segments, num_speakers)
    segmented_audios = segment_audio_by_speaker(
        audio_file_path, speaker_segments)

    results = {"segments": []}
    for speaker, start, end, segment_path in segmented_audios:
        try:
            with open(segment_path, "rb") as f:
                audio_bytes = f.read()
            transcription = convert_voice_to_text(audio_bytes)
        except Exception as e:
            transcription = f"Error processing segment: {str(e)}"
        finally:
            if os.path.exists(segment_path):
                os.remove(segment_path)
        results["segments"].append({
            "speaker": speaker,
            "start": start,
            "end": end,
            "transcription": transcription
        })

    return results
