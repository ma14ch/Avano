# import librosa
# import torch
# import json
# import os
# import sys
# import tempfile
# import uuid
# from pathlib import Path

# import torch
# from hezar.models import \
#     Model  # make sure this import works in your environment
# from pyannote.audio import Pipeline
# from pydub import AudioSegment

# # Load model directly
# from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

# # # Load the processor and model
# # processor = AutoProcessor.from_pretrained("steja/whisper-large-persian")
# # model = AutoModelForSpeechSeq2Seq.from_pretrained(
# #     "steja/whisper-large-persian")


# processor = AutoProcessor.from_pretrained("openai/whisper-large-v3-turbo")
# model = AutoModelForSpeechSeq2Seq.from_pretrained(
#     "openai/whisper-large-v3-turbo")


# # Set device to CUDA if available
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model.to(device)


# # Load the audio file (this returns a waveform and sample rate)
# audio_input, sample_rate = librosa.load("Record.mp3", sr=16000)

# # Process the audio data
# inputs = processor(audio_input, sampling_rate=sample_rate, return_tensors="pt")
# # Move the input features to the same device as the model
# input_features = inputs["input_features"].to(device)

# with torch.no_grad():
#     generated_ids = model.generate(input_features, num_beams=1)

# # Decode the generated IDs to get the transcription text
# transcription = processor.batch_decode(
#     generated_ids, skip_special_tokens=True)[0]
# print(transcription)


import json
import os
import sys
import tempfile
import uuid
from pathlib import Path

import librosa
import torch
from pyannote.audio import Pipeline
from pydub import AudioSegment
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

# --------------------------
# Offline Transcription Setup using Whisper-large-v3-turbo
# --------------------------
# Load the processor and model from Hugging Face
processor = AutoProcessor.from_pretrained("openai/whisper-large-v3-turbo")
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    "openai/whisper-large-v3-turbo")

# Set device to CUDA if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


def convert_voice_to_text(audio_data: bytes) -> str:
    """
    Converts raw audio bytes to text using the Whisper-large-v3-turbo model.
    Writes the bytes to a temporary MP3 file, loads the audio with librosa,
    transcribes it, and then removes the temporary file.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio_file:
        temp_audio_file.write(audio_data)
        audio_path = temp_audio_file.name

    try:
        # Load the audio file with librosa (forcing a 16kHz sample rate)
        audio_input, sample_rate = librosa.load(audio_path, sr=16000)
        # Prepare inputs for the model
        inputs = processor(
            audio_input, sampling_rate=sample_rate, return_tensors="pt")
        input_features = inputs["input_features"].to(device)
        # Generate transcription (using a simple beam search)
        with torch.no_grad():
            generated_ids = model.generate(
                input_features, num_beams=1, language="persian")
        # Decode the generated ids to text
        transcription = processor.batch_decode(
            generated_ids, skip_special_tokens=True)[0]
    except Exception as e:
        transcription = f"Error during transcription: {str(e)}"
    finally:
        os.remove(audio_path)
    return transcription


# --------------------------
# Diarization Setup
# --------------------------


def load_pipeline_from_pretrained(path_to_config: str | Path) -> Pipeline:
    """
    Loads a Pyannote diarization pipeline from the given configuration file.
    """
    path_to_config = Path(path_to_config)
    print(f"Loading pyannote pipeline from {path_to_config}...")
    cwd = Path.cwd().resolve()
    cd_to = path_to_config.parent.parent.resolve()
    print(f"Changing working directory to {cd_to}")
    os.chdir(cd_to)

    pipeline = Pipeline.from_pretrained(path_to_config)
    pipeline.to(torch.device("cuda"))

    print(f"Changing working directory back to {cwd}")
    os.chdir(cwd)

    return pipeline


# Adjust this path as needed for your diarization configuration file
PATH_TO_CONFIG = "models/pyannote_diarization_config.yaml"
DIARIZATION_PIPELINE = load_pipeline_from_pretrained(PATH_TO_CONFIG)


def diarize_audio(audio_file_path: str):
    """
    Performs speaker diarization on the given audio file.
    Returns a list of tuples: (speaker_label, segment_start, segment_end).
    """
    diarization = DIARIZATION_PIPELINE(audio_file_path)
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


def process_voice_file(audio_file_path: str) -> dict:
    """
    Process the given audio file:
     - Performs diarization to get speaker segments.
     - Splits the audio accordingly.
     - Transcribes each segment.
    Returns a dictionary suitable for JSON output.
    """
    # Diarize the full audio file
    speaker_segments = diarize_audio(audio_file_path)
    # Segment the audio based on diarization results
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
            # Clean up the temporary segment file
            if os.path.exists(segment_path):
                os.remove(segment_path)
        results["segments"].append({
            "speaker": speaker,
            "start": start,
            "end": end,
            "transcription": transcription
        })

    return results


def process_audio_file(audio_file_path: str) -> str:
    """
    Processes the audio file by performing diarization and transcription.
    Returns the combined transcription from all segments.
    """
    results = process_voice_file(audio_file_path)
    combined_transcription = ""
    for segment in results.get("segments", []):
        combined_transcription += f"{segment['speaker']}: {segment['transcription']}\n"
    return combined_transcription


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Speaker Diarization and Transcription App")
    parser.add_argument(
        "audio_file", help="Path to the input audio file (e.g., MP3, WAV)")
    parser.add_argument("--json", action="store_true",
                        help="Output results in JSON format")
    args = parser.parse_args()

    audio_file_path = args.audio_file
    if not os.path.exists(audio_file_path):
        print(f"Error: The file {audio_file_path} does not exist.")
        sys.exit(1)

    print("Processing audio file...")
    results = process_voice_file(audio_file_path)
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        transcription = process_audio_file(audio_file_path)
        print("Combined Transcription:")
        print(transcription)


if __name__ == "__main__":
    main()
