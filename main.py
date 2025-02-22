import librosa
import torch
# Load model directly
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

# # Load the processor and model
# processor = AutoProcessor.from_pretrained("steja/whisper-large-persian")
# model = AutoModelForSpeechSeq2Seq.from_pretrained(
#     "steja/whisper-large-persian")


processor = AutoProcessor.from_pretrained("openai/whisper-large-v3-turbo")
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    "openai/whisper-large-v3-turbo")


# Set device to CUDA if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


# Load the audio file (this returns a waveform and sample rate)
audio_input, sample_rate = librosa.load("Record.mp3", sr=16000)

# Process the audio data
inputs = processor(audio_input, sampling_rate=sample_rate, return_tensors="pt")
# Move the input features to the same device as the model
input_features = inputs["input_features"].to(device)

with torch.no_grad():
    generated_ids = model.generate(input_features, num_beams=1)

# Decode the generated IDs to get the transcription text
transcription = processor.batch_decode(
    generated_ids, skip_special_tokens=True)[0]
print(transcription)
