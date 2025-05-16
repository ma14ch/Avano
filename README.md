# VoiceTranscription-whisper3
این ریپازیتوری برای تبدیل متن به صوت چند کاربر همزمان در یک جلسه صوتی ایجاد شده است. 


# API Usage Examples with curl

## Basic API Status Check

Check if the API is running:

```bash
curl -X GET http://localhost:5016/
```

## Speech-to-Text Transcription with Diarization

Send an audio file for transcription:

```bash
curl -X POST http://localhost:5016/api/inference/ \
  -F "audio_file=@/path/to/your/audio/file.mp3" \
  -F "num_speakers=2"
```

### Parameters:

- `audio_file`: The audio file to transcribe (required)
- `num_speakers`: The number of speakers to identify in the audio (optional)

## Check Model Status

Check if the models are loaded correctly:

```bash
curl -X GET http://localhost:5016/debug/models
```

## Response Format

The API will return a JSON response with the transcribed segments:

```json
{
  "segments": [
    {
      "speaker": "SPEAKER_0",
      "start": 0.5,
      "end": 5.2,
      "transcription": "Transcribed text for speaker 0"
    },
    {
      "speaker": "SPEAKER_1",
      "start": 5.8,
      "end": 10.3,
      "transcription": "Transcribed text for speaker 1"
    }
  ]
}
```
