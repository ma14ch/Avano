# Avano - Persian Multi-Speaker Voice Transcription Service

## Avano is a powerful Persian speech-to-text service designed for multi-speaker transcription in a single audio session.  

## <div dir="rtl">آوانو یک سرویس قدرتمند تبدیل صوت به متن فارسی است که برای پیاده‌سازی متن گفتارِ چند سخنران در یک جلسه صوتی طراحی شده است.</div>


<img width="1536" height="1024" alt="Avano" src="https://github.com/user-attachments/assets/d81e4109-0932-491b-b9f4-7d87ab14ac2e" />


## Model Information

Avano uses [OpenAI Whisper Large v3](https://huggingface.co/openai/whisper-large-v3) for speech recognition, configured to transcribe Persian audio.

### Key Features of the Model
- 🎯 Multilingual speech recognition with Persian transcription support
- 🚀 Based on OpenAI's Whisper Large V3 Turbo architecture
- 📊 14.07% Word Error Rate (WER)
- 💪 Optimized for Persian voice transcription

## Installation Guide

### Prerequisites
- Python 3.10 or higher
- CUDA-compatible GPU (recommended)
- Docker and Docker Compose (optional)

### Option 1: Using Docker (Recommended)
1. Clone the repository:
```bash
git clone https://github.com/ma14ch/avano.git
cd avano
```

2. Start the service using Docker Compose:
```bash
docker-compose up --build
```

The service will be available at `http://localhost:5016`.

### Option 2: Manual Installation
1. Clone the repository:
```bash
git clone https://github.com/ma14ch/avano.git
cd avano
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the service:
```bash
python src/main.py
```

The service will be available at `http://localhost:5016`.

### Environment Configuration
- The service automatically detects GPU availability
- Default port is 5016 (can be modified in `main.py`)
- Model files are stored in the `models/` directory

## API Usage Examples with `curl`  

### Basic API Status Check  

Check if the API is running:  

```bash
curl -X GET http://localhost:5016/
````

---

### Speech-to-Text Transcription with Speaker Diarization

Send an audio file for transcription:

```bash
curl -X POST http://localhost:5016/api/inference/ \
  -F "audio_file=@/path/to/your/audio/file.mp3" \
  -F "num_speakers=2"
```

#### Parameters

* `audio_file`: The audio file to transcribe (required)  

* `num_speakers`: Number of speakers to identify (optional)  

---

### Check Model Status

Check if the models are loaded correctly:

```bash
curl -X GET http://localhost:5016/debug/models
```

---

### Response Format

The API returns a JSON response with transcribed segments:

```json
{
  "segments": [
    {
      "speaker": "SPEAKER_0",
      "start": 0.5,
      "end": 5.2,
      "transcription": "متن تبدیل‌شده برای گوینده اول"
    },
    {
      "speaker": "SPEAKER_1",
      "start": 5.8,
      "end": 10.3,
      "transcription": "متن تبدیل‌شده برای گوینده دوم"
    }
  ]
}
```

## Model Limitations
- Optimized for clean audio quality
- Not designed for real-time streaming ASR
- May occasionally produce hallucinations (a common limitation in Whisper models)
- Best performance on standard Persian speech, may have reduced accuracy with heavy accents or dialects

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

