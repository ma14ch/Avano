
# VoiceTranscription-whisper3

This repository is built for multi-user speech-to-text transcription in a single audio session.  
این مخزن برای پیاده‌سازی تبدیل گفتار به متن در یک جلسه صوتی با چند کاربر به‌صورت همزمان طراحی شده است.

![ChatGPT Image May 16, 2025, 05_55_07 PM](https://github.com/user-attachments/assets/bba046c7-12fe-45ad-9244-741632a07b1a)

---

## API Usage Examples with `curl`  
## نمونه‌هایی از استفاده از API با `curl`

### Basic API Status Check  
### بررسی وضعیت پایه API

Check if the API is running:  
برای بررسی اینکه آیا API در حال اجرا است:

```bash
curl -X GET http://localhost:5016/
````

---

### Speech-to-Text Transcription with Diarization

### تبدیل گفتار به متن همراه با شناسایی گویندگان (Diarization)

Send an audio file for transcription:
برای ارسال یک فایل صوتی جهت تبدیل به متن:

```bash
curl -X POST http://localhost:5016/api/inference/ \
  -F "audio_file=@/path/to/your/audio/file.mp3" \
  -F "num_speakers=2"
```

#### Parameters

#### پارامترها

* `audio_file`: The audio file to transcribe (required)
  فایل صوتی برای تبدیل به متن (اجباری)

* `num_speakers`: The number of speakers to identify in the audio (optional)
  تعداد گویندگان موجود در فایل صوتی (اختیاری)

---

### Check Model Status

### بررسی وضعیت مدل‌ها

Check if the models are loaded correctly:
برای بررسی اینکه آیا مدل‌ها به درستی بارگذاری شده‌اند:

```bash
curl -X GET http://localhost:5016/debug/models
```

---

### Response Format

### قالب پاسخ API

The API will return a JSON response with the transcribed segments:
پاسخ API به صورت JSON خواهد بود و شامل بخش‌های مختلف تبدیل‌شده به متن است:

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

```json
{
  "segments": [
    {
      "speaker": "SPEAKER_0",
      "start": 0.5,
      "end": 5.2,
      "transcription": "متن تبدیل‌شده برای گوینده ۰"
    },
    {
      "speaker": "SPEAKER_1",
      "start": 5.8,
      "end": 10.3,
      "transcription": "متن تبدیل‌شده برای گوینده ۱"
    }
  ]
}
```


