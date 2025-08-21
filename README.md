# Avano - Persian Multi-Speaker Voice Transcription Service

Avano is a powerful Persian speech-to-text service designed for multi-speaker transcription in a single audio session.  
آوانو یک سرویس قدرتمند تبدیل گفتار به متن فارسی است که برای پیاده‌سازی ترنویسی چند کاربر در یک جلسه صوتی طراحی شده است.

<img width="1536" height="1024" alt="ChatGPT Image Aug 22, 2025, 01_44_47 AM" src="https://github.com/user-attachments/assets/f0b04dc5-42b7-47af-a895-504fc0c35062" />

## Model Information

Avano uses the state-of-the-art [vhdm/whisper-large-fa-v1](https://huggingface.co/vhdm/whisper-large-fa-v1) model, which is specifically fine-tuned for Persian speech recognition. The model achieves a Word Error Rate (WER) of 14.07% on clean Persian speech data.

### Key Features of the Model
- 🎯 Fine-tuned on high-quality Persian speech data
- 🚀 Based on OpenAI's Whisper Large V3 Turbo architecture
- 📊 14.07% Word Error Rate (WER)
- 💪 Optimized for Persian voice transcription

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

### Speech-to-Text Transcription with Speaker Diarization
### تبدیل گفتار به متن همراه با تشخیص گوینده

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

* `num_speakers`: Number of speakers to identify (optional)  
  تعداد گویندگان برای تشخیص (اختیاری)

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
### قالب پاسخ

The API returns a JSON response with transcribed segments:
API پاسخی در قالب JSON با بخش‌های ترنویسی شده برمی‌گرداند:

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



