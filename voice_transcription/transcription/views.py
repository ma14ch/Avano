# transcription/views.py
import os
import tempfile

from django.http import HttpResponseBadRequest, JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt

from .processor import process_voice_file


def index(request):
    """
    Renders an HTML form for file upload and allows choosing the number of speakers.
    """
    if request.method == "POST":
        uploaded_file = request.FILES.get("audio_file")
        num_speakers = request.POST.get("num_speakers")
        if not uploaded_file:
            return render(request, "transcription/index.html", {"error": "No file uploaded"})
        try:
            num_speakers = int(num_speakers) if num_speakers else None
        except ValueError:
            num_speakers = None

        # Save uploaded file to a temporary location
        temp_path = os.path.join(tempfile.gettempdir(), uploaded_file.name)
        with open(temp_path, "wb") as f:
            for chunk in uploaded_file.chunks():
                f.write(chunk)

        # Process the audio file
        result = process_voice_file(temp_path, num_speakers=num_speakers)
        os.remove(temp_path)
        return render(request, "transcription/result.html", {"result": result})
    return render(request, "transcription/index.html")


@csrf_exempt
def api_inference(request):
    """
    API endpoint for voice transcription.
    Expects a POST with an audio file and an optional 'num_speakers' parameter.
    Returns a JSON response with the transcription result.
    """
    if request.method != "POST":
        return JsonResponse({"error": "Only POST method is allowed"}, status=405)

    uploaded_file = request.FILES.get("audio_file")
    num_speakers = request.POST.get("num_speakers")
    if not uploaded_file:
        return JsonResponse({"error": "No file uploaded"}, status=400)

    try:
        num_speakers = int(num_speakers) if num_speakers else None
    except ValueError:
        num_speakers = None

    temp_path = os.path.join(tempfile.gettempdir(), uploaded_file.name)
    with open(temp_path, "wb") as f:
        for chunk in uploaded_file.chunks():
            f.write(chunk)

    result = process_voice_file(temp_path, num_speakers=num_speakers)
    os.remove(temp_path)
    return JsonResponse(result)
