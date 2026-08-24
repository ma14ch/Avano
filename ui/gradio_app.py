"""Gradio client for the Speech-to-Text API.

The UI deliberately does not import the transcription pipeline. The API owns
the GPU models; this page only submits ``audio_file`` and ``num_speakers`` to
POST /api/inference/ and renders its ``segments`` response.
"""

import os
from pathlib import Path
from typing import Any

import gradio as gr
import httpx


API_URL = os.getenv("ASR_API_URL", "http://ai-tts:5016/api/inference/")
REQUEST_TIMEOUT_SECONDS = float(os.getenv("ASR_API_TIMEOUT", "600"))


def _format_segments(payload: dict[str, Any]) -> tuple[str, list[list[Any]]]:
    """Transform the API's ``{"segments": [...]}`` response for the UI."""
    segments = payload.get("segments")
    if not isinstance(segments, list):
        raise ValueError("The API response does not contain a segments list.")

    transcript_lines: list[str] = []
    rows: list[list[Any]] = []
    for segment in segments:
        if not isinstance(segment, dict):
            continue

        speaker = str(segment.get("speaker", "Unknown"))
        start = float(segment.get("start", 0.0))
        end = float(segment.get("end", 0.0))
        text = str(segment.get("transcription", "")).strip()
        transcript_lines.append(f"{speaker} [{start:.2f}–{end:.2f}s]: {text}")
        rows.append([speaker, round(start, 2), round(end, 2), text])

    return "\n".join(transcript_lines) or "No speech segments were detected.", rows


def _error_message(response: httpx.Response) -> str:
    """Return a useful API error without assuming a particular error shape."""
    try:
        detail = response.json().get("detail")
    except (ValueError, AttributeError):
        detail = response.text.strip()
    return str(detail or response.reason_phrase)


def transcribe(audio_path: str | None, num_speakers: int | float | None):
    """Submit the Gradio audio file according to the FastAPI OpenAPI schema."""
    if not audio_path:
        return "", [], "Please record or upload an audio file first."

    form_data: dict[str, str] = {}
    speaker_count = int(num_speakers or 0)
    if speaker_count > 0:
        form_data["num_speakers"] = str(speaker_count)

    try:
        with Path(audio_path).open("rb") as audio_file:
            files = {
                "audio_file": (
                    Path(audio_path).name,
                    audio_file,
                    "audio/wav",
                )
            }
            with httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS) as client:
                response = client.post(API_URL, data=form_data, files=files)

        if response.is_error:
            return "", [], f"API error ({response.status_code}): {_error_message(response)}"

        transcript, rows = _format_segments(response.json())
        return transcript, rows, "Transcription complete."
    except (OSError, httpx.HTTPError, ValueError) as exc:
        return "", [], f"Request failed: {exc}"


def clear_form():
    return None, 0, "", [], "Ready. Record or upload an audio file."


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Persian Multi-Speaker Transcription") as demo:
        gr.Markdown(
            "# Persian Multi-Speaker Transcription\n"
            "Record in the browser or upload audio. The page submits the API's "
            "required `audio_file` field and optionally sends `num_speakers`."
        )

        with gr.Row():
            with gr.Column(scale=1):
                audio = gr.Audio(
                    sources=["microphone", "upload"],
                    type="filepath",
                    format="wav",
                    label="Audio",
                )
                num_speakers = gr.Slider(
                    minimum=0,
                    maximum=10,
                    step=1,
                    value=0,
                    label="Number of speakers (0 = detect automatically)",
                )
                with gr.Row():
                    transcribe_button = gr.Button("Transcribe", variant="primary")
                    clear_button = gr.Button("Clear")

            with gr.Column(scale=2):
                status = gr.Markdown("Ready. Record or upload an audio file.")
                transcript = gr.Textbox(label="Transcript", lines=12, interactive=False)
                segments = gr.Dataframe(
                    headers=["Speaker", "Start (s)", "End (s)", "Transcription"],
                    datatype=["str", "number", "number", "str"],
                    type="array",
                    interactive=False,
                    label="Speaker segments",
                    wrap=True,
                )

        transcribe_button.click(
            fn=transcribe,
            inputs=[audio, num_speakers],
            outputs=[transcript, segments, status],
            concurrency_limit=1,
            trigger_mode="once",
        )
        clear_button.click(
            fn=clear_form,
            outputs=[audio, num_speakers, transcript, segments, status],
            queue=False,
        )

    return demo.queue(max_size=8, default_concurrency_limit=1)


if __name__ == "__main__":
    build_ui().launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
        show_error=True,
        share=True,
    )
