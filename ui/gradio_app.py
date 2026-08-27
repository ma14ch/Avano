"""Gradio client for the Speech-to-Text API.

The UI deliberately does not import the transcription pipeline. The API owns
the GPU models; this page only submits ``audio_file`` and ``num_speakers`` to
POST /api/inference/ and renders its ``segments`` response.

Design notes (why this file looks the way it does):
 - Meeting transcription jobs can take minutes, so the UI must clearly show
   progress/state instead of looking frozen, and it must give actionable
   error messages when the API/network fails.
 - The transcript is Persian text, so the transcript box is rendered RTL.
 - Operators want the raw result in more than one shape (read on screen,
   paste somewhere, or hand to another tool), so plain text / SRT / JSON
   exports are generated locally from the last response - no extra API call.
 - A lightweight health check against the API is run on page load (and can be
   re-run manually) so operators immediately see whether the backend and its
   models are reachable before they upload a large meeting recording.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import gradio as gr
import httpx

API_URL = os.getenv("ASR_API_URL", "http://ai-tts:5016/api/inference/")
REQUEST_TIMEOUT_SECONDS = float(os.getenv("ASR_API_TIMEOUT", "600"))
HEALTH_TIMEOUT_SECONDS = float(os.getenv("ASR_HEALTH_TIMEOUT", "5"))

# Derive the API's base URL (e.g. "http://ai-tts:5016") from ASR_API_URL so we
# can also hit the root "/" and "/debug/models" endpoints for health checks.
_API_BASE_URL = API_URL.split("/api/")[0].rstrip("/")
_MODELS_STATUS_URL = f"{_API_BASE_URL}/debug/models"
_ROOT_URL = f"{_API_BASE_URL}/"


@dataclass
class TranscriptionState:
    """Holds the last successful result so export buttons can reuse it
    without re-calling the API."""

    segments: list[dict[str, Any]] = field(default_factory=list)
    source_filename: str = "transcript"


# --------------------------------------------------------------------------
# API helpers
# --------------------------------------------------------------------------

def check_api_health() -> str:
    """Ping the API root and model-status endpoints. Returns a status message
    for the health banner; never raises."""
    try:
        with httpx.Client(timeout=HEALTH_TIMEOUT_SECONDS) as client:
            root_response = client.get(_ROOT_URL)
            root_response.raise_for_status()

            models_response = client.get(_MODELS_STATUS_URL)
            models_response.raise_for_status()
            models = models_response.json().get("models", {})
    except httpx.HTTPError as exc:
        return f"🔴 API is unreachable at `{_API_BASE_URL}` ({exc})."

    loaded = [name for name, ok in models.items() if isinstance(ok, bool) and ok and name not in ("whisper_model", "whisper_processor")]
    device = models.get("device", "unknown")
    model_key = models.get("asr_model_key", "unknown")
    model_family = models.get("asr_model_family", "unknown")
    if models.get("whisper_model") and models.get("whisper_processor"):
        return (
            f"🟢 API online — model: **{model_key}** ({model_family}), device: **{device}**"
            + (f", also loaded: {', '.join(loaded)}" if loaded else "")
            + "."
        )
    return (
        f"🟡 API online but model '{model_key}' ({model_family}) is not loaded yet "
        f"(device: **{device}**). The first transcription request will trigger "
        "loading and may be slow."
    )


def _error_message(response: httpx.Response) -> str:
    """Return a useful API error without assuming a particular error shape."""
    try:
        detail = response.json().get("detail")
    except (ValueError, AttributeError):
        detail = response.text.strip()
    return str(detail or response.reason_phrase)


def _format_segments(payload: dict[str, Any]) -> tuple[str, list[list[Any]], list[dict[str, Any]]]:
    """Transform the API's ``{"segments": [...]}`` response for the UI."""
    segments = payload.get("segments")
    if not isinstance(segments, list):
        raise ValueError("The API response does not contain a segments list.")

    clean_segments: list[dict[str, Any]] = []
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        clean_segments.append(
            {
                "speaker": str(segment.get("speaker", "Unknown")),
                "start": float(segment.get("start", 0.0)),
                "end": float(segment.get("end", 0.0)),
                "transcription": str(segment.get("transcription", "")).strip(),
            }
        )
    clean_segments.sort(key=lambda seg: seg["start"])

    transcript_lines = [
        f"{seg['speaker']} [{seg['start']:.2f}–{seg['end']:.2f}s]: {seg['transcription']}"
        for seg in clean_segments
    ]
    rows = [
        [seg["speaker"], round(seg["start"], 2), round(seg["end"], 2), round(seg["end"] - seg["start"], 2), seg["transcription"]]
        for seg in clean_segments
    ]

    transcript = "\n".join(transcript_lines) or "No speech segments were detected."
    return transcript, rows, clean_segments


# --------------------------------------------------------------------------
# Export helpers
# --------------------------------------------------------------------------

def _seconds_to_srt_timestamp(seconds: float) -> str:
    seconds = max(0.0, seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    millis = int(round((secs - int(secs)) * 1000))
    return f"{int(hours):02d}:{int(minutes):02d}:{int(secs):02d},{millis:03d}"


def _write_temp_file(content: str, suffix: str) -> str:
    handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=suffix, delete=False
    )
    with handle:
        handle.write(content)
    return handle.name


def _export(state: TranscriptionState, kind: str) -> str | None:
    if not state.segments:
        raise gr.Error("There is no transcript yet — run a transcription first.")

    stem = Path(state.source_filename).stem or "transcript"

    if kind == "txt":
        lines = [
            f"{seg['speaker']} [{seg['start']:.2f}-{seg['end']:.2f}s]: {seg['transcription']}"
            for seg in state.segments
        ]
        return _write_temp_file("\n".join(lines), f"_{stem}.txt")

    if kind == "srt":
        blocks = []
        for index, seg in enumerate(state.segments, start=1):
            blocks.append(
                f"{index}\n"
                f"{_seconds_to_srt_timestamp(seg['start'])} --> {_seconds_to_srt_timestamp(seg['end'])}\n"
                f"{seg['speaker']}: {seg['transcription']}\n"
            )
        return _write_temp_file("\n".join(blocks), f"_{stem}.srt")

    if kind == "json":
        return _write_temp_file(
            json.dumps({"segments": state.segments}, ensure_ascii=False, indent=2),
            f"_{stem}.json",
        )

    raise ValueError(f"Unknown export kind: {kind}")


def export_txt(state: TranscriptionState) -> str:
    return _export(state, "txt")


def export_srt(state: TranscriptionState) -> str:
    return _export(state, "srt")


def export_json(state: TranscriptionState) -> str:
    return _export(state, "json")


# --------------------------------------------------------------------------
# Main actions
# --------------------------------------------------------------------------

def transcribe(
    audio_path: str | None,
    num_speakers: int | float | None,
    progress: gr.Progress = gr.Progress(track_tqdm=False),
):
    """Submit the Gradio audio file according to the FastAPI OpenAPI schema."""
    empty_state = TranscriptionState()
    if not audio_path:
        return "", [], "⚠️ Please record or upload an audio file first.", empty_state, gr.update(interactive=False)

    form_data: dict[str, str] = {}
    speaker_count = int(num_speakers or 0)
    if speaker_count > 0:
        form_data["num_speakers"] = str(speaker_count)

    progress(0, desc="Uploading audio and starting transcription…")
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
    except httpx.TimeoutException:
        message = (
            f"⏱️ Request timed out after {REQUEST_TIMEOUT_SECONDS:.0f}s. "
            "The recording may be too long, or the API is overloaded."
        )
        return "", [], message, empty_state, gr.update(interactive=False)
    except httpx.HTTPError as exc:
        return "", [], f"🔴 Could not reach the API: {exc}", empty_state, gr.update(interactive=False)
    except OSError as exc:
        return "", [], f"🔴 Could not read the audio file: {exc}", empty_state, gr.update(interactive=False)

    if response.is_error:
        message = f"🔴 API error ({response.status_code}): {_error_message(response)}"
        return "", [], message, empty_state, gr.update(interactive=False)

    progress(1, desc="Formatting results…")
    try:
        transcript, rows, clean_segments = _format_segments(response.json())
    except (ValueError, TypeError) as exc:
        return "", [], f"🔴 Unexpected API response: {exc}", empty_state, gr.update(interactive=False)

    state = TranscriptionState(
        segments=clean_segments,
        source_filename=Path(audio_path).name,
    )
    speaker_count_found = len({seg["speaker"] for seg in clean_segments})
    status = f"✅ Done — {len(clean_segments)} segment(s), {speaker_count_found} speaker(s) detected."
    return transcript, rows, status, state, gr.update(interactive=bool(clean_segments))


def clear_form():
    return (
        None,
        0,
        "",
        [],
        "Ready. Record or upload an audio file.",
        TranscriptionState(),
        gr.update(interactive=False),
    )


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------

CUSTOM_CSS = """
#transcript_box textarea { direction: rtl; text-align: right; font-size: 1.05rem; }
"""


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Persian Multi-Speaker Transcription") as demo:
        gr.Markdown(
            "# 🎙️ Persian Multi-Speaker Meeting Transcription\n"
            "Record in the browser or upload a meeting recording. The audio is sent to the "
            "transcription API, split by speaker, and transcribed segment by segment."
        )

        health_banner = gr.Markdown("Checking API status…")
        refresh_health_button = gr.Button("🔄 Recheck API status", size="sm")

        transcript_state = gr.State(TranscriptionState())

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

                gr.Markdown("### Export")
                with gr.Row():
                    export_txt_button = gr.Button("⬇ .txt", interactive=False)
                    export_srt_button = gr.Button("⬇ .srt", interactive=False)
                    export_json_button = gr.Button("⬇ .json", interactive=False)
                export_file = gr.File(label="Download", visible=True)

            with gr.Column(scale=2):
                status = gr.Markdown("Ready. Record or upload an audio file.")
                transcript = gr.Textbox(
                    label="Transcript",
                    lines=12,
                    interactive=False,
                    elem_id="transcript_box",
                )
                segments = gr.Dataframe(
                    headers=["Speaker", "Start (s)", "End (s)", "Duration (s)", "Transcription"],
                    datatype=["str", "number", "number", "number", "str"],
                    type="array",
                    interactive=False,
                    label="Speaker segments",
                    wrap=True,
                )

        demo.load(fn=check_api_health, outputs=health_banner)
        refresh_health_button.click(fn=check_api_health, outputs=health_banner, queue=False)

        transcribe_button.click(
            fn=transcribe,
            inputs=[audio, num_speakers],
            outputs=[transcript, segments, status, transcript_state, export_txt_button],
            concurrency_limit=1,
            trigger_mode="once",
        ).then(
            fn=lambda state: (gr.update(interactive=bool(state.segments)), gr.update(interactive=bool(state.segments))),
            inputs=transcript_state,
            outputs=[export_srt_button, export_json_button],
            queue=False,
        )

        clear_button.click(
            fn=clear_form,
            outputs=[audio, num_speakers, transcript, segments, status, transcript_state, export_txt_button],
            queue=False,
        ).then(
            fn=lambda: (gr.update(interactive=False), gr.update(interactive=False)),
            outputs=[export_srt_button, export_json_button],
            queue=False,
        )

        export_txt_button.click(fn=export_txt, inputs=transcript_state, outputs=export_file, queue=False)
        export_srt_button.click(fn=export_srt, inputs=transcript_state, outputs=export_file, queue=False)
        export_json_button.click(fn=export_json, inputs=transcript_state, outputs=export_file, queue=False)

    return demo.queue(max_size=8, default_concurrency_limit=1)


if __name__ == "__main__":
    build_ui().launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
        show_error=True,
        share=True,
        css=CUSTOM_CSS,
    )

