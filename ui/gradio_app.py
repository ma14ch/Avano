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
from pydub import AudioSegment

API_URL = os.getenv("ASR_API_URL", "http://ai-tts:5016/api/inference/")
# Large meeting recordings can take a long time to diarize + transcribe
# segment-by-segment, so the default read timeout is generous (1 hour).
# Override with ASR_API_TIMEOUT (seconds) if needed.
REQUEST_TIMEOUT_SECONDS = float(os.getenv("ASR_API_TIMEOUT", "3600"))
HEALTH_TIMEOUT_SECONDS = float(os.getenv("ASR_HEALTH_TIMEOUT", "5"))

# Derive the API's base URL (e.g. "http://ai-tts:5016") from ASR_API_URL so we
# can also hit the root "/" and "/debug/models" endpoints for health checks,
# and the streaming inference endpoint.
_API_BASE_URL = API_URL.split("/api/")[0].rstrip("/")
_MODELS_STATUS_URL = f"{_API_BASE_URL}/debug/models"
_ROOT_URL = f"{_API_BASE_URL}/"
STREAM_API_URL = API_URL.rstrip("/") + "/stream"
# Streaming requests may sit idle between segments while the model is busy on
# a long segment, so give the read timeout the same generous budget as normal
# requests; only the connection itself needs a short timeout.
_STREAM_TIMEOUT = httpx.Timeout(connect=10.0, read=REQUEST_TIMEOUT_SECONDS, write=60.0, pool=10.0)


@dataclass
class TranscriptionState:
    """Holds the last successful result so export buttons can reuse it
    without re-calling the API."""

    segments: list[dict[str, Any]] = field(default_factory=list)
    source_filename: str = "transcript"
    audio_path: str | None = None


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


def _clean_segment(segment: dict[str, Any]) -> dict[str, Any]:
    """Normalize one raw API segment dict into the shape the UI uses."""
    return {
        "speaker": str(segment.get("speaker", "Unknown")),
        "start": float(segment.get("start", 0.0)),
        "end": float(segment.get("end", 0.0)),
        "transcription": str(segment.get("transcription", "")).strip(),
    }


def _render_segments(clean_segments: list[dict[str, Any]]) -> tuple[str, list[list[Any]]]:
    """Build the transcript text and Dataframe rows for a list of already
    cleaned/sorted segments."""
    transcript_lines = [
        f"{seg['speaker']} [{seg['start']:.2f}–{seg['end']:.2f}s]: {seg['transcription']}"
        for seg in clean_segments
    ]
    rows = [
        [seg["speaker"], round(seg["start"], 2), round(seg["end"], 2), round(seg["end"] - seg["start"], 2), seg["transcription"]]
        for seg in clean_segments
    ]
    transcript = "\n".join(transcript_lines) or "No speech segments were detected."
    return transcript, rows


def _format_segments(payload: dict[str, Any]) -> tuple[str, list[list[Any]], list[dict[str, Any]]]:
    """Transform the API's ``{"segments": [...]}`` response for the UI."""
    segments = payload.get("segments")
    if not isinstance(segments, list):
        raise ValueError("The API response does not contain a segments list.")

    clean_segments = [_clean_segment(segment) for segment in segments if isinstance(segment, dict)]
    clean_segments.sort(key=lambda seg: seg["start"])

    transcript, rows = _render_segments(clean_segments)
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


def play_segment(state: TranscriptionState, evt: gr.SelectData) -> tuple[str | None, str]:
    """Slice the source recording to the selected segment's [start, end] and
    return it for playback, so operators can listen to what was transcribed."""
    if not state.segments or not state.audio_path:
        return None, "⚠️ No audio available — run a transcription first."

    row_index = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
    if row_index is None or not (0 <= row_index < len(state.segments)):
        return None, "⚠️ Could not resolve the selected segment."

    seg = state.segments[row_index]
    try:
        audio = AudioSegment.from_file(state.audio_path)
        clip = audio[int(seg["start"] * 1000): int(seg["end"] * 1000)]
        clip_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        clip.export(clip_path, format="wav")
    except (OSError, ValueError) as exc:
        return None, f"🔴 Could not extract segment audio: {exc}"

    label = f"▶️ Playing {seg['speaker']} [{seg['start']:.2f}–{seg['end']:.2f}s]"
    return clip_path, label


# --------------------------------------------------------------------------
# Main actions
# --------------------------------------------------------------------------

def transcribe(
    audio_path: str | None,
    num_speakers: int | float | None,
    progress: gr.Progress = gr.Progress(track_tqdm=False),
):
    """Submit the Gradio audio file to the streaming inference endpoint and
    progressively yield transcript/segment updates as each diarized segment
    comes back, instead of blocking until the whole (possibly long)
    recording has finished processing.
    """
    empty_state = TranscriptionState()
    if not audio_path:
        yield "", [], "⚠️ Please record or upload an audio file first.", empty_state, gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False)
        return

    form_data: dict[str, str] = {}
    speaker_count = int(num_speakers or 0)
    if speaker_count > 0:
        form_data["num_speakers"] = str(speaker_count)

    progress(0, desc="Uploading audio and starting transcription…")

    clean_segments: list[dict[str, Any]] = []
    state = TranscriptionState(source_filename=Path(audio_path).name, audio_path=audio_path)

    try:
        with Path(audio_path).open("rb") as audio_file:
            files = {
                "audio_file": (
                    Path(audio_path).name,
                    audio_file,
                    "audio/wav",
                )
            }
            with httpx.Client(timeout=_STREAM_TIMEOUT) as client:
                with client.stream("POST", STREAM_API_URL, data=form_data, files=files) as response:
                    if response.is_error:
                        response.read()
                        message = f"🔴 API error ({response.status_code}): {_error_message(response)}"
                        yield "", [], message, empty_state, gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False)
                        return

                    for line in response.iter_lines():
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            event = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        if "error" in event:
                            message = f"🔴 API error while streaming: {event['error']}"
                            yield "", [], message, empty_state, gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False)
                            return

                        if event.get("done"):
                            break

                        raw_segment = event.get("segment")
                        if not isinstance(raw_segment, dict):
                            continue

                        clean_segments.append(_clean_segment(raw_segment))
                        clean_segments.sort(key=lambda seg: seg["start"])
                        state = TranscriptionState(
                            segments=clean_segments,
                            source_filename=Path(audio_path).name,
                            audio_path=audio_path,
                        )
                        transcript, rows = _render_segments(clean_segments)
                        speaker_count_found = len({seg["speaker"] for seg in clean_segments})
                        status = f"⏳ Transcribing… {len(clean_segments)} segment(s) so far. Download is available for the results received so far."
                        yield transcript, rows, status, state, gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True)
    except httpx.TimeoutException:
        message = (
            f"⏱️ Request timed out after {REQUEST_TIMEOUT_SECONDS:.0f}s. "
            "The recording may be too long, or the API is overloaded."
        )
        yield "", [], message, empty_state, gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False)
        return
    except httpx.HTTPError as exc:
        yield "", [], f"🔴 Could not reach the API: {exc}", empty_state, gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False)
        return
    except OSError as exc:
        yield "", [], f"🔴 Could not read the audio file: {exc}", empty_state, gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False)
        return

    transcript, rows = _render_segments(clean_segments)
    speaker_count_found = len({seg["speaker"] for seg in clean_segments})
    status = f"✅ Done — {len(clean_segments)} segment(s), {speaker_count_found} speaker(s) detected."
    has_segments = bool(clean_segments)
    yield transcript, rows, status, state, gr.update(interactive=has_segments), gr.update(interactive=has_segments), gr.update(interactive=has_segments)


def clear_form():
    return (
        None,
        0,
        "",
        [],
        "Ready. Record or upload an audio file.",
        TranscriptionState(),
        gr.update(interactive=False),
        None,
        "No segment selected.",
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
                gr.Markdown("### Segment playback\nClick a row above to hear that speaker's segment.")
                segment_player_label = gr.Markdown("No segment selected.")
                segment_player = gr.Audio(label="Segment audio", interactive=False)

        demo.load(fn=check_api_health, outputs=health_banner)
        refresh_health_button.click(fn=check_api_health, outputs=health_banner, queue=False)

        transcribe_button.click(
            fn=transcribe,
            inputs=[audio, num_speakers],
            outputs=[transcript, segments, status, transcript_state, export_txt_button, export_srt_button, export_json_button],
            concurrency_limit=1,
            trigger_mode="once",
        )

        clear_button.click(
            fn=clear_form,
            outputs=[audio, num_speakers, transcript, segments, status, transcript_state, export_txt_button, segment_player, segment_player_label],
            queue=False,
        ).then(
            fn=lambda: (gr.update(interactive=False), gr.update(interactive=False)),
            outputs=[export_srt_button, export_json_button],
            queue=False,
        )

        export_txt_button.click(fn=export_txt, inputs=transcript_state, outputs=export_file, queue=False)
        export_srt_button.click(fn=export_srt, inputs=transcript_state, outputs=export_file, queue=False)
        export_json_button.click(fn=export_json, inputs=transcript_state, outputs=export_file, queue=False)

        segments.select(
            fn=play_segment,
            inputs=transcript_state,
            outputs=[segment_player, segment_player_label],
            queue=False,
        )

    return demo.queue(max_size=8, default_concurrency_limit=1)


if __name__ == "__main__":
    build_ui().launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
        show_error=True,
        share=True,
        css=CUSTOM_CSS,
    )

