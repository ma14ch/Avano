import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr

# Ensure we can import from src when running from project root
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
	sys.path.insert(0, str(SRC))

# Import your processing pipeline
from processor import process_voice_file  # type: ignore

def _to_text_and_df(result: Dict[str, Any]) -> Tuple[str, Optional["pd.DataFrame"]]:
	segments: List[Dict[str, Any]] = result.get("segments", []) if isinstance(result, dict) else []
	lines: List[str] = []
	for seg in segments:
		spk = seg.get("speaker", "")
		start = seg.get("start", 0.0)
		end = seg.get("end", 0.0)
		txt = seg.get("transcription", "")
		lines.append(f"{spk} [{start:.2f}-{end:.2f}]: {txt}")
	full_text = "\n".join(lines)

	df = None
	if segments:
		try:
			import pandas as pd  # local import to keep optional
			rows = []
			for seg in segments:
				rows.append(
					{
						"speaker": seg.get("speaker", ""),
						"start": seg.get("start", 0.0),
						"end": seg.get("end", 0.0),
						"transcription": seg.get("transcription", ""),
					}
				)
			df = pd.DataFrame(rows, columns=["speaker", "start", "end", "transcription"])
		except Exception:
			df = None
	return full_text, df

def infer(audio_path: Optional[str], num_speakers: Optional[int]):
	if not audio_path:
		return "", None
	try:
		res = process_voice_file(audio_path, num_speakers=num_speakers or None)
		text, df = _to_text_and_df(res)
		return text, df if df is not None else "Install pandas to view the segments table."
	except Exception as e:
		return f"Error: {e}", None

def build_ui() -> gr.Blocks:
	with gr.Blocks(title="Persian Transcription UI") as demo:
		gr.Markdown("## Multi-Speaker Persian Voice Transcription")
		with gr.Row():
			audio = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Audio")
		with gr.Row():
			num_speakers = gr.Slider(minimum=0, maximum=10, step=1, value=0, label="Limit number of speakers (0 = auto)")
		run_btn = gr.Button("Transcribe")

		with gr.Row():
			txt_out = gr.Textbox(lines=14, label="Transcript")
		try:
			# Prefer Dataframe if pandas is installed
			import pandas as pd  # noqa: F401
			segments_out = gr.Dataframe(headers=["speaker", "start", "end", "transcription"], label="Segments")
		except Exception:
			segments_out = gr.Textbox(label="Segments (pandas not installed)")

		run_btn.click(fn=infer, inputs=[audio, num_speakers], outputs=[txt_out, segments_out])
		gr.Examples(examples=[], inputs=[audio, num_speakers], label="Examples")
	return demo

if __name__ == "__main__":
	demo = build_ui()
	demo.queue(max_size=8)  # removed concurrency_count for Gradio 4.x
	demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")), show_error=True)