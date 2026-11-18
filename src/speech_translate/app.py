from dataclasses import dataclass, field
from time import perf_counter

import numpy as np

from .audio import normalize_audio
from .config import LANGUAGES


@dataclass
class Session:
    # No model objects in Gradio State: each browser gets its own audio/text buffer.
    pending: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))
    rows: list = field(default_factory=list)
    settings: tuple | None = None


def build_app(pipeline):
    import gradio as gr

    def stream(audio, language, strategy, clean, state, final=False):
        state = state or Session()
        settings = (language, strategy, clean)
        if state.settings is not None and state.settings != settings:
            raise gr.Error("Stop recording and clear the session before changing settings.")
        state.settings = settings
        buffer = pipeline.buffer(strategy)
        buffer.pending = state.pending
        normalized = (normalize_audio(audio[1], audio[0]) if audio is not None
                      else np.empty(0, dtype=np.float32))
        start = perf_counter()
        chunks = buffer.push(normalized, final=final)
        state.pending = buffer.pending
        for chunk in chunks:
            result = pipeline.segment(chunk, language, clean)
            if result.english:
                state.rows.append(result.to_dict())
        english = "\n".join(r["english"] for r in state.rows)
        target = "\n".join(r["translation"] for r in state.rows)
        status = {"segments": len(state.rows), "pending_seconds": round(len(state.pending) / 16000, 2),
                  "last_callback_seconds": round(perf_counter() - start, 3),
                  "note": "First use loads weights; callback time excludes audio capture and queue wait."}
        if state.rows:
            status["last_segment"] = state.rows[-1]
        return english, target, status, state

    def flush(language, strategy, clean, state):
        return stream(None, language, strategy, clean, state, final=True)

    def file_run(audio, language, strategy, clean):
        if audio is None:
            raise gr.Error("Upload or record audio first.")
        results = pipeline.audio(audio[1], audio[0], language, clean, strategy)
        return ("\n".join(r.english for r in results),
                "\n".join(r.translation for r in results),
                [r.to_dict() for r in results])

    with gr.Blocks(title="Live Speech Translation") as app:
        gr.Markdown("# English → Spanish / French / German\n"
                    "Local Whisper + text rules + OPUS-MT. First use downloads model weights.")
        with gr.Row():
            language = gr.Dropdown(choices=[(v, k) for k, v in LANGUAGES.items()], value="es", label="Target language")
            strategy = gr.Radio(["pause", "fixed"], value="pause", label="Audio segmentation")
            clean = gr.Checkbox(value=True, label="Apply text processing")
        with gr.Tab("Live microphone"):
            gr.Markdown("Choose settings before recording. Stop recording to translate the final fragment.")
            microphone = gr.Audio(sources=["microphone"], type="numpy", streaming=True)
            state = gr.State(Session())
            with gr.Row():
                source = gr.Textbox(label="English subtitles", lines=8)
                target = gr.Textbox(label="Translated subtitles", lines=8)
            status = gr.JSON(label="Timing and raw transcript")
            reset = gr.Button("Clear session (after stopping)")
            microphone.stream(stream, [microphone, language, strategy, clean, state],
                              [source, target, status, state], stream_every=.5,
                              concurrency_limit=1, concurrency_id="models")
            microphone.stop_recording(flush, [language, strategy, clean, state],
                                      [source, target, status, state],
                                      concurrency_limit=1, concurrency_id="models")
            reset.click(lambda: ("", "", {}, Session()), outputs=[source, target, status, state],
                        concurrency_id="models", concurrency_limit=1)
        with gr.Tab("Audio file"):
            upload = gr.Audio(sources=["upload"], type="numpy")
            run = gr.Button("Transcribe and translate", variant="primary")
            file_source = gr.Textbox(label="English", lines=6)
            file_target = gr.Textbox(label="Translation", lines=6)
            details = gr.JSON(label="Segment results")
            run.click(file_run, [upload, language, strategy, clean],
                      [file_source, file_target, details], concurrency_limit=1, concurrency_id="models")
    return app.queue(default_concurrency_limit=1, max_size=16)
