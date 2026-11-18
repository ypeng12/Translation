"""Website backend, independent of the Gradio demonstration UI."""
import asyncio
import io
from pathlib import Path

import numpy as np
from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .audio import normalize_audio
from .config import LANGUAGES, validate_language

WEB = Path(__file__).parent / "web"
MAX_UPLOAD = 32 * 1024 * 1024


class TextRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    language: str = "es"
    clean: bool = True


def create_app(pipeline):
    app = FastAPI(title="LiveTranslate API", version="0.1.0")
    app.mount("/assets", StaticFiles(directory=WEB), name="assets")
    # Limit active inference jobs. ModelStore also serializes model access.
    jobs = asyncio.Semaphore(2)

    async def infer(function, *args):
        async with jobs:
            return await asyncio.to_thread(function, *args)

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(WEB / "index.html")

    @app.get("/api/health")
    def health():
        return {"status": "ok", "languages": LANGUAGES,
                "models_loaded": getattr(pipeline.models, "_asr", None) is not None}

    @app.post("/api/translate/text")
    async def translate_text(request: TextRequest):
        try:
            validate_language(request.language)
            english = pipeline.processor.process(request.text) if request.clean else request.text
            translation = await infer(pipeline.models.translate, english, request.language)
            return {"english": english, "translation": translation, "language": request.language}
        except ValueError as error:
            raise HTTPException(400, str(error)) from error

    @app.post("/api/translate/audio")
    async def translate_audio(audio: UploadFile = File(...), language: str = Form("es"),
                              strategy: str = Form("pause"), clean: bool = Form(True)):
        import soundfile as sf

        try:
            validate_language(language)
            if strategy not in {"pause", "fixed"}:
                raise ValueError("Strategy must be pause or fixed")
            content = await audio.read(MAX_UPLOAD + 1)
            if len(content) > MAX_UPLOAD:
                raise HTTPException(413, "Audio file exceeds 32 MB")
            try:
                with sf.SoundFile(io.BytesIO(content)) as file:
                    duration = file.frames / file.samplerate
                    if duration > 120 or duration <= 0:
                        raise ValueError("Audio must be between 0 and 120 seconds")
                    samples, rate = file.read(dtype="float32"), file.samplerate
            except (sf.LibsndfileError, RuntimeError) as error:
                raise ValueError("Cannot decode this file. Please use WAV or FLAC.") from error
            results = await infer(pipeline.audio, samples, rate, language, clean, strategy)
            return {"english": "\n".join(r.english for r in results),
                    "translation": "\n".join(r.translation for r in results),
                    "language": language, "segments": [r.to_dict() for r in results]}
        except ValueError as error:
            raise HTTPException(400, str(error)) from error
        finally:
            await audio.close()

    @app.websocket("/api/live")
    async def live(socket: WebSocket):
        await socket.accept()
        try:
            # First message sets the session; subsequent binary frames are float32 mono PCM.
            settings = await asyncio.wait_for(socket.receive_json(), timeout=15)
            language = settings.get("language", "es")
            validate_language(language)
            rate = int(settings.get("sample_rate", 16000))
            if rate not in {8000, 16000, 22050, 24000, 32000, 44100, 48000, 96000}:
                raise ValueError("Unsupported sample rate")
            strategy = settings.get("strategy", "pause")
            clean = settings.get("clean", True)
            if not isinstance(clean, bool):
                raise TypeError("clean must be a boolean")
            buffer = pipeline.buffer(strategy)
            await socket.send_json({"type": "ready"})

            def consume(data, final):
                chunks = buffer.push(data, final=final)
                return [pipeline.segment(chunk, language, clean).to_dict() for chunk in chunks]

            while True:
                message = await socket.receive()
                if message["type"] == "websocket.disconnect":
                    return
                final = message.get("text") == '{"type":"stop"}'
                if message.get("text") and not final:
                    import json
                    final = json.loads(message["text"]).get("type") == "stop"
                    if not final:
                        raise ValueError("Expected a PCM frame or stop message")
                raw = message.get("bytes") or b""
                if len(raw) % 4 or len(raw) > rate * 4 * 2:
                    raise ValueError("PCM frames must contain at most 2 seconds of float32 mono audio")
                samples = np.frombuffer(raw, dtype="<f4").copy()
                normalized = normalize_audio(samples, rate, pipeline.config.sample_rate)
                results = await infer(consume, normalized, final)
                for result in results:
                    await socket.send_json({"type": "segment", **result})
                await socket.send_json({"type": "done" if final else "ack"})
                if final:
                    await socket.close()
                    return
        except WebSocketDisconnect:
            pass
        except (ValueError, TypeError, asyncio.TimeoutError) as error:
            await socket.send_json({"type": "error", "message": str(error) or "Session timed out"})
            await socket.close(code=1008)
        except Exception:
            # Errors stay visible to the website without exposing server internals.
            try:
                await socket.send_json({"type": "error", "message": "Model inference failed. Check the server terminal."})
                await socket.close(code=1011)
            except RuntimeError:
                pass
            raise

    return app
