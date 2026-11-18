import io

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from speech_translate.pipeline import TranslationPipeline
from speech_translate.server import MAX_DECODED_SAMPLES, create_app


class DemoModels:
    def timestamps(self, audio):
        indices = np.flatnonzero(np.abs(audio) > .01)
        return [{"start": int(indices[0]), "end": int(indices[-1] + 1)}] if indices.size else []

    def transcribe(self, audio):
        return "Hello."

    def translate(self, text, language):
        return f"{language}:{text}"


def client():
    return TestClient(create_app(TranslationPipeline(DemoModels())))


def test_website_health_and_text_api():
    with client() as api:
        assert api.get("/").status_code == 200
        assert api.get("/assets/recorder.js").status_code == 200
        assert api.get("/api/health").json()["languages"]["es"] == "Spanish"
        response = api.post("/api/translate/text", json={"text": "um hello", "language": "fr"})
        assert response.json()["translation"] == "fr:Hello."
        assert api.post("/api/translate/text", json={"text": "hi", "language": "xx"}).status_code == 400


def test_audio_api_validates_files_and_preserves_segments():
    wav = io.BytesIO()
    sf.write(wav, np.ones(16000) * .1, 16000, format="WAV")
    with client() as api:
        response = api.post("/api/translate/audio", files={"audio": ("demo.wav", wav.getvalue(), "audio/wav")},
                            data={"language": "de"})
        assert response.status_code == 200
        assert response.json()["english"] == "Hello."
        assert response.json()["segments"][0]["audio_seconds"] == 1
        response = api.post("/api/translate/audio", files={"audio": ("bad.wav", b"not audio")})
        assert response.status_code == 400


def test_websocket_flushes_final_fragment_once():
    with client() as api, api.websocket_connect("/api/live") as ws:
        ws.send_json({"language": "es", "sample_rate": 16000})
        assert ws.receive_json()["type"] == "ready"
        ws.send_bytes((np.ones(1600, dtype="<f4") * .1).tobytes())
        assert ws.receive_json()["type"] == "ack"
        ws.send_json({"type": "stop"})
        segment = ws.receive_json()
        assert segment["type"] == "segment"
        assert segment["translation"] == "es:Hello."
        assert segment["audio_seconds"] == .1
        assert ws.receive_json()["type"] == "done"


def test_websocket_pause_then_stop_does_not_duplicate():
    with client() as api, api.websocket_connect("/api/live") as ws:
        ws.send_json({"language": "es", "sample_rate": 16000})
        ws.receive_json()
        audio = np.concatenate([np.ones(8000) * .1, np.zeros(9600)]).astype("<f4")
        ws.send_bytes(audio.tobytes())
        assert ws.receive_json()["type"] == "segment"
        assert ws.receive_json()["type"] == "ack"
        ws.send_json({"type": "stop"})
        assert ws.receive_json()["type"] == "done"


@pytest.mark.parametrize("settings", [[], None, {"language": []},
                                     {"sample_rate": 16000.5}, {"clean": "false"}])
def test_websocket_rejects_invalid_settings(settings):
    with client() as api, api.websocket_connect("/api/live") as ws:
        ws.send_json(settings)
        response = ws.receive_json()
        assert response["type"] == "error"
        assert "inference failed" not in response["message"]
        assert ws.receive()["code"] == 1008


@pytest.mark.parametrize("control", ["[]", "null", "", '{"type":"unknown"}'])
def test_websocket_rejects_invalid_control(control):
    with client() as api, api.websocket_connect("/api/live") as ws:
        ws.send_json({})
        assert ws.receive_json()["type"] == "ready"
        ws.send_text(control)
        assert ws.receive_json()["type"] == "error"
        assert ws.receive()["code"] == 1008


def test_audio_rejects_large_decoded_payload_before_reading(monkeypatch):
    class OversizedAudio:
        samplerate = 96000
        frames = 120 * samplerate
        channels = 8

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self, **kwargs):
            pytest.fail("Oversized audio must be rejected before allocating samples")

    assert OversizedAudio.frames * OversizedAudio.channels > MAX_DECODED_SAMPLES
    monkeypatch.setattr(sf, "SoundFile", lambda _: OversizedAudio())
    with client() as api:
        response = api.post("/api/translate/audio", files={"audio": ("large.flac", b"compressed")})
        assert response.status_code == 400
        assert "too large" in response.json()["detail"]
