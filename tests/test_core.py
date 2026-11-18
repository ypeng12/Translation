import json

import numpy as np
import pytest

from speech_translate.audio import AudioBuffer, normalize_audio
from speech_translate.config import Config
from speech_translate.evaluation import accuracy_ratio, read_manifest
from speech_translate.pipeline import TranslationPipeline
from speech_translate.text import SentenceBuffer, TextProcessor


class FakeModels:
    def timestamps(self, audio):
        indices = np.flatnonzero(np.abs(audio) > .01)
        return [{"start": int(indices[0]), "end": int(indices[-1] + 1)}] if indices.size else []

    def transcribe(self, audio):
        return "um i i can't go"

    def translate(self, text, language):
        return f"{language}:{text}"


def test_cleaning_preserves_meaning_and_names():
    clean = TextProcessor({"pie torch": "PyTorch"})
    assert clean.process("um i i can't use pie torch") == "I can't use PyTorch."
    assert clean.process("I know you know him") == "I know you know him."
    assert clean.process("it is very very good") == "It is very very good."
    assert clean.process("uh um") == ""
    assert clean.process("I don't have 2.5 dollars") == "I don't have 2.5 dollars."


def test_pcm_stereo_resampling_and_invalid_audio():
    audio = np.full((48000, 2), 16384, dtype=np.int16)
    normalized = normalize_audio(audio, 48000)
    assert normalized.shape == (16000,)
    assert abs(normalized[8000] - .5) < .01
    assert normalize_audio(np.array([128], dtype=np.uint8), 16000)[0] == 0
    with pytest.raises(ValueError):
        normalize_audio(np.array([np.nan]), 16000)


def test_pause_cut_and_final_flush_do_not_duplicate_or_drop_speech():
    buffer = AudioBuffer(FakeModels(), sample_rate=10, max_seconds=4, pause_seconds=.6)
    assert buffer.push(np.ones(10)) == []
    chunks = buffer.push(np.zeros(6))
    assert len(chunks) == 1
    assert len(chunks[0]) == 16
    assert buffer.push(np.ones(3)) == []
    final = buffer.push(np.empty(0), final=True)
    assert len(final) == 1 and len(final[0]) == 3
    assert buffer.push(np.empty(0), final=True) == []


def test_fixed_boundaries_and_silence():
    buffer = AudioBuffer(FakeModels(), sample_rate=10, max_seconds=2, strategy="fixed")
    chunks = buffer.push(np.ones(45), final=True)
    assert [len(chunk) for chunk in chunks] == [20, 20, 5]
    assert buffer.push(np.zeros(20), final=True) == []


def test_pipeline_raw_ablation_and_target_validation():
    pipeline = TranslationPipeline(FakeModels())
    samples = np.ones(16000, dtype=np.float32)
    processed = pipeline.audio(samples, 16000, "es")[0]
    raw = pipeline.audio(samples, 16000, "es", clean=False)[0]
    assert processed.english == "I can't go."
    assert raw.english == "um i i can't go"
    assert processed.translation == "es:I can't go."
    assert processed.rtf >= 0
    assert pipeline.audio(np.zeros(16000), 16000) == []
    with pytest.raises(ValueError):
        pipeline.audio(samples, 16000, "xx")


def test_punctuation_buffer_uses_original_punctuation():
    buffer = SentenceBuffer()
    assert buffer.push("I need") == []
    assert buffer.push("help. Can you") == ["I need help."]
    assert buffer.push("come", final=True) == ["Can you come"]


def test_manifest_explicit_alignment_and_duplicate_validation(tmp_path):
    (tmp_path / "sample.wav").touch()
    row = {"id": "aligned-1", "audio": "sample.wav", "english": "Hello.",
           "references": {"es": "Hola."}}
    manifest = tmp_path / "samples.jsonl"
    manifest.write_text(json.dumps(row) + "\n")
    assert read_manifest(manifest)[0]["audio"] == str(tmp_path / "sample.wav")
    manifest.write_text((json.dumps(row) + "\n") * 2)
    with pytest.raises(ValueError, match="Duplicate"):
        read_manifest(manifest)


def test_ratio_is_not_bleu_80_or_clamped():
    assert accuracy_ratio(40, 50) == .8
    assert accuracy_ratio(55, 50) == 1.1
    assert accuracy_ratio(10, 0) is None
    with pytest.raises(ValueError):
        Config(max_segment_seconds=31)


def test_evaluation_uses_same_asr_for_both_variants(tmp_path):
    import soundfile as sf

    from speech_translate.evaluation import evaluate

    class CountingModels(FakeModels):
        device = "cpu"
        calls = 0

        def warmup(self, languages):
            pass

        def transcribe(self, audio):
            self.calls += 1
            return super().transcribe(audio)

    path = tmp_path / "speech.wav"
    sf.write(path, np.ones(16000, dtype=np.float32) * .1, 16000)
    models = CountingModels()
    records = [{"id": "1", "audio": str(path), "english": "I can't go.",
                "references": {"es": "es:I can't go."}}]
    report = evaluate(TranslationPipeline(models), records, ["es"])
    assert models.calls == 1
    assert report["languages"]["es"]["accuracy_ratio_chrf_pp"] == 1.0
    assert report["samples"][0]["raw_english"] == "um i i can't go"
    assert report["languages"]["es"]["rtf"] >= 0
    json.dumps(report)
