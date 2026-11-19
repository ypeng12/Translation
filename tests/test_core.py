import json

import numpy as np
import pytest

from speech_translate.audio import (
    AudioBuffer,
    StreamingResampler,
    normalize_audio,
    recorded_chunks,
)
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


@pytest.mark.parametrize("rate", [8000, 16000, 22050, 24000, 32000, 44100, 48000, 96000])
def test_streaming_resampling_matches_uninterrupted_pcm_with_fragmented_frames(rate):
    # Non-periodic boundaries stress both FIR history and rational sample phase.
    source = np.random.default_rng(2718).uniform(-.25, .25, rate + 17).astype(np.float32)
    stream = StreamingResampler(rate)
    output = []
    for frame in np.array_split(source, [1, 89, 2741, 4003, 7027]):
        output.append(stream.push(frame))
    output.append(stream.push(source[:0], final=True))
    np.testing.assert_allclose(np.concatenate(output), normalize_audio(source, rate), atol=1e-6, rtol=0)
    assert not len(stream.push(source[:0], final=True))


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


def test_fixed_window_only_runs_vad_when_ready():
    class CountingDetector(FakeModels):
        calls = 0

        def timestamps(self, audio):
            self.calls += 1
            return super().timestamps(audio)

    detector = CountingDetector()
    buffer = AudioBuffer(detector, sample_rate=10, max_seconds=2, strategy="fixed")
    for _ in range(3):
        assert buffer.push(np.ones(5)) == []
    assert detector.calls == 0
    assert len(buffer.push(np.ones(5))) == 1
    assert detector.calls == 1


def test_pause_inside_frame_preserves_following_speech():
    class Detector:
        def timestamps(self, audio):
            mask = np.abs(audio) > .01
            edges = np.diff(np.r_[False, mask, False].astype(int))
            return [{"start": int(a), "end": int(b)}
                    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))]

    buffer = AudioBuffer(Detector(), sample_rate=10, max_seconds=4, pause_seconds=.6)
    audio = np.r_[np.ones(10), np.zeros(8), np.ones(5)]
    chunks = buffer.push(audio)
    assert [len(chunk) for chunk in chunks] == [16]
    chunks.extend(buffer.push(np.empty(0), final=True))
    np.testing.assert_array_equal(np.concatenate(chunks), audio)


def test_recording_scans_once_and_uses_context_beyond_live_limit():
    class Detector:
        calls = 0

        def timestamps(self, audio):
            self.calls += 1
            # A quiet opening/ending may fall below the detector threshold.
            return [{"start": 12, "end": 44}, {"start": 57, "end": 103}]

    detector = Detector()
    audio = np.arange(110)
    chunks = list(recorded_chunks(audio, detector, sample_rate=10, max_seconds=25, pause_seconds=.6))
    assert detector.calls == 1 and len(chunks) == 1
    np.testing.assert_array_equal(chunks[0], audio)


def test_recorded_window_prefers_a_breath_and_retains_all_speech():
    class Detector:
        def timestamps(self, audio):
            return [{"start": 0, "end": 77}, {"start": 81, "end": 209},
                    {"start": 240, "end": 250}]

    audio = np.arange(250)
    chunks = list(recorded_chunks(audio, Detector(), sample_rate=10, max_seconds=10, pause_seconds=.6))
    assert chunks[0][-1] == 78  # Cut in the short breath rather than mid-word at sample 100.
    assert all(0 < len(chunk) <= 100 for chunk in chunks)
    joined = np.concatenate(chunks)
    assert len(np.unique(joined)) == len(joined)
    assert set(np.r_[0:77, 81:209, 240:250]).issubset(joined)


def test_streaming_pause_prefers_a_breath_before_hard_limit():
    class Detector:
        def timestamps(self, audio):
            return [{"start": 0, "end": 29}, {"start": 31, "end": 40}] if len(audio) == 40 else [
                {"start": 0, "end": len(audio)}]

    buffer = AudioBuffer(Detector(), sample_rate=10, max_seconds=4, pause_seconds=.6)
    audio = np.arange(40)
    chunks = buffer.push(audio)
    assert len(chunks[0]) == 30
    chunks.extend(buffer.push(np.empty(0), final=True))
    np.testing.assert_array_equal(np.concatenate(chunks), audio)


def test_continuous_live_voicing_scans_recent_audio_and_keeps_onset():
    class Detector(FakeModels):
        def __init__(self):
            self.scanned = []

        def timestamps(self, audio):
            self.scanned.append(len(audio))
            return super().timestamps(audio)

    detector = Detector()
    buffer = AudioBuffer(detector, sample_rate=10, max_seconds=8, pause_seconds=.6)
    for _ in range(12):
        assert not buffer.push(np.ones(5))
    assert detector.scanned[-1] < len(buffer.pending) // 2
    chunks = buffer.push(np.zeros(6))
    assert len(chunks) == 1
    np.testing.assert_array_equal(chunks[0], np.r_[np.ones(60), np.zeros(6)])


def test_invalid_audio_configuration():
    with pytest.raises(ValueError):
        Config(sample_rate=8000)
    with pytest.raises(ValueError):
        AudioBuffer(FakeModels(), max_seconds=0)
    with pytest.raises(ValueError):
        normalize_audio(np.empty((2, 0)), 16000)
    with pytest.raises(ValueError):
        normalize_audio(np.ones(2), 16000, 0)


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
