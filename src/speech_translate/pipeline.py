from dataclasses import asdict, dataclass
from time import perf_counter

from .audio import AudioBuffer, normalize_audio
from .config import Config, validate_language
from .text import TextProcessor


@dataclass
class SegmentResult:
    raw_english: str
    english: str
    translation: str
    language: str
    audio_seconds: float
    asr_seconds: float
    processing_seconds: float
    translation_seconds: float

    @property
    def compute_seconds(self):
        return self.asr_seconds + self.processing_seconds + self.translation_seconds

    @property
    def rtf(self):
        return self.compute_seconds / self.audio_seconds if self.audio_seconds else None

    def to_dict(self):
        return {**asdict(self), "compute_seconds": self.compute_seconds, "rtf": self.rtf}


class TranslationPipeline:
    def __init__(self, models, config=None, processor=None):
        self.models = models
        self.config = config or Config()
        self.processor = processor or TextProcessor()

    def buffer(self, strategy="pause"):
        return AudioBuffer(self.models, self.config.sample_rate,
                           self.config.max_segment_seconds, self.config.pause_seconds, strategy)

    def segment(self, audio, language="es", clean=True):
        validate_language(language)
        start = perf_counter()
        raw = self.models.transcribe(audio)
        asr_end = perf_counter()
        english = self.processor.process(raw) if clean else raw
        text_end = perf_counter()
        translation = self.models.translate(english, language) if english else ""
        end = perf_counter()
        return SegmentResult(raw, english, translation, language,
                             len(audio) / self.config.sample_rate,
                             asr_end - start, text_end - asr_end, end - text_end)

    def audio(self, samples, sample_rate, language="es", clean=True, strategy="pause"):
        validate_language(language)
        audio = normalize_audio(samples, sample_rate, self.config.sample_rate)
        buffer = self.buffer(strategy)
        chunks = []
        step = self.config.sample_rate // 2
        for offset in range(0, len(audio), step):
            chunks.extend(buffer.push(audio[offset:offset + step]))
        chunks.extend(buffer.push(audio[:0], final=True))
        return [self.segment(chunk, language, clean) for chunk in chunks]
