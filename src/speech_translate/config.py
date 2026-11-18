from dataclasses import dataclass, field

LANGUAGES = {"es": "Spanish", "fr": "French", "de": "German"}


@dataclass(frozen=True)
class Config:
    asr_model: str = "openai/whisper-base.en"
    translation_models: dict[str, str] = field(default_factory=lambda: {
        code: f"Helsinki-NLP/opus-mt-en-{code}" for code in LANGUAGES
    })
    device: str = "auto"
    sample_rate: int = 16000
    max_segment_seconds: float = 8.0
    pause_seconds: float = 0.6
    local_files_only: bool = False

    def __post_init__(self):
        if self.sample_rate != 16000:
            raise ValueError("Whisper requires a 16000 Hz pipeline sample rate")
        if not 0 < self.pause_seconds < self.max_segment_seconds <= 25:
            raise ValueError("Require 0 < pause_seconds < max_segment_seconds <= 25")


def validate_language(language: str) -> None:
    if language not in LANGUAGES:
        raise ValueError(f"Unsupported language {language!r}; choose es, fr or de")
