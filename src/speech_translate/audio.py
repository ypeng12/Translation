from math import gcd
from typing import Protocol

import numpy as np
from scipy.signal import resample_poly


def normalize_audio(samples, sample_rate: int, target_rate: int = 16000) -> np.ndarray:
    """Accept soundfile float audio or Gradio integer/float audio."""
    if sample_rate <= 0 or target_rate <= 0:
        raise ValueError("Sample rate must be positive")
    audio = np.asarray(samples)
    if audio.ndim not in (1, 2):
        raise ValueError("Expected mono or channels-last stereo audio")
    if np.issubdtype(audio.dtype, np.integer):
        info = np.iinfo(audio.dtype)
        midpoint = (info.max + 1) / 2 if info.min == 0 else 0
        audio = (audio.astype(np.float32) - midpoint) / (midpoint or -info.min)
    else:
        audio = audio.astype(np.float32)
    if audio.ndim == 2:
        if audio.shape[1] == 0:
            raise ValueError("Audio must contain at least one channel")
        audio = audio.mean(axis=1)
    if not np.isfinite(audio).all():
        raise ValueError("Audio contains NaN or infinity")
    if sample_rate != target_rate and audio.size:
        divisor = gcd(sample_rate, target_rate)
        audio = resample_poly(audio, target_rate // divisor, sample_rate // divisor)
    return np.ascontiguousarray(np.clip(audio, -1, 1), dtype=np.float32)


class SpeechDetector(Protocol):
    def timestamps(self, audio: np.ndarray) -> list[dict]: ...


class AudioBuffer:
    """Per-session buffer; bounded segments, pause cuts, explicit final flush."""

    def __init__(self, detector: SpeechDetector, sample_rate=16000,
                 max_seconds=8.0, pause_seconds=0.6, strategy="pause"):
        if strategy not in {"fixed", "pause"}:
            raise ValueError("Audio strategy must be fixed or pause")
        if sample_rate <= 0 or not 0 < pause_seconds < max_seconds <= 25:
            raise ValueError("Require positive sample rate and 0 < pause_seconds < max_seconds <= 25")
        self.detector = detector
        self.sample_rate = sample_rate
        self.max_samples = int(max_seconds * sample_rate)
        self.pause_samples = int(pause_seconds * sample_rate)
        self.strategy = strategy
        self.pending = np.empty(0, dtype=np.float32)

    def push(self, audio: np.ndarray, final=False) -> list[np.ndarray]:
        self.pending = np.concatenate([self.pending, audio])
        output = []
        while self.pending.size:
            cut = min(self.pending.size, self.max_samples)
            candidate = self.pending[:cut]
            ready = final or self.pending.size >= self.max_samples
            # Fixed windows need only one VAD pass when the window is complete.
            if self.strategy == "fixed" and not ready:
                break
            spans = self.detector.timestamps(candidate)
            if self.strategy == "pause" and spans:
                # A pause may be followed by new speech within the same input frame.
                for left, right in zip(spans, spans[1:]):
                    if right["start"] - left["end"] >= self.pause_samples:
                        cut = left["end"] + self.pause_samples
                        candidate = self.pending[:cut]
                        ready = True
                        break
                end = spans[-1]["end"]
                if cut - end >= self.pause_samples:
                    ready = True
            if not ready:
                break
            if spans:
                output.append(candidate.copy())
            self.pending = self.pending[cut:]
        return output
