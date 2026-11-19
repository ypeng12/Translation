from itertools import pairwise
from math import gcd
from typing import Protocol

import numpy as np
from scipy.signal import firwin, resample_poly


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


class StreamingResampler:
    """Continuous mono PCM conversion with the same polyphase filter as file input."""

    def __init__(self, source_rate: int, target_rate: int = 16000):
        if source_rate <= 0 or target_rate <= 0:
            raise ValueError("Sample rates must be positive")
        divisor = gcd(source_rate, target_rate)
        self.up = target_rate // divisor
        self.down = source_rate // divisor
        self.pending = np.empty(0, dtype=np.float32)
        self.emitted = 0
        if self.up != self.down:
            max_rate = max(self.up, self.down)
            half_length = 10 * max_rate
            self.filter = firwin(2 * half_length + 1, 1 / max_rate, window=("kaiser", 5)).astype(np.float32)
            self.context = (half_length + self.up - 1) // self.up + 1

    def push(self, frame: np.ndarray, *, final: bool = False) -> np.ndarray:
        frame = np.asarray(frame, dtype=np.float32)
        if frame.ndim != 1 or not np.isfinite(frame).all():
            raise ValueError("Expected finite mono PCM samples")
        if self.up == self.down:
            return np.clip(frame, -1, 1)
        self.pending = np.concatenate((self.pending, frame))
        # Withhold the few samples whose filter still needs future input.
        available = len(self.pending) if final else max(0, len(self.pending) - self.context)
        stop = (available * self.up + self.down - 1) // self.down
        if stop <= self.emitted:
            return np.empty(0, dtype=np.float32)
        converted = resample_poly(self.pending, self.up, self.down, window=self.filter)
        output = np.clip(converted[self.emitted:stop], -1, 1)
        # Drop whole input periods so subsequent blocks use exactly the same
        # fractional sample positions as a single conversion of the recording.
        drop = max(0, stop * self.down // self.up - self.context) // self.down * self.down
        self.pending = self.pending[drop:].copy()
        self.emitted = stop - (drop // self.down) * self.up
        return output


class SpeechDetector(Protocol):
    def timestamps(self, audio: np.ndarray) -> list[dict]: ...


def _latest_gap(spans, start, limit):
    """Prefer a breath in the latter half of the window to a cut through a word."""
    earliest = (start + limit) // 2
    for right, left in pairwise(reversed(spans)):
        midpoint = (left["end"] + right["start"]) // 2
        if start < midpoint <= limit and midpoint >= earliest and right["start"] > left["end"]:
            return midpoint
    return limit


def recorded_chunks(audio, detector, sample_rate, max_seconds, pause_seconds):
    """Use the full recording to preserve context and avoid replaying VAD per frame."""
    spans = detector.timestamps(audio) if audio.size else []
    if not spans:
        return
    limit = max(1, int(max_seconds * sample_rate))
    # A short recording already fits into one Whisper input. Keep its full
    # context, including quiet onsets and pauses between related clauses.
    if audio.size <= limit:
        yield audio
        return
    pause = max(1, int(pause_seconds * sample_rate))
    # A VAD boundary can be late for a quiet first word. Keep original recording
    # edges and allow context around internal gaps; trim only prolonged silence.
    edge_context = min(limit // 4, 2 * sample_rate)
    first = 0
    for index, span in enumerate(spans):
        if index + 1 < len(spans) and spans[index + 1]["start"] - span["end"] < pause:
            continue
        group = spans[first:index + 1]
        left = (spans[first - 1]["end"] + group[0]["start"]) // 2 if first else 0
        right = ((span["end"] + spans[index + 1]["start"]) // 2
                 if index + 1 < len(spans) else audio.size)
        start = max(left, group[0]["start"] - edge_context)
        end = min(right, span["end"] + edge_context)
        while start + limit < end:
            cut = _latest_gap(group, start, start + limit)
            yield audio[start:cut]
            start = cut
        yield audio[start:end]
        first = index + 1


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
        self.max_samples = max(1, int(max_seconds * sample_rate))
        self.pause_samples = max(1, int(pause_seconds * sample_rate))
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
            if not ready:
                # For a growing live utterance, use enough history to see a pause
                # crossing the new frame. Continuous voicing needs no full rescan.
                history = max(2 * self.pause_samples, self.sample_rate)
                tail_size = audio.size + history
                if candidate.size >= 2 * tail_size:
                    tail = self.detector.timestamps(candidate[-tail_size:])
                    if (len(tail) == 1 and tail[0]["start"] <= history - self.pause_samples
                            and tail_size - tail[0]["end"] < self.pause_samples):
                        break
            spans = self.detector.timestamps(candidate)
            if self.strategy == "pause" and spans:
                # A pause may be followed by new speech within the same input frame.
                for left, right in pairwise(spans):
                    if right["start"] - left["end"] >= self.pause_samples:
                        cut = left["end"] + self.pause_samples
                        candidate = self.pending[:cut]
                        ready = True
                        break
                end = spans[-1]["end"]
                if cut - end >= self.pause_samples:
                    ready = True
                if cut == self.max_samples and end == cut:
                    cut = _latest_gap(spans, 0, cut)
                    candidate = self.pending[:cut]
            if not ready:
                break
            if spans:
                output.append(candidate.copy())
            self.pending = self.pending[cut:]
        return output
