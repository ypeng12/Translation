import os
import threading

import numpy as np

from .config import Config, validate_language

# This project only uses PyTorch; avoid importing unrelated global TF/Flax installs.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")


class ModelStore:
    """Lazy local inference. A lock protects models shared by UI sessions."""

    def __init__(self, config: Config):
        self.config = config
        self.lock = threading.RLock()
        self._asr = None
        self._translations = {}
        self._vad = None

    @property
    def device(self):
        import torch
        if self.config.device != "auto":
            return self.config.device
        if torch.cuda.is_available():
            return "cuda"
        # CPU is the predictable baseline; MPS can be selected explicitly.
        return "cpu"

    def load_asr(self):
        with self.lock:
            if self._asr is None:
                from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
                kwargs = {"local_files_only": self.config.local_files_only}
                processor = AutoProcessor.from_pretrained(self.config.asr_model, **kwargs)
                model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    self.config.asr_model, **kwargs).to(self.device).eval()
                self._asr = processor, model
            return self._asr

    def transcribe(self, audio: np.ndarray) -> str:
        import torch
        with self.lock, torch.inference_mode():
            processor, model = self.load_asr()
            inputs = processor(audio, sampling_rate=self.config.sample_rate,
                               return_tensors="pt", return_attention_mask=True).to(self.device)
            ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)
            return processor.batch_decode(ids, skip_special_tokens=True)[0].strip()

    def load_translation(self, language: str):
        validate_language(language)
        with self.lock:
            if language not in self._translations:
                from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
                name = self.config.translation_models[language]
                kwargs = {"local_files_only": self.config.local_files_only}
                tokenizer = AutoTokenizer.from_pretrained(name, **kwargs)
                model = AutoModelForSeq2SeqLM.from_pretrained(name, **kwargs).to(self.device).eval()
                self._translations[language] = tokenizer, model
            return self._translations[language]

    def translate(self, text: str, language: str) -> str:
        import torch
        validate_language(language)
        if not text.strip():
            return ""
        with self.lock, torch.inference_mode():
            tokenizer, model = self.load_translation(language)
            inputs = tokenizer(text, return_tensors="pt").to(self.device)
            limit = min(tokenizer.model_max_length, 512)
            if inputs["input_ids"].shape[-1] > limit:
                raise ValueError(f"Text exceeds {limit} tokens; split into smaller chunks")
            ids = model.generate(**inputs, max_new_tokens=512, num_beams=4, do_sample=False)
            return tokenizer.batch_decode(ids, skip_special_tokens=True)[0].strip()

    def timestamps(self, audio: np.ndarray) -> list[dict]:
        import torch
        from silero_vad import get_speech_timestamps, load_silero_vad
        with self.lock:
            if self._vad is None:
                self._vad = load_silero_vad()
            return get_speech_timestamps(torch.from_numpy(audio), self._vad,
                                         sampling_rate=self.config.sample_rate,
                                         min_silence_duration_ms=200,
                                         speech_pad_ms=0)

    def warmup(self, languages):
        self.load_asr()
        for language in languages:
            self.load_translation(language)
        self.timestamps(np.zeros(16000, dtype=np.float32))
