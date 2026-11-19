"""Replay a real recording at microphone speed through the server's streaming pipeline."""

import argparse
import json
import platform
import re
from pathlib import Path
from time import perf_counter, sleep

import soundfile as sf
import torch
from jiwer import wer
from sacrebleu.metrics import CHRF

from speech_translate import audio
from speech_translate.config import Config
from speech_translate.evaluation import read_manifest
from speech_translate.models import ModelStore
from speech_translate.pipeline import TranslationPipeline


def replay(pipeline, samples, rate, language):
    frame_size = round(rate * .5)
    buffer = pipeline.buffer("pause")
    converter = getattr(audio, "StreamingResampler", None)
    resampler = converter(rate, pipeline.config.sample_rate) if converter else None
    captions = []
    compute = 0.
    start = perf_counter()

    def submit(frame, final=False):
        nonlocal compute
        tick = perf_counter()
        data = (resampler.push(frame, final=final) if resampler else
                audio.normalize_audio(frame, rate, pipeline.config.sample_rate))
        responses = [pipeline.segment(chunk, language) for chunk in buffer.push(data, final=final)]
        # The websocket sends captions when the current frame has finished processing.
        available = perf_counter()
        compute += available - tick
        for item in responses:
            if item.english:
                captions.append({"available_s": available - start, "raw_english": item.raw_english,
                                 "translation": item.translation})

    for offset in range(0, len(samples), frame_size):
        frame = samples[offset:offset + frame_size]
        capture_end = (offset + len(frame)) / rate
        sleep(max(0., start + capture_end - perf_counter()))
        submit(frame)
    submit(samples[:0], final=True)
    duration = len(samples) / rate
    return {"captions": captions, "audio_s": duration, "compute_s": compute, "rtf": compute / duration,
            "first_caption_s": captions[0]["available_s"] if captions else None,
            "last_caption_minus_capture_end_s": captions[-1]["available_s"] - duration if captions else None,
            "done_after_capture_end_s": perf_counter() - start - duration}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--id", default="fleurs-dev-1544")
    parser.add_argument("--label", required=True)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--language", choices=["es", "fr", "de"], default="es")
    args = parser.parse_args()
    records = {row["id"]: row for row in read_manifest(args.manifest)}
    row = records[args.id]
    samples, rate = sf.read(row["audio"], dtype="float32")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    config = Config(device="cpu", local_files_only=True)
    models = ModelStore(config)
    models.warmup([args.language])
    pipeline = TranslationPipeline(models, config)
    result = replay(pipeline, samples, rate, args.language)
    raw = " ".join(caption["raw_english"] for caption in result["captions"])
    translation = " ".join(caption["translation"] for caption in result["captions"])
    gold_words = re.findall(r"\b\w+\b", row["english"].lower())
    words = re.findall(r"\b\w+\b", raw.lower())
    result.update(label=args.label, id=args.id, language=args.language, platform=platform.platform(),
                  threads_requested=args.threads, threads_effective=torch.get_num_threads(),
                  frame_s=.5, wer=wer(" ".join(gold_words), " ".join(words)),
                  chrf_pp=CHRF(word_order=2).corpus_score([translation], [[row["references"][args.language]]]).score)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "captions"}, indent=2))


if __name__ == "__main__":
    main()
