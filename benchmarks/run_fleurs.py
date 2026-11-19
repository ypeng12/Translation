"""Measure the English-to-Spanish upload path; score all targets on shared ASR."""
import argparse
import json
import platform
import re
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import soundfile as sf
import torch
from jiwer import wer
from sacrebleu.metrics import CHRF

from speech_translate.config import Config
from speech_translate.evaluation import read_manifest
from speech_translate.models import ModelStore
from speech_translate.pipeline import TranslationPipeline


def normalized(text):
    return " ".join(re.findall(r"\b\w+\b", text.lower()))


def run(args):
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    config = Config(device="cpu", local_files_only=args.offline)
    models = ModelStore(config)
    languages = ["es", "fr", "de"]
    models.warmup(languages)
    pipeline = TranslationPipeline(models, config)
    records = read_manifest(args.manifest)
    rows = []
    for record in records:
        samples, rate = sf.read(record["audio"], dtype="float32")
        start = perf_counter()
        segments = pipeline.audio(samples, rate, "es", strategy="pause")
        wall = perf_counter() - start
        joined = " ".join(segment.raw_english for segment in segments)
        outputs = {"es": " ".join(segment.translation for segment in segments)}
        for language in ("fr", "de"):
            outputs[language] = " ".join(models.translate(segment.english, language)
                                          for segment in segments if segment.english)
        gold = {language: models.translate(record["english"], language) for language in languages}
        rows.append({"id": record["id"], "asr": joined, "outputs": outputs, "gold_mt": gold,
                     "segment_count": len(segments), "audio_s": len(samples) / rate, "wall_s_es": wall,
                     "asr_s": sum(s.asr_seconds for s in segments),
                     "mt_s_es": sum(s.translation_seconds for s in segments)})
        print(f"{record['id']} {wall:.2f}s ({len(segments)} segments)", flush=True)
    chrf = CHRF(word_order=2)
    scores = {}
    for language in languages:
        refs = [[r["references"][language] for r in records]]
        speech = chrf.corpus_score([r["outputs"][language] for r in rows], refs).score
        text = chrf.corpus_score([r["gold_mt"][language] for r in rows], refs).score
        scores[language] = {"speech_chrf_pp": speech, "gold_text_chrf_pp": text,
                            "ratio": speech / text if text else None}
    duration = sum(r["audio_s"] for r in rows)
    seconds = sum(r["wall_s_es"] for r in rows)
    tokenizer = models.load_asr()[0].tokenizer
    whisper_wer = wer([tokenizer.normalize(r["english"]) for r in records],
                      [tokenizer.normalize(r["asr"]) for r in rows])
    result = {"label": args.label, "platform": platform.platform(), "torch": str(torch.__version__),
              "transformers": version("transformers"), "requested_threads": args.threads,
              "effective_threads": torch.get_num_threads(),
              "asr_model": config.asr_model, "mt_models": config.translation_models,
              "audio_s": duration, "wall_s_es": seconds, "rtf_es": seconds / duration,
              "wer": wer([normalized(r["english"]) for r in records],
                         [normalized(r["asr"]) for r in rows]),
              "wer_whisper_english": whisper_wer, "scores": scores, "samples": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("rtf_es", "wer", "scores")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--offline", action="store_true")
    run(parser.parse_args())
