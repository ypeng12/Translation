import json
from pathlib import Path
from time import perf_counter

from .config import validate_language


def read_manifest(path):
    """Require explicit alignment; never join FLEURS/FLORES by row number."""
    manifest = Path(path).resolve()
    records, seen = [], set()
    for line_number, line in enumerate(manifest.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not all(key in row for key in ("id", "audio", "english", "references")):
            raise ValueError(f"Line {line_number}: missing manifest fields")
        if row["id"] in seen:
            raise ValueError(f"Duplicate sample id: {row['id']}")
        seen.add(row["id"])
        if not isinstance(row["english"], str) or not row["english"].strip():
            raise ValueError(f"Line {line_number}: empty English reference")
        if not isinstance(row["references"], dict) or not row["references"]:
            raise ValueError(f"Line {line_number}: missing target references")
        for language, reference in row["references"].items():
            validate_language(language)
            if not isinstance(reference, str) or not reference.strip():
                raise ValueError(f"Line {line_number}: empty {language} reference")
        row["audio"] = str((manifest.parent / row["audio"]).resolve())
        if not Path(row["audio"]).is_file():
            raise ValueError(f"Missing audio: {row['audio']}")
        records.append(row)
    if not records:
        raise ValueError("Manifest is empty")
    return records


def accuracy_ratio(pipeline_score, text_score):
    return pipeline_score / text_score if text_score > 0 else None


def evaluate(pipeline, records, languages, strategy="pause", comet=False):
    import soundfile as sf
    from jiwer import wer
    from sacrebleu.metrics import BLEU, CHRF

    comet_model = None
    if comet:
        from comet import download_model, load_from_checkpoint
        comet_model = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da"))
    report = {"config": {"asr": pipeline.config.asr_model,
                         "translation": pipeline.config.translation_models,
                         "device": pipeline.models.device, "strategy": strategy},
              "languages": {}, "samples": []}
    for language in languages:
        validate_language(language)
        selected = [row for row in records if language in row["references"]]
        if not selected:
            raise ValueError(f"No human references for {language}")
        pipeline.models.warmup([language])
        gold, raw, processed, references, english, asr = [], [], [], [], [], []
        seconds, compute, asr_time, text_time, mt_time = 0., 0., 0., 0., 0.
        for row in selected:
            audio, rate = sf.read(row["audio"])
            start = perf_counter()
            results = pipeline.audio(audio, rate, language, strategy=strategy)
            elapsed = perf_counter() - start
            raw_text = " ".join(r.raw_english for r in results)
            cleaned = " ".join(r.english for r in results)
            # Both variants use exactly the same ASR segments.
            raw_translation = " ".join(pipeline.models.translate(r.raw_english, language)
                                       for r in results if r.raw_english)
            processed_translation = " ".join(r.translation for r in results)
            gold_translation = pipeline.models.translate(row["english"], language)
            gold.append(gold_translation)
            raw.append(raw_translation)
            processed.append(processed_translation)
            references.append(row["references"][language])
            english.append(row["english"])
            asr.append(raw_text)
            duration = len(audio) / rate
            seconds += duration
            compute += elapsed
            asr_time += sum(r.asr_seconds for r in results)
            text_time += sum(r.processing_seconds for r in results)
            mt_time += sum(r.translation_seconds for r in results)
            report["samples"].append({"id": row["id"], "language": language,
                                      "raw_english": raw_text, "english": cleaned,
                                      "raw_translation": raw_translation,
                                      "translation": processed_translation,
                                      "gold_text_translation": gold_translation,
                                      "reference": row["references"][language],
                                      "audio_seconds": duration,
                                      "pipeline_seconds": elapsed,
                                      "segments": [r.to_dict() for r in results]})
        scores = {}
        for name, hypotheses in [("gold_text", gold), ("raw_asr", raw), ("processed", processed)]:
            scores[name] = {"chrf_pp": CHRF(word_order=2).corpus_score(hypotheses, [references]).score,
                            "bleu": BLEU().corpus_score(hypotheses, [references]).score}
            if comet_model is not None:
                examples = [{"src": src, "mt": mt, "ref": ref}
                            for src, mt, ref in zip(english, hypotheses, references)]
                scores[name]["comet"] = float(comet_model.predict(
                    examples, batch_size=8, gpus=0).system_score)
        ratio = accuracy_ratio(scores["processed"]["chrf_pp"], scores["gold_text"]["chrf_pp"])
        # Case/punctuation-normalized WER, explicitly identified in the report.
        import re
        normalize = lambda s: " ".join(re.findall(r"\b\w+\b", s.lower()))
        report["languages"][language] = {
            "count": len(selected), "scores": scores,
            "accuracy_ratio_chrf_pp": ratio,
            "ratio_target_met": ratio is not None and ratio >= .8,
            "cleaning_delta_chrf_pp": scores["processed"]["chrf_pp"] - scores["raw_asr"]["chrf_pp"],
            "wer_normalized": wer([normalize(s) for s in english], [normalize(s) for s in asr]),
            "audio_seconds": seconds, "pipeline_seconds": compute,
            "rtf": compute / seconds if seconds else None,
            "asr_seconds": asr_time, "processing_seconds": text_time,
            "translation_seconds": mt_time,
            "mean_utterance_compute_seconds": compute / len(selected),
        }
    report["timing_note"] = ("Warm model weights; pipeline wall time includes resampling and VAD. "
                             "Excludes baseline translations, metric scoring and audio capture wait. "
                             "This is offline throughput, not end-to-end live subtitle latency.")
    return report
