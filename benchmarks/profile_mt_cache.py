"""Measure the OPUS-MT decoder cache on the same English inputs and CPU."""

import argparse
import json
import platform
from importlib.metadata import version
from pathlib import Path
from statistics import median
from time import perf_counter

import torch

from speech_translate.config import Config
from speech_translate.models import ModelStore


def profile(manifest, language, repeats, threads, offline):
    records = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    if not records or repeats < 2 or repeats % 2 or threads < 1:
        raise ValueError("Supply at least one recording, an even repeat count >= 2 and positive threads.")
    torch.set_num_threads(threads)
    config = Config(device="cpu", local_files_only=offline)
    tokenizer, model = ModelStore(config).load_translation(language)
    inputs = [tokenizer(row["english"], return_tensors="pt") for row in records]

    def generate(batch, cached):
        return model.generate(**batch, num_beams=4, do_sample=False, max_new_tokens=512,
                              use_cache=cached)

    passes = {"on": [], "off": []}
    reference_ids = {}
    identical = True
    with torch.inference_mode():
        for cached in (False, True):
            generate(inputs[0], cached)
        for run in range(repeats):
            for cached in ((False, True) if run % 2 == 0 else (True, False)):
                seconds = 0.0
                for row, batch in zip(records, inputs):
                    start = perf_counter()
                    output = generate(batch, cached)
                    seconds += perf_counter() - start
                    ids = output[0].tolist()
                    if reference_ids.setdefault(row["id"], ids) != ids:
                        identical = False
                passes["on" if cached else "off"].append(seconds)
    return {"model": config.translation_models[language], "revision": model.config._commit_hash,
            "platform": platform.platform(), "torch": str(torch.__version__),
            "transformers": version("transformers"), "threads": threads, "num_beams": 4,
            "input_ids": [row["id"] for row in records], "runs_seconds": passes,
            "median_seconds": {mode: median(times) for mode, times in passes.items()},
            "output_token_ids_equal_across_all_runs": identical}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--language", choices=["es", "fr", "de"], default="es")
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    result = profile(args.manifest, args.language, args.repeats, args.threads, args.offline)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["median_seconds"]), flush=True)
