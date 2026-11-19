"""Compare live captions and full-recording review on a running public Space."""

import argparse
import io
import json
import re
from math import gcd
from pathlib import Path

import httpx
import numpy as np
import soundfile as sf
from jiwer import process_words
from sacrebleu.metrics import CHRF
from scipy.signal import resample_poly
from websockets.sync.client import connect

from speech_translate.evaluation import read_manifest

RATE = 44100
OPTIONS = {"language": "es", "strategy": "pause", "clean": True}


def running_space(client, repo):
    response = client.get(f"https://huggingface.co/api/spaces/{repo}", headers={"Cache-Control": "no-cache"})
    response.raise_for_status()
    info = response.json()
    runtime = info["runtime"]
    if runtime["stage"] != "RUNNING" or not runtime.get("sha") or runtime["sha"] != info["sha"]:
        raise RuntimeError(f"Space is not serving its current revision: {runtime}")
    return info["host"], runtime["sha"]


def collect_until(socket, until, segments):
    while True:
        message = json.loads(socket.recv(timeout=180))
        if message["type"] == until:
            return
        if message["type"] != "segment":
            raise ValueError(f"Unexpected live response: {message}")
        segments.append(message)


def live_captions(host, samples):
    segments = []
    with connect(host.replace("https://", "wss://", 1) + "/api/live", open_timeout=30) as socket:
        socket.send(json.dumps({**OPTIONS, "sample_rate": RATE}))
        collect_until(socket, "ready", segments)
        for offset in range(0, len(samples), RATE // 2):
            socket.send(samples[offset:offset + RATE // 2].astype("<f4", copy=False).tobytes())
            collect_until(socket, "ack", segments)
        socket.send(json.dumps({"type": "stop"}))
        collect_until(socket, "done", segments)
    return segments


def review_captions(client, host, samples):
    # Match the browser's asymmetric float scaling and JS Math.round semantics.
    scaled = samples.astype(np.float64) * np.where(samples < 0, 32768, 32767)
    pcm = np.floor(scaled + .5).astype("int16")
    wav = io.BytesIO()
    sf.write(wav, pcm, RATE, format="WAV", subtype="PCM_16")
    response = client.post(host + "/api/translate/audio", data={**OPTIONS, "clean": "true"},
                           files={"audio": ("microphone.wav", wav.getvalue(), "audio/wav")})
    response.raise_for_status()
    return response.json()["segments"]


def summarize(segments):
    return {"segments": len(segments),
            "english": " ".join(segment["english"].strip() for segment in segments),
            "spanish": " ".join(segment["translation"].strip() for segment in segments)}


def words(text):
    return " ".join(re.findall(r"\b\w+\b", text.lower()))


def run(args):
    rows = []
    with httpx.Client(timeout=180, follow_redirects=True) as client:
        host, revision = running_space(client, args.space)
        if args.revision and args.revision != revision:
            raise RuntimeError(f"Expected {args.revision} but runtime serves {revision}")
        for record in read_manifest(args.manifest)[:args.limit]:
            samples, rate = sf.read(record["audio"], dtype="float32")
            if samples.ndim != 1:
                raise ValueError(f"Expected mono FLEURS audio in {record['id']}")
            factor = gcd(rate, RATE)
            mic = np.clip(resample_poly(samples, RATE // factor, rate // factor), -1, 1)
            row = {"id": record["id"], "audio_s": len(samples) / rate,
                   "english_reference": record["english"], "spanish_reference": record["references"]["es"],
                   "live": summarize(live_captions(host, mic)),
                   "review": summarize(review_captions(client, host, mic))}
            rows.append(row)
            print(f"{record['id']}: {row['live']['segments']} live, {row['review']['segments']} review", flush=True)
        if running_space(client, args.space) != (host, revision):
            raise RuntimeError("Space revision changed during comparison; rerun on a stable deployment")
    scores = {}
    for mode in ("live", "review"):
        alignment = process_words([words(row["english_reference"]) for row in rows],
                                  [words(row[mode]["english"]) for row in rows])
        spanish = CHRF(word_order=2).corpus_score([row[mode]["spanish"] for row in rows],
                                                 [[row["spanish_reference"] for row in rows]])
        scores[mode] = {"wer": alignment.wer, "word_errors": alignment.substitutions + alignment.deletions + alignment.insertions,
                        "reference_words": alignment.hits + alignment.substitutions + alignment.deletions,
                        "spanish_chrf_pp": spanish.score}
    result = {"space": args.space, "host": host, "revision": revision, "sample_rate": RATE,
              "options": OPTIONS, "frame_s": .5, "scores": scores, "samples": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--space", default="Ypeng12/live-translate")
    parser.add_argument("--revision", help="Require this running Space revision")
    parser.add_argument("--limit", type=int, help="Optional smoke-test limit, e.g. 1")
    run(parser.parse_args())
