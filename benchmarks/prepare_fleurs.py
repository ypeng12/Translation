"""Fetch four fixed and aligned Google FLEURS utterances per selected split."""
import argparse
import csv
import hashlib
import io
import json
import shutil
import tarfile
from pathlib import Path
from urllib.request import urlopen


REVISION = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"
ROOT = f"https://huggingface.co/datasets/google/fleurs/resolve/{REVISION}/data"
ARCHIVE_SHA = {
    "dev": "2658fda72f199e12676ecac9415094667a4e14e149b146e568ea00b2a2f0954c",
    "test": "d9c2e37b41aacd41bc283554a0a82b5476b36887049774ecb2819dcaaa55a356",
}
# First two distinct shared sentences in each duration bucket
# (3–8 s and >8–25 s), in the original en_us metadata order.
RECORDINGS = {
    "dev": {"1590": "15158676295442294624.wav", "1544": "16131823300806444840.wav",
            "1646": "4175831345558885921.wav", "1557": "665062717314476757.wav"},
    "test": {"1980": "12741024238657315067.wav", "1771": "2589557553267364799.wav",
             "1669": "15618717836108767541.wav", "1949": "2652155470394119415.wav"},
}
LOCALES = {"en": "en_us", "es": "es_419", "fr": "fr_fr", "de": "de_de"}


def fetch(url, destination):
    if destination.exists():
        return
    partial = destination.with_suffix(".part")
    with urlopen(url, timeout=120) as response, partial.open("wb") as output:
        shutil.copyfileobj(response, output)
    partial.replace(destination)


def prepare(folder, split):
    folder.mkdir(parents=True, exist_ok=True)
    recordings = RECORDINGS[split]
    languages = {}
    for code, locale in LOCALES.items():
        with urlopen(f"{ROOT}/{locale}/{split}.tsv", timeout=60) as response:
            rows = csv.reader(io.TextIOWrapper(response, encoding="utf-8"), delimiter="\t",
                              quoting=csv.QUOTE_NONE)
            texts = {}
            for sentence_id, filename, text, *_ in rows:
                if sentence_id not in recordings:
                    continue
                if code == "en" and filename != recordings[sentence_id]:
                    continue
                if texts.setdefault(sentence_id, text) != text:
                    raise ValueError(f"Conflicting {code} references for sentence {sentence_id}")
        if texts.keys() != recordings.keys():
            raise ValueError(f"Incomplete {code} references")
        languages[code] = texts
    archive = folder / f"{split}.tar.gz"
    fetch(f"{ROOT}/en_us/audio/{split}.tar.gz", archive)
    checksum = hashlib.sha256()
    with archive.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
    if checksum.hexdigest() != ARCHIVE_SHA[split]:
        raise ValueError(f"Unexpected checksum for {archive}. Remove that file and retry.")
    with tarfile.open(archive, "r:gz") as compressed:
        for name in recordings.values():
            with compressed.extractfile(f"{split}/{name}") as source, (folder / name).open("wb") as target:
                shutil.copyfileobj(source, target)
    rows = [{"id": f"fleurs-{split}-{sid}", "audio": name, "english": languages["en"][sid],
             "references": {code: languages[code][sid] for code in ("es", "fr", "de")}}
            for sid, name in recordings.items()]
    manifest = folder / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                        encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path, nargs="?")
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    args = parser.parse_args()
    default = Path("tmp/fleurs-benchmark" if args.split == "dev" else "tmp/fleurs-test")
    print(prepare(args.folder or default, args.split))
