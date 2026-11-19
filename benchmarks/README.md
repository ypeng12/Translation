# FLEURS speech translation benchmark

This paired check measures English audio uploads on the default pause setting. It uses four English recordings and human references in Spanish, French and German from the [Google FLEURS dataset](https://huggingface.co/datasets/google/fleurs/tree/70bb2e84b976b7e960aa89f1c648e09c59f894dd). FLEURS contains readings of parallel FLORES sentences. Its license is [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

The preparation script joins the official raw transcripts on parallel sentence ID within each split. It checks repeated readings for agreement on the target text. The fixed English recordings are the first two unique shared sentences in each of the 3–8 second and over-8–25 second duration ranges in the pinned English TSV. The four development recordings total 37.80 seconds. This selection covers both sides of the old eight-second segmentation boundary.

A second manifest applies the same selection rule independently to the test split. Its four IDs (1980, 1771, 1669, 1949) were fixed before scoring. They total 35.50 seconds. The test results evaluate the configuration selected on development; no recordings overlap.

## Measured upload results

| Metric | Development before | Development after | Held-out before | Held-out after |
| --- | ---: | ---: | ---: | ---: |
| Spanish pipeline wall time (s) | 30.17 | 13.20 | 26.37 | 23.39 |
| Spanish RTF (lower is faster) | 0.798 | 0.349 | 0.743 | 0.659 |
| English WER (lower is better) | 5.32% | 4.26% | 5.26% | 6.58% |
| English WER, Whisper normalization (secondary) | 4.26% | 3.19% | 4.05% | 2.70% |
| Spanish chrF++ | 54.24 | 53.39 | 44.48 | 44.49 |
| French chrF++ | 62.23 | 63.72 | 54.32 | 56.95 |
| German chrF++ | 62.19 | 62.90 | 58.87 | 59.01 |

On the four held-out utterances the updated pipeline retained 98.98%, 95.23% and 93.55% of the gold-text translation score for Spanish, French and German, respectively. Spanish and German moved by less than 0.2 chrF++ points. English ASR made five word errors against 76 reference words, versus four before. The low sample counts and a single timing pass limit any general accuracy or speed claim. The held-out set did not drive configuration selection. Formal acceptance requires a larger frozen set plus human review.

The secondary WER was computed after freezing both sets of predictions. It applies Transformers 4.49.0's Whisper **EnglishTextNormalizer**, including its spelling map from the pinned Whisper tokenizer, to both references and predictions. Under that convention development errors fell from 4/94 to 3/94 and held-out errors from 3/74 to 2/74. Only held-out ID 1980 changed after Whisper normalization. The original recognized “one 100%” and the update recognized “100%”. The human reference “one-hundred percent” normalizes to “100%”. The primary WER uses lowercase word tokens and gives partial credit for “one” in the original prediction. Both metrics are included so the scoring convention is explicit.

The result files contain per-utterance hypotheses, model outputs, recording lengths, segment counts and timings. Compare [development before](results/baseline.json) and [after](results/candidate.json), and [held-out before](results/test-baseline.json) and [after](results/test-candidate.json).

## Reproduce

Run these commands from the project root after installing the project. The first command downloads a 171 MB audio archive, checks its digest, and writes four WAV files and an evaluation manifest under the ignored temporary directory.

```sh
python benchmarks/prepare_fleurs.py
mkdir -p tmp/fleurs-benchmark/baseline
git archive 31a5262aa532c5b4c19359009c3e279d82a39606 src | tar -xf - -C tmp/fleurs-benchmark/baseline
OMP_NUM_THREADS=2 PYTHONPATH=tmp/fleurs-benchmark/baseline/src python benchmarks/run_fleurs.py tmp/fleurs-benchmark/manifest.jsonl outputs/fleurs-before.json --threads 2 --label baseline-31a5262
OMP_NUM_THREADS=2 PYTHONPATH=src python benchmarks/run_fleurs.py tmp/fleurs-benchmark/manifest.jsonl outputs/fleurs-after.json --threads 2 --label file-context
```

Run `python benchmarks/prepare_fleurs.py --split test` and repeat the last two commands with `tmp/fleurs-test/manifest.jsonl` and distinct output filenames to evaluate the held-out sample. This downloads the 290 MB English test archive.

Each revision runs in its own process. Use `--offline` with a populated model cache. The harness measures wall time from decoded waveform to completed Spanish captions, including resampling, VAD, ASR and translation. Weight loading, file input, and translation of the human reference are outside the timer. French and German translations use the same ASR segments and are scored separately. Their inference times do not enter the Spanish RTF.

The recorded run used an Intel Core i7-9750H and the cached model revisions below.

| Model | Revision |
| --- | --- |
| Whisper base.en | `911407f4214e0e1d82085af863093ec0b66f9cd6` |
| OPUS-MT en-es | `5bc4493d463cf000c1f0b50f8d56886a392ed4ab` |
| OPUS-MT en-fr | `dd7f6540a7a48a7f4db59e5c0b9c42c8eea67f18` |
| OPUS-MT en-de | `6183067f769a302e3861815543b9f312c71b0ca4` |

WER uses the same punctuation and case normalization as the main evaluation CLI. chrF++ is corpus chrF with word order 2. The per-language retention ratio is the speech translation chrF++ divided by the chrF++ from translating gold English with the same MT weights. RTF is total timed seconds divided by total source audio seconds. An RTF below one means the upload pipeline processed audio faster than its duration on that machine. These one-pass measurements include ordinary timing variation. Both processes requested two CPU threads. In the original application, Silero reset PyTorch to one. The updated application preserved two. The comparison measures the combined effect of segmentation and thread management.

## Microphone pacing

The [stream replay](replay_live.py) feeds 12.84 seconds of natural English speech (development ID 1544) through the live buffer in half-second PCM frames, sleeping until each frame would arrive from a microphone. It records the first-caption time and the delay from the end of the source audio until the last caption. The replay includes queueing behind prior segments and the final buffer flush. Device capture, browser rendering and network transit remain outside these measurements.

In [the baseline replay](results/live-baseline.json) the first caption arrived at 8.09 s and the last at 3.37 s after capture ended. In [the updated replay](results/live-candidate.json) these were 7.86 s and 3.82 s. Streaming compute RTF changed from 0.584 to 0.599. Both versions returned two captions, zero WER and 57.66 Spanish chrF++. One backend replay cannot establish a consistent change in interactive latency.

```sh
OMP_NUM_THREADS=2 PYTHONPATH=tmp/fleurs-benchmark/baseline/src python benchmarks/replay_live.py tmp/fleurs-benchmark/manifest.jsonl outputs/live-before.json --label baseline-31a5262
OMP_NUM_THREADS=2 PYTHONPATH=src python benchmarks/replay_live.py tmp/fleurs-benchmark/manifest.jsonl outputs/live-after.json --label final
```

## Full-recording review on the public Space

[The capture comparison](compare_capture.py) resamples each of the four held-out FLEURS recordings to mono 44.1 kHz to simulate a browser microphone. It sends half-second float32 frames through the deployed WebSocket, waiting for each acknowledgement, then submits an upload encoded from those same samples with the browser review feature's 16-bit WAV conversion. Both paths use natural pauses, English text cleanup and Spanish output. WER scores the visible English captions against the FLEURS transcript. Spanish chrF++ scores the joined translations against the corresponding human references. Frames are sent as acknowledgements arrive, so this comparison measures caption quality only.

```sh
python -m pip install 'websockets>=14,<16' httpx
python benchmarks/prepare_fleurs.py --split test
PYTHONPATH=src python benchmarks/compare_capture.py tmp/fleurs-test/manifest.jsonl outputs/capture-review.json
```

The script records the running Space revision and checks it again after the last clip. Supply `--revision REVISION` to require a particular deployment.

On [Space revision d48ddd5](https://huggingface.co/spaces/Ypeng12/live-translate/tree/d48ddd525e67f233c9b8669d180b6dbc172a2db4), all four live sessions and all four reviews completed. English WER went from **5.26% live** (4 word errors / 76 reference words) to **6.58% after review** (5 / 76). Spanish chrF++ moved from **44.48 to 44.49**. Three recordings returned identical captions in both modes. The difference came from test ID 1980, whose reference includes “one-hundred percent”. Review recognized “100%” and live captions used “one. 100%”. The primary WER treats numerals and spelled-out words as different tokens. [Per-recording captions, references and aggregate scores](results/hosted-capture-review.json)
