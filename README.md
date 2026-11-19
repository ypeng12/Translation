# LiveTranslate

LiveTranslate turns spoken English into bilingual captions. Choose Spanish, French, or German, then try one of four sample recordings, speak into a microphone, or upload a WAV or FLAC file. The website displays the English transcript alongside the translation and can export both to a text file. [Try the demo](https://ypeng12-live-translate.hf.space/) · [Hugging Face Space](https://huggingface.co/spaces/Ypeng12/live-translate)

The local website and the Space run the same FastAPI application. Model inference happens on the server hosting the page, so running it on your laptop keeps microphone and uploaded audio on that machine. The project uses pretrained models and requires no external inference API key.

Use English for code identifiers, comments, documentation and interface text. Translation outputs, human references and test fixtures use the language being translated or scored.

## Run locally

Use Python 3.10–3.12. Run these commands from the project root. The preload step downloads and checks the model weights for all three languages.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
live-translate preload --languages es fr de
live-translate web
```

Open the [local website](http://127.0.0.1:8000/), select a language, and click **Translate sample**. Expand **Read the reference transcript** to check the English words against the recording. For live captions, click **Record microphone**, then **Stop and finish captions** to translate the remaining audio. Browser microphone access requires localhost or HTTPS.

### When a spoken word comes out wrong

1. **Check what the microphone picked up.** Use **Speech** in a noisy room or **Original** with a clean external mic. These choices request different browser processing settings. Watch the input meter as you speak. A peak warning means the signal is approaching digital clipping.
2. **Listen back, then try more context.** After stopping a recording of up to two minutes, play your captured audio. **Review recording** sends that recording through the upload pipeline, which can recognize up to 25 seconds of speech together. Live captions stay visible while the review runs. The replay is held in browser memory and clears with the session.
3. **Correct a name, number, or phrase.** Click **Edit a caption**, select the segment, and enter what was said. **Translate correction** updates that segment and the bilingual download. Only the edited English text is sent for this action.

The sample menu includes an offer of help (4 s), a hawk in pursuit (4 s), a question about a mountain (6 s), and Rachel's arrival (11 s). Each has a different speaker and a reference transcript. Try the hawk example to exercise correction: in our check Whisper heard “fouled” where the reader said “followed.”

| Task | Command |
| --- | --- |
| Translate a recording from the terminal | `live-translate transcribe recording.wav --language es` |
| Serve the website using cached weights | `HF_HUB_OFFLINE=1 live-translate --offline web` |
| Use Apple Metal on a supported Mac | `live-translate --device mps web` |
| Load a glossary of English names and aliases | `live-translate --glossary configs/glossary.example.json web` |
| Try the Gradio interface on port 7860 | `live-translate ui` |
| Compare a larger English recognizer | `live-translate --asr-model openai/whisper-small.en web` |

Automatic device selection uses CUDA when available and otherwise uses the CPU. The local web server binds to `127.0.0.1:8000` by default. Use `live-translate web --host 0.0.0.0 --port 8000` to serve it on a network.

## How captions are produced

| Stage | Implementation |
| --- | --- |
| Prepare audio | Convert to 16 kHz mono. Silero voice activity detection (VAD) finds speech and skips silent input. |
| Transcribe | Whisper `openai/whisper-base.en` produces an English transcript for each segment. |
| Clean up English | Optional rules remove selected fillers and accidental repetitions, tidy punctuation and capitalization, and apply glossary aliases. The API also returns the raw transcript for comparison. |
| Translate | A separate Helsinki-NLP OPUS-MT model handles each language pair. |
| Display | The website pairs each English segment with its translation. Microphone captions appear as each segment finishes. |

Uploads and live speech use different windows. With **Natural pauses**, a recording up to 25 seconds stays together after a speech check. Keeping the full recording gives Whisper context around quiet words and sentence boundaries. Longer uploads split at pauses into windows of up to 25 seconds. Microphone input uses a 0.6-second pause threshold and an eight-second maximum window. **Fixed duration** also uses an eight-second cap. Loading Silero preserves the CPU thread budget available to Whisper and OPUS-MT.

The model adapters, segmentation logic, cleanup rules and pipeline can be used independently of the website. [Model adapters](src/speech_translate/models.py) · [Audio segmentation](src/speech_translate/audio.py) · [Text rules](src/speech_translate/text.py) · [Pipeline](src/speech_translate/pipeline.py)

The microphone converter now preserves the resampling filter history between WebSocket frames and flushes the final partial frame on stop. With the 4.09-second sample played at simulated 44.1 and 48 kHz capture rates, each stream produced the same 65,440 output samples as a single-pass conversion, including matching waveform values. This test measures signal fidelity. Speaker-specific accuracy requires a recording and its intended words.

We compared live captions with **Review recording** on the public Space using four held-out English recordings. Three outputs were identical. In the fourth, live recognition split a percentage into “one. 100%” and review recovered “100%.” Strict word error rate changed from 5.26% to 6.58% because the reference spells out “one-hundred percent.” Spanish chrF++ moved from 44.48 to 44.49. Review can help at a phrase boundary, while names may still need a correction. [Hosted comparison and reproduction steps](benchmarks/README.md) · [Per-recording transcripts](benchmarks/results/hosted-capture-review.json)

## Use the API from a website

The included frontend uses plain HTML, CSS, JavaScript and a browser AudioWorklet. FastAPI serves it from the same origin as the API. To integrate with an existing website, route the API paths through its backend or reverse proxy and enable WebSocket forwarding for live captions. [Frontend source](src/speech_translate/web/) · [FastAPI routes](src/speech_translate/server.py) · [Interactive API documentation](https://ypeng12-live-translate.hf.space/docs)

| Endpoint | Request and response |
| --- | --- |
| `GET /api/health` | Returns service status, supported languages and whether the recognizer has loaded. |
| `POST /api/translate/text` | Accepts JSON with `text`, `language` and `clean`. Returns English text and its translation. |
| `POST /api/translate/audio` | Accepts multipart fields `audio`, `language`, `strategy` and `clean`. Returns both captions and per-segment details and timings. The website accepts WAV or FLAC up to 32 MiB and 120 seconds. |
| `WS /api/live` | Accepts session settings followed by mono PCM audio frames. Streams caption segments and acknowledges each frame. |

Try the local audio endpoint using the sample included with the website.

```bash
curl --fail-with-body http://127.0.0.1:8000/api/translate/audio \
  -F 'audio=@src/speech_translate/web/example.wav' \
  -F 'language=fr' -F 'strategy=pause' -F 'clean=true'
```

For live input, send `{"language":"fr","sample_rate":16000,"strategy":"pause","clean":true}` as the first WebSocket message. Wait for `ready`, send one binary frame of little-endian float32 mono PCM (up to two seconds), collect any `segment` messages, and wait for `ack` before sending the next frame. At the end, send `{"type":"stop"}` and collect the remaining segments through `done`. The server allows 15 seconds for settings and 60 seconds between subsequent messages. [Browser client](src/speech_translate/web/app.js)

## Measured accuracy and speed

Four held-out FLEURS recordings contained 35.50 seconds of English speech with human references in all three target languages. A separate four-recording development set guided configuration selection. The table compares the previous and updated pipelines in one warm run per version on a local Intel Core i7-9750H. Upload timing covers audio preparation, VAD, recognition, cleanup and Spanish translation. Model loading, network transit and browser rendering are outside the timer.

| Held-out metric | Before | After |
| --- | ---: | ---: |
| English-to-Spanish upload processing time | 26.37 s | 23.39 s |
| Spanish real-time factor (processing time / audio duration) | 0.743 | 0.659 |
| English word error rate (project scoring) | 5.26% | 6.58% |
| English word error rate (Whisper normalization) | 4.05% | 2.70% |
| Spanish translation chrF++ | 44.48 | 44.49 |
| French translation chrF++ | 54.32 | 56.95 |
| German translation chrF++ | 58.87 | 59.01 |

Spanish processing took 2.98 seconds less. Translation chrF++ increased most for French. The English word error rates moved in opposite directions because the recognizer changed “one 100%” to “100%” for the reference “one-hundred percent.” Project scoring counted one additional error (5 of 76 words). Whisper's number normalization counted one fewer (2 of 74 words).

Live timing was measured separately by replaying one 12.84-second recording at microphone pace. The first Spanish caption arrived at 7.86 seconds, down from 8.09. The final caption arrived 3.82 seconds after the recording ended, up from 3.37. Both versions produced the same captions. These small samples describe this machine and these recordings. [Benchmark method, per-recording results and reproduction commands](benchmarks/README.md)

### Evaluate on your own recordings

Create a JSONL manifest with one unique sentence ID, an audio path relative to the manifest, the English transcript and human translations for the languages being scored. Align recordings and translations by sentence ID within the same split, then verify the text. The example manifest illustrates the format and requires you to supply its audio. [Example manifest](data/example.jsonl) · [Evaluation code](src/speech_translate/evaluation.py)

```bash
live-translate evaluate data/test.jsonl --languages es fr de --output outputs/evaluation.json
live-translate evaluate data/test.jsonl --languages es --strategy fixed --output outputs/fixed.json

# Optional COMET scoring downloads the evaluation model on first use.
python -m pip install -e '.[comet]'
live-translate evaluate data/test.jsonl --languages es --comet
```

The evaluator translates the human English transcript, the raw recognized English and the cleaned recognized English with the same translation model. It scores all three against the same human translation with chrF++ and BLEU. The project target is a cleaned-speech chrF++ score of at least 80% of the score from human English input. The updated pipeline retained 98.98% for Spanish, 95.23% for French and 93.55% for German on the four held-out recordings. These percentages measure relative translation scores.

### Next evaluation priorities

1. Expand the held-out set across speakers, recording lengths, names and numbers. Repeat timing runs and report WER, chrF++ and processing time per language before changing the default recognizer.
2. Measure the delay from the end of speech to the on-screen caption in a real browser. Prioritize the final caption, which took longer in the microphone replay.
3. Evaluate the proposal's COMET target of 0.80 and mean human rating of 4/5 on the expanded set. Punctuation-based buffering and alternative translation models also remain experiments. [Project proposal](641%20.pdf)

### Streaming, LLMs and the KV cache

The first streaming experiment is to show words that agree across two consecutive Whisper passes, then translate a stable phrase. For automatic corrections, evaluate a local language model that picks among a few ASR hypotheses and can keep the original. Track changed names, numbers and negations alongside overall word error rate. Both are proposed experiments. [Whisper-Streaming](https://aclanthology.org/2023.ijcnlp-demo.3/) · [HyPoradise](https://proceedings.neurips.cc/paper_files/paper/2023/hash/6492267465a7ac507be1f9fd1174e78d-Abstract-Datasets_and_Benchmarks.html)

Whisper and OPUS-MT already reuse attention keys and values while decoding a single segment. On four English transcripts, Spanish translation generation took a median of 4.05 s with caching and 12.27 s without it on the local CPU. Output tokens matched in every run. A new audio window or edited English phrase changes the encoder input and needs a fresh generation. The research notes also describe a separate prefix-cache experiment for a future local LLM corrector. [Research plan, papers and cache experiment](benchmarks/RESEARCH.md)

## Deploy a Hugging Face Space

The public demo runs on CPU Basic. Its Docker image downloads the weights at build time and runs the same web server on port 7860 with offline model loading. The publishing script selects the application code, sample and deployment files, and uses the English Space card as the hosted README. [Space card](space/README.md) · [Dockerfile](Dockerfile) · [Publishing script](scripts/publish_space.py)

From an activated environment authenticated with write access to your Hugging Face namespace, replace `YOUR_NAMESPACE` with your username or organization.

```bash
# Inspect the files that will be published.
python scripts/publish_space.py YOUR_NAMESPACE/live-translate --dry-run

# Create a public Docker Space and publish the demo.
python scripts/publish_space.py YOUR_NAMESPACE/live-translate

# Publish a later revision to the existing Space.
python scripts/publish_space.py YOUR_NAMESPACE/live-translate --update
```

## Tests and credits

Run the model-independent Python suite and browser recorder tests from the project root. CI runs the Python tests on 3.10 and 3.12 and the JavaScript checks on Node.js 22.

```bash
python -m pytest -q
node --check src/speech_translate/web/app.js
node --test tests/*.cjs
```

The four demo recordings come from LibriSpeech by V. Panayotov, G. Chen, D. Povey and S. Khudanpur. Clips `1272-135031-0006`, `7176-88083-0009`, `4507-16021-0033` and `5683-32879-0002` were converted to WAV. The reference panel identifies the selected clip. [LibriSpeech](https://www.openslr.org/12/) · [CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/)

See the upstream model cards for intended uses, licenses and training data. [Silero VAD](https://github.com/snakers4/silero-vad) · [Whisper base.en](https://huggingface.co/openai/whisper-base.en) · [OPUS-MT English–Spanish](https://huggingface.co/Helsinki-NLP/opus-mt-en-es) · [English–French](https://huggingface.co/Helsinki-NLP/opus-mt-en-fr) · [English–German](https://huggingface.co/Helsinki-NLP/opus-mt-en-de)
