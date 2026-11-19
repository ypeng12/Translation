---
title: LiveTranslate
emoji: 🎙️
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
short_description: English speech with Spanish, French, or German captions
models:
  - openai/whisper-base.en
  - Helsinki-NLP/opus-mt-en-es
  - Helsinki-NLP/opus-mt-en-fr
  - Helsinki-NLP/opus-mt-en-de
tags:
  - automatic-speech-recognition
  - translation
---

# LiveTranslate

Choose Spanish, French, or German and try four short recordings from different LibriSpeech speakers. Each has a player and a reference transcript. You can also use your microphone or upload an English WAV or FLAC file. The app displays the English transcript alongside the translation and exports both. [Open the demo in its own tab](https://ypeng12-live-translate.hf.space/)

For your own voice, choose **Speech** to request background noise reduction, or **Original** for a quiet room. Check the input level while you speak. After stopping, you can listen to recordings up to two minutes long and choose **Review recording** for a second pass with longer context. To fix an individual recognition error, choose **Edit a caption** and translate just the corrected text. Try this with “followed” in the hawk example.

The same FastAPI application runs on a laptop and in this Space. The container downloads public model weights during the build. It processes audio inside the running service. Its REST endpoints and live WebSocket share the speech recognition, cleanup, and translation code.

## Use the demo from a website

Send an English WAV or FLAC file to `POST /api/translate/audio` with a target language of `es`, `fr`, or `de`. The response includes both transcripts and per-segment timings. The upload limit is 32 MiB and 120 seconds. Microphone clients use `/api/live` over WebSocket. [Interactive API documentation](https://ypeng12-live-translate.hf.space/docs) · [Browser client](https://huggingface.co/spaces/Ypeng12/live-translate/blob/main/src/speech_translate/web/app.js)

## Accuracy and speed

On four held-out FLEURS recordings totaling 35.50 seconds, the updated pipeline processed English-to-Spanish uploads in 23.39 seconds, down from 26.37. These timings came from a single warm run per version on a local Intel Core i7-9750H. Hosted latency varies with load.

| Held-out metric | Before | After |
| --- | ---: | ---: |
| English word error rate | 5.26% | 6.58% |
| English word error rate with Whisper normalization | 4.05% | 2.70% |
| Spanish chrF++ | 44.48 | 44.49 |
| French chrF++ | 54.32 | 56.95 |
| German chrF++ | 58.87 | 59.01 |

The two word error measures score a spoken percentage differently. A separate 12.84-second microphone-paced backend replay produced the same captions on both versions. The first caption arrived at 7.86 seconds, down from 8.09. The final caption arrived 3.82 seconds after the recording ended, up from 3.37. The next evaluation will use more speakers and measure the delay through to the browser.

## Research direction

Two experiments address different parts of the experience. Committing words that agree across overlapping Whisper passes could produce earlier stable captions. Selecting among alternative ASR hypotheses with a local language model could help with misheard phrases. Both need a quality and latency comparison on recorded speech. The current models already cache decoder attention keys and values within a generation. [Whisper-Streaming](https://aclanthology.org/2023.ijcnlp-demo.3/) · [HyPoradise](https://proceedings.neurips.cc/paper_files/paper/2023/hash/6492267465a7ac507be1f9fd1174e78d-Abstract-Datasets_and_Benchmarks.html)

## Models

Silero detects speech, OpenAI Whisper transcribes it, and Helsinki-NLP OPUS-MT translates the English text. See the model cards for intended use, credits, and licenses. [Silero VAD](https://github.com/snakers4/silero-vad) · [Whisper](https://huggingface.co/openai/whisper-base.en) · [OPUS-MT Spanish](https://huggingface.co/Helsinki-NLP/opus-mt-en-es) · [OPUS-MT French](https://huggingface.co/Helsinki-NLP/opus-mt-en-fr) · [OPUS-MT German, CC BY 4.0](https://huggingface.co/Helsinki-NLP/opus-mt-en-de)
