FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/demo/.cache/huggingface \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2

RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 demo

WORKDIR /app
RUN python -m pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml .
# Cache the dependency layer when only the application or example audio changes.
RUN python -c "import tomllib; print('\n'.join(tomllib.load(open('pyproject.toml', 'rb'))['project']['dependencies']))" > /tmp/requirements.txt \
    && python -m pip install -r /tmp/requirements.txt 'websockets>=13,<16'
COPY src/speech_translate/ src/speech_translate/
RUN python -m pip install --no-deps .

USER demo
# Fetch and load the public weights once when building the image. Any missing
# artifact fails the build before a visitor submits audio.
RUN live-translate --device cpu preload --languages es fr de
ENV HF_HUB_OFFLINE=1
EXPOSE 7860
CMD ["live-translate", "--offline", "--device", "cpu", "web", "--host", "0.0.0.0", "--port", "7860"]
