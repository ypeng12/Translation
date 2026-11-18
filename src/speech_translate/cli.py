import argparse
import json
from pathlib import Path

from .config import Config


def main():
    parser = argparse.ArgumentParser(description="Local English speech translation")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--asr-model", default="openai/whisper-base.en")
    parser.add_argument("--offline", action="store_true", help="Use cached HF model files only")
    parser.add_argument("--glossary", type=Path, help="JSON mapping ASR aliases to canonical English names")
    sub = parser.add_subparsers(dest="command", required=True)
    ui = sub.add_parser("ui")
    ui.add_argument("--port", type=int, default=7860)
    website = sub.add_parser("web", help="Website + REST API + microphone WebSocket")
    website.add_argument("--port", type=int, default=8000)
    website.add_argument("--host", default="127.0.0.1")
    preload = sub.add_parser("preload")
    preload.add_argument("--languages", nargs="+", choices=["es", "fr", "de"], default=["es", "fr", "de"])
    transcribe = sub.add_parser("transcribe")
    transcribe.add_argument("audio", type=Path)
    transcribe.add_argument("--language", choices=["es", "fr", "de"], default="es")
    transcribe.add_argument("--strategy", choices=["pause", "fixed"], default="pause")
    transcribe.add_argument("--raw", action="store_true")
    evaluation = sub.add_parser("evaluate")
    evaluation.add_argument("manifest", type=Path)
    evaluation.add_argument("--languages", nargs="+", choices=["es", "fr", "de"], default=["es", "fr", "de"])
    evaluation.add_argument("--strategy", choices=["pause", "fixed"], default="pause")
    evaluation.add_argument("--output", type=Path, default=Path("outputs/evaluation.json"))
    evaluation.add_argument("--comet", action="store_true")
    args = parser.parse_args()
    from .models import ModelStore
    from .pipeline import TranslationPipeline
    from .text import TextProcessor
    config = Config(asr_model=args.asr_model, device=args.device, local_files_only=args.offline)
    glossary = json.loads(args.glossary.read_text()) if args.glossary else {}
    models = ModelStore(config)
    pipeline = TranslationPipeline(models, config, TextProcessor(glossary))
    if args.command == "web":
        import uvicorn

        from .server import create_app
        uvicorn.run(create_app(pipeline), host=args.host, port=args.port)
    elif args.command == "ui":
        from .app import build_app
        build_app(pipeline).launch(server_name="127.0.0.1", server_port=args.port, share=False)
    elif args.command == "preload":
        models.warmup(args.languages)
        print("Models loaded successfully:", ", ".join(args.languages), "on", models.device)
    elif args.command == "transcribe":
        import soundfile as sf
        audio, rate = sf.read(args.audio)
        results = pipeline.audio(audio, rate, args.language, not args.raw, args.strategy)
        print(json.dumps([r.to_dict() for r in results], ensure_ascii=False, indent=2))
    elif args.command == "evaluate":
        from .evaluation import evaluate, read_manifest
        records = read_manifest(args.manifest)
        report = evaluate(pipeline, records, args.languages, args.strategy, args.comet)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(report["languages"], ensure_ascii=False, indent=2))
        print("Saved:", args.output)


if __name__ == "__main__":
    main()
