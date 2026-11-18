"""Local entry point, also reusable for a later Hugging Face Space."""
from speech_translate.app import build_app
from speech_translate.config import Config
from speech_translate.models import ModelStore
from speech_translate.pipeline import TranslationPipeline

if __name__ == "__main__":
    config = Config()
    build_app(TranslationPipeline(ModelStore(config), config)).launch(
        server_name="127.0.0.1", share=False)
