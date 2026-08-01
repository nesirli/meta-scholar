from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    openai_api_key: str
    app_name: str = "MetaScholar"
    corpus_path: Path = PROJECT_DIR / "data" / "corpus.jsonl"


@lru_cache
def get_settings() -> Settings:
    return Settings()

settings = get_settings()