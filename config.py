import os
from functools import lru_cache
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

# Load .env once from a single authoritative place
load_dotenv()


class GenAIConfig(BaseModel):
    api_key: str = Field(..., min_length=1)
    model_name: str = Field(..., min_length=1)
    embedding_model_name: str = Field(..., min_length=1)


@lru_cache(maxsize=1)
def get_genai_config() -> GenAIConfig:
    """
    Single authoritative accessor for Gemini credentials and model names.
    All other modules must call this instead of reading env vars directly.
    """
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    model_name = os.getenv("GEMINI_MODEL", "").strip()
    embedding_model_name = os.getenv("GEMINI_EMBEDDING_MODEL", "").strip()

    missing = []
    if not api_key:
        missing.append("GEMINI_API_KEY")
    if not model_name:
        missing.append("GEMINI_MODEL")
    if not embedding_model_name:
        missing.append("GEMINI_EMBEDDING_MODEL")

    if missing:
        raise RuntimeError(
            f"Missing required environment configuration in config.py: {', '.join(missing)}"
        )

    try:
        return GenAIConfig(
            api_key=api_key,
            model_name=model_name,
            embedding_model_name=embedding_model_name,
        )
    except ValidationError as exc:
        raise RuntimeError(f"Invalid GenAI configuration: {exc}") from exc