from typing import List
from google import genai
from google.genai import types

from config import get_genai_config


def get_genai_client() -> genai.Client:
    cfg = get_genai_config()
    return genai.Client(api_key=cfg.api_key)


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 80) -> List[str]:
    """Split text into overlapping chunks while avoiding infinite loops."""
    cleaned = " ".join(text.split())
    if not cleaned:
        return []
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")

    chunks: List[str] = []
    start = 0
    length = len(cleaned)

    while start < length:
        end = min(start + chunk_size, length)
        chunks.append(cleaned[start:end])
        if end == length:
            break
        start += chunk_size - overlap

    return chunks


def generate_embeddings(texts: List[str], output_dimensionality: int = 768) -> List[List[float]]:
    """Generate vector embeddings for a list of text chunks via config.py."""
    if not texts:
        return []

    cfg = get_genai_config()
    client = get_genai_client()
    response = client.models.embed_content(
        model=cfg.embedding_model_name,
        contents=texts,
        config=types.EmbedContentConfig(output_dimensionality=output_dimensionality),
    )
    return [emb.values for emb in response.embeddings]