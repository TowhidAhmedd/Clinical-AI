"""
Local embedding model wrapper using sentence-transformers.
Supports BAAI/bge-small-en-v1.5 and all-MiniLM-L6-v2.
"""
from functools import lru_cache
from typing import List
from loguru import logger

try:
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    logger.warning("HuggingFaceEmbeddings not available — using mock embeddings")

from backend.config import get_settings

settings = get_settings()


class MockEmbeddings:
    """Fallback mock embeddings for testing without model downloads."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        import hashlib
        results = []
        for text in texts:
            h = hashlib.md5(text.encode()).digest()
            vec = [((b - 128) / 128.0) for b in h] * (384 // 16)
            results.append(vec[:384])
        return results

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


@lru_cache(maxsize=1)
def get_embedding_model():
    """
    Load and cache the embedding model.
    Falls back to mock embeddings if model cannot be loaded.
    """
    if not EMBEDDINGS_AVAILABLE:
        logger.warning("Using MockEmbeddings — install sentence-transformers for real embeddings")
        return MockEmbeddings()

    model_name = settings.EMBEDDING_MODEL
    logger.info(f"Loading embedding model: {model_name}")
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        logger.info(f"Embedding model loaded: {model_name}")
        return embeddings
    except Exception as e:
        logger.warning(f"Failed to load {model_name}: {e}. Falling back to MockEmbeddings.")
        return MockEmbeddings()


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed a list of texts."""
    model = get_embedding_model()
    return model.embed_documents(texts)


def embed_query(query: str) -> List[float]:
    """Embed a single query string."""
    model = get_embedding_model()
    return model.embed_query(query)


def get_embedding_dimension() -> int:
    """Return the embedding dimension for the configured model."""
    model_dim_map = {
        "BAAI/bge-small-en-v1.5": 384,
        "sentence-transformers/all-MiniLM-L6-v2": 384,
        "BAAI/bge-base-en-v1.5": 768,
        "BAAI/bge-large-en-v1.5": 1024,
    }
    return model_dim_map.get(settings.EMBEDDING_MODEL, 384)
