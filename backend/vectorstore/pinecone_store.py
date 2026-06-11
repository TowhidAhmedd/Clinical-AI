"""
Vector store module: Pinecone integration for document storage and retrieval.
Includes index management, upsert, and semantic search.
"""
import json
from typing import List, Optional
from loguru import logger

try:
    from pinecone import Pinecone, ServerlessSpec
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False
    logger.warning("pinecone-client not installed — using in-memory store")

from backend.config import get_settings
from backend.rag.document_processor import DocumentChunk
# NOTE: embedder is imported lazily inside functions to avoid loading the
# embedding model (and triggering a download) at module import time.

settings = get_settings()


# --- In-Memory Fallback ---
class InMemoryVectorStore:
    """Simple in-memory vector store for testing/development without Pinecone."""

    def __init__(self):
        self._store: dict = {}
        logger.warning("Using InMemoryVectorStore — set PINECONE_API_KEY for production")

    def upsert(self, chunks: List[DocumentChunk], embeddings: List[List[float]]):
        for chunk, emb in zip(chunks, embeddings):
            self._store[chunk.chunk_id] = {
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "metadata": chunk.metadata,
                "embedding": emb,
            }
        logger.info(f"InMemoryStore: upserted {len(chunks)} chunks")

    def query(self, query_embedding: List[float], top_k: int = 10, filter: dict = None) -> List[dict]:
        import numpy as np

        if not self._store:
            return []

        query_vec = np.array(query_embedding)
        scored = []
        for item in self._store.values():
            vec = np.array(item["embedding"])
            if len(vec) != len(query_vec):
                continue
            # Cosine similarity
            norm_q = np.linalg.norm(query_vec)
            norm_v = np.linalg.norm(vec)
            if norm_q == 0 or norm_v == 0:
                score = 0.0
            else:
                score = float(np.dot(query_vec, vec) / (norm_q * norm_v))

            # Apply metadata filter
            if filter:
                match = all(
                    item["metadata"].get(k) == v for k, v in filter.items()
                )
                if not match:
                    continue

            scored.append({
                "chunk_id": item["chunk_id"],
                "text": item["text"],
                "metadata": item["metadata"],
                "score": score,
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def delete_by_doc_id(self, doc_id: str):
        keys_to_delete = [k for k, v in self._store.items() if v["metadata"].get("doc_id") == doc_id]
        for k in keys_to_delete:
            del self._store[k]
        logger.info(f"Deleted {len(keys_to_delete)} chunks for doc_id={doc_id}")

    def list_documents(self) -> List[dict]:
        seen = {}
        for item in self._store.values():
            doc_id = item["metadata"].get("doc_id", "unknown")
            if doc_id not in seen:
                seen[doc_id] = {
                    "doc_id": doc_id,
                    "filename": item["metadata"].get("filename", "unknown"),
                    "chunk_count": 0,
                }
            seen[doc_id]["chunk_count"] += 1
        return list(seen.values())


# --- Pinecone Store ---
class PineconeVectorStore:
    """Production Pinecone vector store."""

    def __init__(self):
        from backend.embeddings.embedder import get_embedding_dimension
        self.pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        self.index_name = settings.PINECONE_INDEX_NAME
        self.dimension = get_embedding_dimension()
        self._ensure_index()
        self.index = self.pc.Index(self.index_name)
        logger.info(f"PineconeVectorStore connected: index={self.index_name}")

    def _ensure_index(self):
        """Create the Pinecone index if it doesn't exist."""
        existing = [idx.name for idx in self.pc.list_indexes()]
        if self.index_name not in existing:
            logger.info(f"Creating Pinecone index: {self.index_name}")
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region=settings.PINECONE_ENVIRONMENT),
            )
            logger.info(f"Index {self.index_name} created")

    def upsert(self, chunks: List[DocumentChunk], embeddings: List[List[float]]):
        """Upsert chunks with embeddings into Pinecone."""
        vectors = []
        for chunk, emb in zip(chunks, embeddings):
            meta = {k: str(v) for k, v in chunk.metadata.items()}
            meta["text"] = chunk.text[:512]  # Pinecone metadata limit
            vectors.append((chunk.chunk_id, emb, meta))

        # Batch upsert in groups of 100
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i : i + batch_size]
            self.index.upsert(vectors=batch)

        logger.info(f"Upserted {len(chunks)} chunks to Pinecone")

    def query(self, query_embedding: List[float], top_k: int = 10, filter: dict = None) -> List[dict]:
        """Query Pinecone for similar chunks."""
        kwargs = {
            "vector": query_embedding,
            "top_k": top_k,
            "include_metadata": True,
        }
        if filter:
            kwargs["filter"] = filter

        response = self.index.query(**kwargs)

        results = []
        for match in response.matches:
            metadata = dict(match.metadata)
            text = metadata.pop("text", "")
            results.append({
                "chunk_id": match.id,
                "text": text,
                "metadata": metadata,
                "score": match.score,
            })

        return results

    def delete_by_doc_id(self, doc_id: str):
        """Delete all chunks belonging to a document."""
        results = self.index.query(
            vector=[0.0] * self.dimension,
            top_k=10000,
            include_metadata=True,
            filter={"doc_id": {"$eq": doc_id}},
        )
        ids = [m.id for m in results.matches]
        if ids:
            self.index.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} chunks for doc_id={doc_id}")

    def list_documents(self) -> List[dict]:
        """List all unique documents in the index (best-effort)."""
        stats = self.index.describe_index_stats()
        return [{"total_vectors": stats.total_vector_count}]


# --- Factory ---
_vector_store_instance = None


def get_vector_store():
    """Get or create the vector store singleton."""
    global _vector_store_instance
    if _vector_store_instance is None:
        if PINECONE_AVAILABLE and settings.PINECONE_API_KEY:
            try:
                _vector_store_instance = PineconeVectorStore()
            except Exception as e:
                logger.warning(f"Pinecone init failed: {e} — falling back to InMemoryVectorStore")
                _vector_store_instance = InMemoryVectorStore()
        else:
            _vector_store_instance = InMemoryVectorStore()
    return _vector_store_instance


def index_document(chunks: List[DocumentChunk]) -> int:
    """Embed and index a list of document chunks."""
    from backend.embeddings.embedder import embed_texts

    texts = [c.text for c in chunks]
    embeddings = embed_texts(texts)

    store = get_vector_store()
    store.upsert(chunks, embeddings)
    return len(chunks)


def search_documents(
    query: str,
    top_k: int = 10,
    filter_doc_id: Optional[str] = None,
) -> List[dict]:
    """Search for relevant document chunks."""
    from backend.embeddings.embedder import embed_query

    query_emb = embed_query(query)
    store = get_vector_store()

    filter_dict = None
    if filter_doc_id:
        filter_dict = {"doc_id": filter_doc_id}

    return store.query(query_emb, top_k=top_k, filter=filter_dict)
