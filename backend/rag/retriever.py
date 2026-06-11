"""
RAG retrieval: semantic search, metadata filtering, reranking,
context compression, and citation generation.
"""
from typing import List, Optional
from pydantic import BaseModel
from loguru import logger


class RetrievedChunk(BaseModel):
    chunk_id: str
    text: str
    score: float
    metadata: dict


class Citation(BaseModel):
    chunk_id: str
    document_name: str
    page_number: int
    score: float
    excerpt: str


class RetrievalResult(BaseModel):
    chunks: List[RetrievedChunk]
    citations: List[Citation]
    context: str
    confidence: float
    total_retrieved: int


def rerank_chunks(
    query: str,
    chunks: List[RetrievedChunk],
    top_k: int = 5,
) -> List[RetrievedChunk]:
    """
    Rerank retrieved chunks using FlashRank or BM25 scoring.
    Falls back to score-based sorting if FlashRank is unavailable.
    """
    try:
        from flashrank import Ranker, RerankRequest

        ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="/tmp/flashrank")
        passages = [{"id": c.chunk_id, "text": c.text} for c in chunks]
        rerank_request = RerankRequest(query=query, passages=passages)
        results = ranker.rerank(rerank_request)

        # Map back to chunks with updated scores
        score_map = {r["id"]: r["score"] for r in results}
        for chunk in chunks:
            if chunk.chunk_id in score_map:
                chunk.score = score_map[chunk.chunk_id]

        chunks.sort(key=lambda x: x.score, reverse=True)
        logger.info(f"Reranked {len(chunks)} chunks with FlashRank")
        return chunks[:top_k]

    except Exception as e:
        logger.warning(f"FlashRank unavailable ({e}), using score-based sorting")
        chunks.sort(key=lambda x: x.score, reverse=True)
        return chunks[:top_k]


def compress_context(chunks: List[RetrievedChunk], max_tokens: int = 3000) -> str:
    """
    Compress retrieved context to fit within token budget.
    Concatenates chunks in order, truncating at max_tokens (approx 4 chars/token).
    """
    max_chars = max_tokens * 4
    context_parts = []
    total_chars = 0

    for i, chunk in enumerate(chunks):
        header = f"\n[Source {i+1}: {chunk.metadata.get('filename', 'Unknown')}, Page {chunk.metadata.get('page', '?')}]\n"
        content = chunk.text.strip()
        part = header + content

        if total_chars + len(part) > max_chars:
            remaining = max_chars - total_chars
            if remaining > 200:
                context_parts.append(part[:remaining] + "...")
            break

        context_parts.append(part)
        total_chars += len(part)

    return "\n".join(context_parts)


def generate_citations(chunks: List[RetrievedChunk]) -> List[Citation]:
    """Generate structured citations from retrieved chunks."""
    citations = []
    seen_chunks = set()

    for chunk in chunks:
        if chunk.chunk_id in seen_chunks:
            continue
        seen_chunks.add(chunk.chunk_id)

        citations.append(
            Citation(
                chunk_id=chunk.chunk_id,
                document_name=chunk.metadata.get("filename", "Unknown Document"),
                page_number=chunk.metadata.get("page", 0),
                score=round(chunk.score, 4),
                excerpt=chunk.text[:200] + ("..." if len(chunk.text) > 200 else ""),
            )
        )

    return citations


def calculate_confidence(chunks: List[RetrievedChunk]) -> float:
    """
    Calculate an overall confidence score based on retrieval scores.
    Returns a value between 0.0 and 1.0.
    """
    if not chunks:
        return 0.0

    scores = [c.score for c in chunks]
    # Weighted average: top chunks count more
    weights = [1 / (i + 1) for i in range(len(scores))]
    weighted_sum = sum(s * w for s, w in zip(scores, weights))
    weight_total = sum(weights)
    confidence = weighted_sum / weight_total if weight_total > 0 else 0.0
    return round(min(max(confidence, 0.0), 1.0), 3)


def build_retrieval_result(
    query: str,
    raw_chunks: List[dict],
    top_k: int = 5,
) -> RetrievalResult:
    """
    Build a full RetrievalResult from raw vector store results:
    Rerank → Compress → Cite → Score
    """
    # Convert raw dicts to RetrievedChunk objects
    chunks = []
    for raw in raw_chunks:
        chunks.append(
            RetrievedChunk(
                chunk_id=raw.get("chunk_id", raw.get("id", "unknown")),
                text=raw.get("text", raw.get("page_content", "")),
                score=raw.get("score", 0.5),
                metadata=raw.get("metadata", {}),
            )
        )

    # Rerank
    reranked = rerank_chunks(query=query, chunks=chunks, top_k=top_k)

    # Compress context
    context = compress_context(reranked)

    # Generate citations
    citations = generate_citations(reranked)

    # Confidence score
    confidence = calculate_confidence(reranked)

    return RetrievalResult(
        chunks=reranked,
        citations=citations,
        context=context,
        confidence=confidence,
        total_retrieved=len(raw_chunks),
    )
