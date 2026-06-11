"""
Unit tests for the RAG pipeline: chunking, retrieval, citation generation.
"""
import pytest
import sys
import os
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestDocumentProcessor:
    """Tests for document processing pipeline."""

    def test_clean_text_removes_extra_whitespace(self):
        from backend.rag.document_processor import clean_text
        result = clean_text("  Hello   world  \n\n  test  ")
        assert "  " not in result
        assert result == "Hello world test"

    def test_clean_text_removes_non_printable(self):
        from backend.rag.document_processor import clean_text
        result = clean_text("Hello\x00World\x01Test")
        assert "\x00" not in result
        assert "\x01" not in result

    def test_chunk_text_creates_chunks(self):
        from backend.rag.document_processor import chunk_text
        pages = [
            {
                "text": "This is a test document about medical education. " * 20,
                "page": 1,
            }
        ]
        chunks = chunk_text(pages, doc_id="test123", filename="test.txt")
        assert len(chunks) > 0
        assert all(c.chunk_id for c in chunks)
        assert all(c.text for c in chunks)

    def test_chunk_text_preserves_metadata(self):
        from backend.rag.document_processor import chunk_text
        pages = [{"text": "Sample medical text. " * 30, "page": 5}]
        chunks = chunk_text(pages, doc_id="doc123", filename="medical.pdf")
        assert all(c.metadata["filename"] == "medical.pdf" for c in chunks)
        assert all(c.metadata["doc_id"] == "doc123" for c in chunks)
        assert all(c.metadata["page"] == 5 for c in chunks)

    def test_chunk_text_generates_unique_ids(self):
        from backend.rag.document_processor import chunk_text
        pages = [{"text": "Text about anatomy. " * 50, "page": 1}]
        chunks = chunk_text(pages, doc_id="doc123", filename="test.txt")
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "Chunk IDs must be unique"

    def test_chunk_respects_chunk_size(self):
        from backend.rag.document_processor import chunk_text
        pages = [{"text": "word " * 200, "page": 1}]
        chunks = chunk_text(pages, doc_id="doc", filename="f.txt", chunk_size=100)
        for chunk in chunks:
            assert len(chunk.text) <= 200  # Allow some overlap tolerance

    def test_extract_text_from_txt(self):
        from backend.rag.document_processor import extract_text_from_txt
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("This is a test medical document.\n" * 20)
            tmp_path = f.name
        try:
            pages = extract_text_from_txt(tmp_path)
            assert len(pages) > 0
            assert all("text" in p for p in pages)
            assert all("page" in p for p in pages)
        finally:
            os.unlink(tmp_path)

    def test_extract_text_from_nonexistent_file(self):
        from backend.rag.document_processor import extract_text_from_txt
        result = extract_text_from_txt("/nonexistent/path/file.txt")
        assert result == []

    def test_process_document_txt(self):
        from backend.rag.document_processor import process_document
        content = "Medical education content about the heart. " * 50
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(content)
            tmp_path = f.name
        try:
            result = process_document(tmp_path, "test_medical.txt", doc_id="testdoc123")
            assert result.doc_id == "testdoc123"
            assert result.filename == "test_medical.txt"
            assert result.total_chunks > 0
            assert len(result.chunks) == result.total_chunks
        finally:
            os.unlink(tmp_path)


class TestRetriever:
    """Tests for the retrieval pipeline."""

    def test_generate_citations(self):
        from backend.rag.retriever import generate_citations, RetrievedChunk
        chunks = [
            RetrievedChunk(
                chunk_id="chunk_001",
                text="ACE inhibitors block angiotensin-converting enzyme.",
                score=0.92,
                metadata={"filename": "cardiology.pdf", "page": 12, "chunk_index": 0},
            ),
            RetrievedChunk(
                chunk_id="chunk_002",
                text="Beta blockers reduce heart rate and blood pressure.",
                score=0.85,
                metadata={"filename": "cardiology.pdf", "page": 15, "chunk_index": 1},
            ),
        ]
        citations = generate_citations(chunks)
        assert len(citations) == 2
        assert citations[0].document_name == "cardiology.pdf"
        assert citations[0].page_number == 12
        assert citations[0].score == 0.92
        assert len(citations[0].excerpt) <= 203  # 200 chars + "..."

    def test_generate_citations_deduplicates(self):
        from backend.rag.retriever import generate_citations, RetrievedChunk
        chunks = [
            RetrievedChunk(
                chunk_id="same_id",
                text="Duplicate content.",
                score=0.9,
                metadata={"filename": "doc.pdf", "page": 1},
            ),
            RetrievedChunk(
                chunk_id="same_id",  # Duplicate
                text="Duplicate content.",
                score=0.9,
                metadata={"filename": "doc.pdf", "page": 1},
            ),
        ]
        citations = generate_citations(chunks)
        assert len(citations) == 1

    def test_calculate_confidence_empty(self):
        from backend.rag.retriever import calculate_confidence
        assert calculate_confidence([]) == 0.0

    def test_calculate_confidence_high_score(self):
        from backend.rag.retriever import calculate_confidence, RetrievedChunk
        chunks = [
            RetrievedChunk(chunk_id="c1", text="t", score=0.9, metadata={}),
            RetrievedChunk(chunk_id="c2", text="t", score=0.8, metadata={}),
        ]
        conf = calculate_confidence(chunks)
        assert 0.0 < conf <= 1.0
        assert conf > 0.7

    def test_calculate_confidence_low_score(self):
        from backend.rag.retriever import calculate_confidence, RetrievedChunk
        chunks = [
            RetrievedChunk(chunk_id="c1", text="t", score=0.1, metadata={}),
        ]
        conf = calculate_confidence(chunks)
        assert conf < 0.5

    def test_compress_context(self):
        from backend.rag.retriever import compress_context, RetrievedChunk
        chunks = [
            RetrievedChunk(
                chunk_id=f"c{i}",
                text=f"Medical content chunk {i}. " * 20,
                score=0.9 - (i * 0.05),
                metadata={"filename": "doc.pdf", "page": i + 1},
            )
            for i in range(5)
        ]
        context = compress_context(chunks, max_tokens=500)
        assert len(context) > 0
        assert len(context) <= 500 * 4 + 500  # Some tolerance

    def test_build_retrieval_result(self):
        from backend.rag.retriever import build_retrieval_result
        raw_chunks = [
            {
                "chunk_id": f"chunk_{i}",
                "text": f"Medical education content about cardiology. Point {i}. " * 5,
                "score": 0.9 - (i * 0.1),
                "metadata": {"filename": "cardiology.pdf", "page": i + 1},
            }
            for i in range(8)
        ]
        result = build_retrieval_result("What are ACE inhibitors?", raw_chunks, top_k=5)
        assert len(result.chunks) <= 5
        assert len(result.citations) > 0
        assert len(result.context) > 0
        assert 0.0 <= result.confidence <= 1.0
        assert result.total_retrieved == 8
