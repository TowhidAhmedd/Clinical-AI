"""
API integration tests: authentication, upload, chat endpoints.
"""
import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


class TestHealthEndpoint:
    """Tests for the health check endpoint."""

    def test_health_check(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "embedding_model" in data
        assert "llm_model" in data

    def test_root_endpoint(self):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "docs" in data


class TestAuthentication:
    """Tests for JWT authentication."""

    def test_login_valid_credentials(self):
        response = client.post(
            "/auth/login",
            json={"username": "admin", "password": "admin123"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        assert data["expires_in"] > 0

    def test_login_invalid_password(self):
        response = client.post(
            "/auth/login",
            json={"username": "admin", "password": "wrongpassword"},
        )
        assert response.status_code == 401

    def test_login_invalid_username(self):
        response = client.post(
            "/auth/login",
            json={"username": "nonexistent", "password": "password"},
        )
        assert response.status_code == 401

    def test_login_demo_credentials(self):
        response = client.post(
            "/auth/login",
            json={"username": "demo", "password": "demo123"},
        )
        assert response.status_code == 200

    def test_chat_requires_auth(self):
        response = client.post(
            "/chat/query",
            json={"query": "What is the heart?"},
        )
        assert response.status_code == 401

    def test_documents_requires_auth(self):
        response = client.get("/documents/list")
        assert response.status_code == 401

    def test_invalid_token_rejected(self):
        response = client.post(
            "/chat/query",
            json={"query": "test"},
            headers={"Authorization": "Bearer invalid.token.here"},
        )
        assert response.status_code == 401

    def _get_token(self) -> str:
        response = client.post(
            "/auth/login",
            json={"username": "admin", "password": "admin123"},
        )
        return response.json()["access_token"]


class TestChatEndpoint:
    """Tests for the RAG chat endpoint."""

    def _get_auth_headers(self) -> dict:
        response = client.post(
            "/auth/login",
            json={"username": "admin", "password": "admin123"},
        )
        token = response.json()["access_token"]
        return {"Authorization": f"Bearer {token}"}

    @patch("backend.api.chat_routes.run_rag_pipeline")
    def test_chat_query_success(self, mock_pipeline):
        mock_pipeline.return_value = {
            "answer": "ACE inhibitors block the angiotensin-converting enzyme.",
            "sources": [
                {
                    "chunk_id": "c001",
                    "document_name": "cardiology.pdf",
                    "page_number": 12,
                    "score": 0.92,
                    "excerpt": "ACE inhibitors are commonly used in cardiovascular medicine.",
                }
            ],
            "confidence": 0.92,
            "query_type": "MEDICAL_EDUCATION",
            "blocked": False,
            "blocked_by": None,
            "safety_note": "Educational information only.",
        }
        headers = self._get_auth_headers()
        response = client.post(
            "/chat/query",
            json={"query": "What are ACE inhibitors?"},
            headers=headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "confidence" in data
        assert "safety_note" in data
        assert data["blocked"] is False

    @patch("backend.api.chat_routes.run_rag_pipeline")
    def test_chat_blocked_response(self, mock_pipeline):
        mock_pipeline.return_value = {
            "answer": "This assistant provides educational information only.",
            "sources": [],
            "confidence": 0.0,
            "blocked": True,
            "blocked_by": "input_guardrail",
            "safety_note": "Not medical advice.",
        }
        headers = self._get_auth_headers()
        response = client.post(
            "/chat/query",
            json={"query": "Diagnose my chest pain"},
            headers=headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["blocked"] is True

    def test_chat_empty_query_rejected(self):
        headers = self._get_auth_headers()
        response = client.post(
            "/chat/query",
            json={"query": ""},
            headers=headers,
        )
        assert response.status_code == 422  # Pydantic validation error

    def test_chat_too_long_query_rejected(self):
        headers = self._get_auth_headers()
        response = client.post(
            "/chat/query",
            json={"query": "a" * 2001},
            headers=headers,
        )
        assert response.status_code == 422


class TestDocumentEndpoints:
    """Tests for document upload and management."""

    def _get_auth_headers(self) -> dict:
        response = client.post(
            "/auth/login",
            json={"username": "admin", "password": "admin123"},
        )
        token = response.json()["access_token"]
        return {"Authorization": f"Bearer {token}"}

    def test_list_documents_empty(self):
        headers = self._get_auth_headers()
        response = client.get("/documents/list", headers=headers)
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_upload_invalid_file_type(self):
        headers = self._get_auth_headers()
        response = client.post(
            "/documents/upload",
            files={"file": ("malware.exe", b"binary content", "application/octet-stream")},
            headers=headers,
        )
        assert response.status_code == 400

    @patch("backend.api.document_routes.process_document")
    @patch("backend.api.document_routes.index_document")
    def test_upload_valid_txt(self, mock_index, mock_process):
        from backend.rag.document_processor import ProcessedDocument, DocumentChunk
        mock_process.return_value = ProcessedDocument(
            doc_id="abc123",
            filename="test.txt",
            total_chunks=5,
            chunks=[
                DocumentChunk(
                    chunk_id=f"c{i}",
                    text=f"Medical content {i}",
                    metadata={"doc_id": "abc123", "filename": "test.txt", "page": 1, "chunk_index": i},
                )
                for i in range(5)
            ],
        )
        mock_index.return_value = 5

        headers = self._get_auth_headers()
        content = b"Medical education content about the heart and circulatory system. " * 20
        response = client.post(
            "/documents/upload",
            files={"file": ("medical_notes.txt", content, "text/plain")},
            headers=headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert "doc_id" in data
        assert "total_chunks" in data
        assert data["total_chunks"] == 5

    def test_delete_nonexistent_document(self):
        headers = self._get_auth_headers()
        response = client.delete("/documents/nonexistent-doc-id", headers=headers)
        assert response.status_code == 404
