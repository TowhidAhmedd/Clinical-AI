"""
Shared pytest fixtures for the Clinical RAG Assistant test suite.
"""
import os
import sys
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set test environment variables before any imports
os.environ.setdefault("JWT_SECRET_KEY", "test-secret-key-for-pytest-only")
os.environ.setdefault("JWT_ALGORITHM", "HS256")
os.environ.setdefault("API_KEY", "test-api-key")
os.environ.setdefault("GROQ_API_KEY", "")
os.environ.setdefault("PINECONE_API_KEY", "")
os.environ.setdefault("APP_ENV", "testing")


@pytest.fixture(scope="session")
def test_settings():
    """Provide test settings."""
    from backend.config import Settings
    return Settings(
        APP_ENV="testing",
        JWT_SECRET_KEY="test-secret-key",
        GROQ_API_KEY="",
        PINECONE_API_KEY="",
    )


@pytest.fixture
def sample_chunks():
    """Provide sample document chunks for testing."""
    from backend.rag.retriever import RetrievedChunk
    return [
        RetrievedChunk(
            chunk_id=f"chunk_{i:03d}",
            text=f"Medical education content about cardiovascular medicine. Point {i}. "
                 f"ACE inhibitors block the angiotensin-converting enzyme.",
            score=0.9 - (i * 0.05),
            metadata={"filename": "cardiology.pdf", "page": i + 1, "doc_id": "doc001"},
        )
        for i in range(5)
    ]


@pytest.fixture
def sample_graph_state():
    """Provide a default GraphState for agent tests."""
    from backend.agents.workflow_agents import GraphState
    return GraphState(
        query="What is the mechanism of action of ACE inhibitors?",
        user_id="test_user",
        doc_filter=None,
        query_type=None,
        retrieval_result=None,
        context=None,
        is_safe=True,
        safety_reason=None,
        blocked_by=None,
        answer=None,
        citations=None,
        confidence=None,
        final_response=None,
        error=None,
    )


@pytest.fixture
def auth_headers(client):
    """Provide authenticated headers for API tests."""
    response = client.post(
        "/auth/login",
        json={"username": "admin", "password": "admin123"},
    )
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture(scope="module")
def client():
    """Provide a FastAPI test client."""
    from fastapi.testclient import TestClient
    from backend.main import app
    return TestClient(app)
