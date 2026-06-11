"""
API client for the Clinical RAG Assistant backend.
Timeouts are generous because:
  - login: bcrypt verify is slow on first call (~2-3s on Windows)
  - chat:  LLM inference + retrieval can take 30-60s
  - upload: embedding model loads on first use (can take 60-120s first time)
  - health: fast, but backend may be starting up
"""
import os
import time
import requests
from typing import Optional

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# Timeout constants (seconds)
TIMEOUT_LOGIN   = 60    # bcrypt + any first-startup overhead
TIMEOUT_HEALTH  = 10
TIMEOUT_CHAT    = 180   # LLM can be slow; first call loads embedding model
TIMEOUT_UPLOAD  = 300   # First upload triggers embedding model download
TIMEOUT_LIST    = 30
TIMEOUT_DELETE  = 30


class APIClient:
    def __init__(self, base_url: str = BACKEND_URL):
        self.base_url = base_url.rstrip("/")
        self.token: Optional[str] = None

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def login(self, username: str, password: str) -> dict:
        """Authenticate and store JWT token."""
        resp = requests.post(
            f"{self.base_url}/auth/login",
            json={"username": username, "password": password},
            timeout=TIMEOUT_LOGIN,
        )
        resp.raise_for_status()
        data = resp.json()
        self.token = data["access_token"]
        return data

    def health(self) -> dict:
        resp = requests.get(
            f"{self.base_url}/health",
            timeout=TIMEOUT_HEALTH,
        )
        resp.raise_for_status()
        return resp.json()

    def wait_for_backend(self, retries: int = 10, delay: float = 3.0) -> bool:
        """
        Poll /health until the backend is up.
        Returns True if backend is reachable, False after retries exhausted.
        Useful to call once on app startup.
        """
        for attempt in range(1, retries + 1):
            try:
                self.health()
                return True
            except Exception:
                if attempt < retries:
                    time.sleep(delay)
        return False

    def chat(self, query: str, doc_filter: Optional[str] = None) -> dict:
        """Send a query to the RAG pipeline."""
        payload = {"query": query}
        if doc_filter:
            payload["doc_filter"] = doc_filter
        resp = requests.post(
            f"{self.base_url}/chat/query",
            json=payload,
            headers=self._headers(),
            timeout=TIMEOUT_CHAT,
        )
        resp.raise_for_status()
        return resp.json()

    def upload_document(self, file_bytes: bytes, filename: str) -> dict:
        """Upload a document for indexing."""
        resp = requests.post(
            f"{self.base_url}/documents/upload",
            files={"file": (filename, file_bytes)},
            headers={"Authorization": f"Bearer {self.token}"},
            timeout=TIMEOUT_UPLOAD,
        )
        resp.raise_for_status()
        return resp.json()

    def list_documents(self) -> list:
        resp = requests.get(
            f"{self.base_url}/documents/list",
            headers=self._headers(),
            timeout=TIMEOUT_LIST,
        )
        resp.raise_for_status()
        return resp.json()

    def delete_document(self, doc_id: str) -> dict:
        resp = requests.delete(
            f"{self.base_url}/documents/{doc_id}",
            headers=self._headers(),
            timeout=TIMEOUT_DELETE,
        )
        resp.raise_for_status()
        return resp.json()
