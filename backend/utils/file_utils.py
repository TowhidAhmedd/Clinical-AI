"""
Utility functions: file handling, text utilities, validators.
"""
import os
import uuid
import hashlib
from pathlib import Path
from typing import Optional
from loguru import logger


DATA_DIR = Path("data")
UPLOADS_DIR = DATA_DIR / "uploads"
PROCESSED_DIR = DATA_DIR / "processed"


def ensure_data_dirs():
    """Create data directories if they don't exist."""
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(exist_ok=True)


def save_uploaded_file(file_content: bytes, filename: str) -> tuple[str, str]:
    """
    Save an uploaded file to disk.
    Returns (file_path, doc_id).
    """
    ensure_data_dirs()
    doc_id = hashlib.md5(f"{filename}:{len(file_content)}:{uuid.uuid4()}".encode()).hexdigest()[:16]
    safe_filename = f"{doc_id}_{filename}"
    file_path = str(UPLOADS_DIR / safe_filename)
    with open(file_path, "wb") as f:
        f.write(file_content)
    logger.info(f"Saved uploaded file: {safe_filename}")
    return file_path, doc_id


def delete_uploaded_file(file_path: str):
    """Delete a file from disk if it exists."""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Deleted file: {file_path}")
    except Exception as e:
        logger.error(f"Failed to delete file {file_path}: {e}")


def format_response_for_display(response: dict) -> dict:
    """Clean up the response dict for API output."""
    return {
        "answer": response.get("answer", ""),
        "sources": response.get("sources", []),
        "confidence": round(float(response.get("confidence", 0.0)), 3),
        "query_type": response.get("query_type", "UNKNOWN"),
        "blocked": response.get("blocked", False),
        "blocked_by": response.get("blocked_by"),
        "safety_note": response.get(
            "safety_note",
            "This is educational information only. Not medical advice.",
        ),
    }


def truncate_text(text: str, max_length: int = 500) -> str:
    """Truncate text to max_length, adding ellipsis if needed."""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."
