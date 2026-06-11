"""
Observability module: LangSmith tracing configuration and custom logging.
"""
import os
import time
import uuid
from contextlib import contextmanager
from typing import Optional
from loguru import logger

from backend.config import get_settings

settings = get_settings()


def configure_langsmith():
    """Configure LangSmith tracing if enabled."""
    if settings.LANGCHAIN_TRACING_V2 and settings.LANGCHAIN_API_KEY:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = settings.LANGCHAIN_API_KEY
        os.environ["LANGCHAIN_PROJECT"] = settings.LANGCHAIN_PROJECT
        os.environ["LANGCHAIN_ENDPOINT"] = settings.LANGCHAIN_ENDPOINT
        logger.info(f"LangSmith tracing enabled: project={settings.LANGCHAIN_PROJECT}")
    else:
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        logger.info("LangSmith tracing disabled")


def configure_logging():
    """Configure application logging."""
    logger.remove()
    logger.add(
        "logs/app.log",
        rotation="10 MB",
        retention="7 days",
        level=settings.LOG_LEVEL,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
    )
    logger.add(
        lambda msg: print(msg, end=""),
        colorize=True,
        level=settings.LOG_LEVEL,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> | {message}",
    )


class RAGTracer:
    """
    Simple trace collector for RAG pipeline events.
    Logs to LangSmith if configured, otherwise to local logs.
    """

    def __init__(self):
        self.trace_id = str(uuid.uuid4())[:8]

    def log_query(self, query: str, user_id: Optional[str] = None):
        logger.info(f"[Trace:{self.trace_id}] Query: {query[:100]!r} | User: {user_id}")

    def log_router(self, query_type: str, is_safe: bool):
        logger.info(f"[Trace:{self.trace_id}] Router: type={query_type}, safe={is_safe}")

    def log_retrieval(self, num_chunks: int, confidence: float, latency_ms: float):
        logger.info(
            f"[Trace:{self.trace_id}] Retrieval: chunks={num_chunks}, "
            f"confidence={confidence:.3f}, latency={latency_ms:.0f}ms"
        )

    def log_safety(self, is_safe: bool, reason: Optional[str] = None):
        if is_safe:
            logger.info(f"[Trace:{self.trace_id}] Safety: PASSED")
        else:
            logger.warning(f"[Trace:{self.trace_id}] Safety: BLOCKED — {reason}")

    def log_answer(self, answer_length: int, latency_ms: float):
        logger.info(
            f"[Trace:{self.trace_id}] Answer: length={answer_length}, latency={latency_ms:.0f}ms"
        )

    def log_error(self, error: str, stage: str):
        logger.error(f"[Trace:{self.trace_id}] Error at {stage}: {error}")


@contextmanager
def timed(label: str):
    """Context manager to measure and log execution time."""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.debug(f"[TIMING] {label}: {elapsed_ms:.1f}ms")
