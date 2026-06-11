"""Chat routes — main RAG/web-search query endpoint."""
import time
from fastapi import APIRouter, Depends, HTTPException
from loguru import logger

from backend.api.schemas import ChatRequest, ChatResponse, CitationModel
from backend.security.auth import get_current_user
from backend.graph.rag_workflow import run_rag_pipeline
from backend.utils.file_utils import format_response_for_display

router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post("/query", response_model=ChatResponse)
async def query(request: ChatRequest,
                current_user: str = Depends(get_current_user)):
    start = time.perf_counter()
    logger.info(f"Query user={current_user!r}: {request.query[:80]!r}")

    try:
        response = run_rag_pipeline(
            query=request.query,
            user_id=current_user,
            doc_filter=request.doc_filter,
        )
        elapsed = (time.perf_counter() - start) * 1000
        logger.info(f"Query completed in {elapsed:.0f}ms, mode={response.get('search_mode')}")

        cleaned = format_response_for_display(response)

        sources = []
        for src in cleaned.get("sources", []):
            try:
                sources.append(CitationModel(
                    chunk_id      = src.get("chunk_id", ""),
                    document_name = src.get("document_name", src.get("title", "Web Source")),
                    page_number   = src.get("page_number", 0),
                    score         = src.get("score", 0.0),
                    excerpt       = src.get("excerpt", ""),
                    url           = src.get("url"),
                    source_type   = src.get("source_type", "document"),
                ))
            except Exception:
                pass

        return ChatResponse(
            answer      = cleaned["answer"],
            sources     = sources,
            confidence  = cleaned["confidence"],
            query_type  = cleaned.get("query_type"),
            search_mode = response.get("search_mode"),
            blocked     = cleaned.get("blocked", False),
            blocked_by  = cleaned.get("blocked_by"),
            safety_note = cleaned["safety_note"],
        )
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
