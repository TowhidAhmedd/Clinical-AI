"""Clinical RAG Assistant — FastAPI Application"""
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from loguru import logger

from backend.config import get_settings
from backend.observability.tracing import configure_langsmith, configure_logging
from backend.utils.file_utils import ensure_data_dirs
from backend.api.auth_routes     import router as auth_router
from backend.api.chat_routes     import router as chat_router
from backend.api.document_routes import router as document_router
from backend.api.schemas import HealthResponse

settings = get_settings()
limiter  = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_data_dirs()
    configure_logging()
    configure_langsmith()
    logger.info("=" * 55)
    logger.info("Clinical RAG Assistant starting")
    logger.info(f"Env:          {settings.APP_ENV}")
    logger.info(f"LLM:          {settings.GROQ_MODEL}")
    logger.info(f"Embeddings:   {settings.EMBEDDING_MODEL}")
    logger.info(f"VectorStore:  {'Pinecone' if settings.PINECONE_API_KEY else 'In-Memory'}")
    logger.info(f"WebSearch:    {'Tavily' if settings.TAVILY_API_KEY else 'Scraping fallback'}")
    logger.info(f"LangSmith:    {'on' if settings.LANGCHAIN_TRACING_V2 else 'off'}")
    logger.info("=" * 55)
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="Clinical RAG Assistant API",
    description=(
        "Medical education AI — answers questions with web search (Tavily) "
        "or uploaded documents (RAG). No document upload required."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled: {exc}")
    return JSONResponse(status_code=500, content={"detail": "Unexpected error. Please try again."})


app.include_router(auth_router)
app.include_router(chat_router)
app.include_router(document_router)


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    from backend.config import get_settings
    s = get_settings()
    web_search = "tavily" if s.TAVILY_API_KEY else "scraping_fallback"
    return HealthResponse(
        status="healthy", version="2.0.0",
        embedding_model=s.EMBEDDING_MODEL,
        vector_store="pinecone" if s.PINECONE_API_KEY else "in-memory",
        llm_model=s.GROQ_MODEL,
        web_search=web_search,
        langsmith_enabled=s.LANGCHAIN_TRACING_V2,
    )


@app.get("/", tags=["Root"])
async def root():
    return {"message": "Clinical RAG Assistant API v2.0",
            "docs": "/docs", "health": "/health"}
