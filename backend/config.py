from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # App
    APP_ENV: str = "development"
    LOG_LEVEL: str = "INFO"
    MAX_FILE_SIZE_MB: int = 50
    ALLOWED_ORIGINS: str = "http://localhost:8501"

    # LLM
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "llama3-70b-8192"

    # Web Search
    TAVILY_API_KEY: str = ""

    # Vector DB
    PINECONE_API_KEY: str = ""
    PINECONE_ENVIRONMENT: str = "us-east-1"
    PINECONE_INDEX_NAME: str = "clinical-rag"

    # Embeddings
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"

    # Security
    JWT_SECRET_KEY: str = "changeme-use-long-random-string-in-production"
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    API_KEY: str = "changeme-api-key"
    BACKEND_URL: str = "http://localhost:8000"

    # LangSmith
    LANGCHAIN_TRACING_V2: bool = False
    LANGCHAIN_API_KEY: str = ""
    LANGCHAIN_PROJECT: str = "clinical-rag-assistant"
    LANGCHAIN_ENDPOINT: str = "https://api.smith.langchain.com"

    model_config = {"env_file": ".env", "extra": "ignore"}


@lru_cache()
def get_settings() -> Settings:
    return Settings()
# force reload
