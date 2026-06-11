"""
LLM module: Groq-backed LangChain LLM with retry logic and mock fallback.
"""
from functools import lru_cache
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

try:
    from langchain_groq import ChatGroq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

from langchain_core.messages import HumanMessage, SystemMessage
from backend.config import get_settings

settings = get_settings()


class MockLLM:
    def invoke(self, messages):
        return type("R", (), {"content": (
            "⚠️ [Mock LLM] GROQ_API_KEY not configured. "
            "Set it in your .env file to get real answers.")})()
    async def ainvoke(self, messages):
        return self.invoke(messages)


@lru_cache(maxsize=1)
def get_llm():
    if not GROQ_AVAILABLE or not settings.GROQ_API_KEY:
        logger.warning("Using MockLLM — set GROQ_API_KEY in .env")
        return MockLLM()
    try:
        llm = ChatGroq(model=settings.GROQ_MODEL,
                       api_key=settings.GROQ_API_KEY,
                       temperature=0.1, max_tokens=2048)
        logger.info(f"ChatGroq ready: {settings.GROQ_MODEL}")
        return llm
    except Exception as e:
        logger.error(f"ChatGroq init failed: {e}")
        return MockLLM()


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def call_llm(system_prompt: str, user_message: str) -> str:
    llm = get_llm()
    resp = llm.invoke([SystemMessage(content=system_prompt),
                       HumanMessage(content=user_message)])
    return resp.content


# ── System Prompts ────────────────────────────────────────────────
ROUTER_SYSTEM_PROMPT = """You are a query classifier for a clinical medical education assistant.
Classify the query into exactly ONE of:
MEDICAL_EDUCATION  — general medical education (anatomy, physiology, disease info, pharmacology)
DOCUMENT_QUESTION  — question about an uploaded document
GENERAL_QUESTION   — non-medical general question
UNSAFE_MEDICAL_REQUEST — requests for diagnosis, prescription, dosage, treatment plan, emergency advice
Respond with ONLY the category name."""

SAFETY_CHECK_SYSTEM_PROMPT = """You are a medical safety agent.
Determine if the query requests: diagnosis, specific medication recommendations,
dosage instructions, a treatment plan, or emergency medical advice for a specific patient.
Respond:
  SAFE   — general medical education question
  UNSAFE: <reason>  — clinical/patient-specific advice request
Respond ONLY in one of these two formats."""

ANSWER_SYSTEM_PROMPT = """You are a clinical medical education assistant helping students
and healthcare learners understand medical concepts.

STRICT RULES:
1. Answer ONLY using the provided context. Never add outside information.
2. Never diagnose, prescribe, recommend dosages, or create treatment plans.
3. Cite source documents in your answer.
4. If context is insufficient, say clearly what is missing.
5. Be educational, clear, and factual.

Format:
## Answer
[Educational answer strictly from context]

## Key Points
- [2-3 bullet takeaways]

## Sources Used
[List the source names you drew from]"""

ANSWER_WEB_SYSTEM_PROMPT = """You are a clinical medical education assistant.
You have been given information retrieved from trusted medical education websites.

STRICT RULES:
1. Answer ONLY using the provided web context.
2. Never diagnose, prescribe, recommend dosages, or create treatment plans.
3. Mention the web sources you used.
4. Be educational, clear, and factual.
5. If the context doesn't answer the question, say so.

Format:
## Answer
[Educational answer from retrieved web content]

## Key Points
- [2-3 bullet takeaways]

## Sources
[List the web sources / URLs used]"""
