"""
LangGraph multi-agent workflow nodes.
Supports three modes:
  - WEB_ONLY   : no documents uploaded → Tavily + BeautifulSoup
  - DOC_ONLY   : user explicitly filters to a document
  - HYBRID     : documents exist → try docs first, supplement with web
"""
from typing import TypedDict, Optional, List, Any
from loguru import logger

from backend.llm.groq_llm import (
    call_llm,
    ROUTER_SYSTEM_PROMPT,
    ANSWER_SYSTEM_PROMPT,
    ANSWER_WEB_SYSTEM_PROMPT,
    SAFETY_CHECK_SYSTEM_PROMPT,
)
from backend.guardrails.medical_guardrails import (
    check_input_guardrails,
    check_output_guardrails,
    check_retrieval_guardrails,
    check_hallucination_guardrails,
    add_safety_disclaimer,
    get_blocked_response,
)
from backend.vectorstore.pinecone_store import search_documents
from backend.rag.retriever import build_retrieval_result
# web_search imported lazily inside retrieval_agent to avoid startup model load


# ── State ────────────────────────────────────────────────────────
class GraphState(TypedDict):
    query:            str
    user_id:          Optional[str]
    doc_filter:       Optional[str]
    has_documents:    bool           # True if any docs are indexed

    query_type:       Optional[str]
    search_mode:      Optional[str]  # WEB_ONLY | DOC_ONLY | HYBRID

    retrieval_result: Optional[Any]
    web_results:      Optional[List[dict]]
    context:          Optional[str]

    is_safe:          bool
    safety_reason:    Optional[str]
    blocked_by:       Optional[str]

    answer:           Optional[str]
    citations:        Optional[List[dict]]
    confidence:       Optional[float]

    final_response:   Optional[dict]
    error:            Optional[str]


# ── Agent 1: Query Router ────────────────────────────────────────
def query_router_agent(state: GraphState) -> GraphState:
    query = state["query"]
    logger.info(f"[Router] query={query[:80]!r}")

    # Fast regex safety check first
    gr = check_input_guardrails(query)
    if not gr.is_safe:
        return {**state,
                "query_type": "UNSAFE_MEDICAL_REQUEST",
                "is_safe": False,
                "safety_reason": gr.reason,
                "blocked_by": gr.blocked_by}

    # Decide search mode
    has_docs   = state.get("has_documents", False)
    doc_filter = state.get("doc_filter")

    if doc_filter:
        search_mode = "DOC_ONLY"
    elif has_docs:
        search_mode = "HYBRID"
    else:
        search_mode = "WEB_ONLY"

    # LLM routing
    try:
        query_type = call_llm(ROUTER_SYSTEM_PROMPT, query).strip().upper()
        valid = {"MEDICAL_EDUCATION", "DOCUMENT_QUESTION", "GENERAL_QUESTION",
                 "UNSAFE_MEDICAL_REQUEST"}
        if query_type not in valid:
            query_type = "MEDICAL_EDUCATION"
    except Exception as e:
        logger.error(f"[Router] LLM error: {e}")
        query_type = "MEDICAL_EDUCATION"

    is_safe = query_type != "UNSAFE_MEDICAL_REQUEST"
    logger.info(f"[Router] type={query_type} mode={search_mode}")
    return {**state,
            "query_type":   query_type,
            "search_mode":  search_mode,
            "is_safe":      is_safe,
            "safety_reason": None if is_safe else "Unsafe query type",
            "blocked_by":   None if is_safe else "router_agent"}


# ── Agent 2: Safety Agent ────────────────────────────────────────
def safety_agent(state: GraphState) -> GraphState:
    if not state.get("is_safe", True):
        return state
    query = state["query"]
    try:
        result = call_llm(SAFETY_CHECK_SYSTEM_PROMPT, query).strip()
        if result.upper().startswith("UNSAFE"):
            reason = result[7:].strip() or "Clinical advice request detected"
            logger.warning(f"[Safety] Blocked: {reason}")
            return {**state, "is_safe": False,
                    "safety_reason": reason, "blocked_by": "safety_agent"}
    except Exception as e:
        logger.error(f"[Safety] LLM error: {e} — failing open")
    return {**state, "is_safe": True}


# ── Agent 3: Retrieval Agent ─────────────────────────────────────
def retrieval_agent(state: GraphState) -> GraphState:
    if not state.get("is_safe", True):
        return state

    query       = state["query"]
    mode        = state.get("search_mode", "WEB_ONLY")
    doc_filter  = state.get("doc_filter")

    doc_context  = ""
    web_context  = ""
    citations    = []
    confidence   = 0.0
    web_results_dicts = []

    # ── Document retrieval ───────────────────────────────────────
    if mode in ("DOC_ONLY", "HYBRID"):
        try:
            raw = search_documents(query=query, top_k=15, filter_doc_id=doc_filter)
            if raw:
                rr = build_retrieval_result(query=query, raw_chunks=raw, top_k=5)
                doc_context = rr.context
                citations   = [c.model_dump() for c in rr.citations]
                confidence  = rr.confidence
                logger.info(f"[Retrieval] docs: {len(rr.chunks)} chunks, conf={confidence:.2f}")
            else:
                logger.info("[Retrieval] No document chunks found")
        except Exception as e:
            logger.error(f"[Retrieval] Doc search error: {e}")

        # If DOC_ONLY and nothing found, block
        if mode == "DOC_ONLY" and not doc_context:
            return {**state, "is_safe": False,
                    "safety_reason": "No relevant content found in the selected document.",
                    "blocked_by": "retrieval_guardrail"}

    # ── Web search ────────────────────────────────────────────────
    if mode in ("WEB_ONLY", "HYBRID"):
        # For HYBRID only supplement if doc confidence is low
        if mode == "HYBRID" and confidence >= 0.5:
            logger.info("[Retrieval] HYBRID: doc confidence sufficient, skipping web")
        else:
            try:
                # Import here to get the pre-search query check result
                from backend.rag.web_search import (
                    web_search_medical, format_web_context,
                    check_query_for_web_search,
                )
                # Run pre-search guardrail explicitly so we can surface it
                web_safe, web_reason = check_query_for_web_search(query)
                if not web_safe:
                    # If we have no doc context either, block entirely
                    if not doc_context:
                        return {**state, "is_safe": False,
                                "safety_reason": web_reason,
                                "blocked_by": "web_search_guardrail"}
                    # If we have doc context, just skip web and proceed with docs
                    logger.warning(f"[Retrieval] Web search blocked ({web_reason}), using docs only")
                else:
                    web_results = web_search_medical(query, max_results=5)
                    if web_results:
                        web_context = format_web_context(web_results)
                        web_results_dicts = [r.to_dict() for r in web_results]
                        if not confidence:
                            confidence = 0.65
                        logger.info(f"[Retrieval] web: {len(web_results)} safe results")
                    else:
                        logger.warning("[Retrieval] No web results returned")
            except Exception as e:
                logger.error(f"[Retrieval] Web search error: {e}")

    # ── Merge context ─────────────────────────────────────────────
    parts = []
    if doc_context:
        parts.append("## From your documents:\n" + doc_context)
    if web_context:
        parts.append("## From web sources:\n" + web_context)
    merged_context = "\n\n".join(parts)

    if not merged_context:
        return {**state, "is_safe": False,
                "safety_reason": ("No information could be retrieved from documents "
                                  "or the web for this query."),
                "blocked_by": "retrieval_guardrail"}

    return {**state,
            "retrieval_result":    None,
            "web_results":         web_results_dicts,
            "context":             merged_context,
            "citations":           citations + web_results_dicts,
            "confidence":          confidence}


# ── Agent 4: Answer Agent ─────────────────────────────────────────
def answer_agent(state: GraphState) -> GraphState:
    if not state.get("is_safe", True):
        return state

    query    = state["query"]
    context  = state.get("context", "")
    mode     = state.get("search_mode", "WEB_ONLY")

    if not context:
        return {**state, "is_safe": False,
                "safety_reason": "No context available for answer generation",
                "blocked_by": "answer_agent"}

    # Choose system prompt based on mode
    system_prompt = (ANSWER_WEB_SYSTEM_PROMPT if mode == "WEB_ONLY"
                     else ANSWER_SYSTEM_PROMPT)

    user_message = (
        f"Context:\n{context}\n\n"
        f"---\nUser question: {query}\n\n"
        "Answer ONLY based on the context above. "
        "If context is insufficient, say so clearly. "
        "Never fabricate medical information."
    )

    try:
        answer = call_llm(system_prompt, user_message)

        # Hallucination check
        context_chunks = [context]
        hall = check_hallucination_guardrails(answer, context_chunks)
        if not hall.is_safe:
            return {**state, "is_safe": False,
                    "safety_reason": hall.reason, "blocked_by": "hallucination_guardrail"}

        # Output guardrail
        out = check_output_guardrails(answer)
        if not out.is_safe:
            return {**state, "is_safe": False,
                    "safety_reason": out.reason, "blocked_by": "output_guardrail"}

        answer = add_safety_disclaimer(answer)
        logger.info("[Answer] Generated successfully")
        return {**state, "answer": answer}

    except Exception as e:
        logger.error(f"[Answer] Error: {e}")
        return {**state, "is_safe": False,
                "safety_reason": f"Answer generation error: {e}",
                "blocked_by": "answer_agent"}


# ── Final response builder ────────────────────────────────────────
def build_final_response(state: GraphState) -> GraphState:
    if not state.get("is_safe", True):
        return {**state,
                "final_response": get_blocked_response(
                    state.get("safety_reason", "Unknown"),
                    state.get("blocked_by", "safety_agent"))}

    return {**state, "final_response": {
        "answer":      state.get("answer", "No answer generated."),
        "sources":     state.get("citations", []),
        "confidence":  state.get("confidence", 0.0),
        "query_type":  state.get("query_type", "UNKNOWN"),
        "search_mode": state.get("search_mode", "WEB_ONLY"),
        "blocked":     False,
        "blocked_by":  None,
        "safety_note": ("Educational information only. Not medical advice. "
                        "Consult a qualified healthcare professional."),
    }}
