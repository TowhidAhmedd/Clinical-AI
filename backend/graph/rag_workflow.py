"""
LangGraph workflow: routes between WEB_ONLY, DOC_ONLY, HYBRID modes.
"""
from typing import Optional
from loguru import logger

try:
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False

from backend.agents.workflow_agents import (
    GraphState, query_router_agent, safety_agent,
    retrieval_agent, answer_agent, build_final_response,
)
from backend.vectorstore.pinecone_store import get_vector_store


def _has_documents() -> bool:
    """Check if the vector store has any indexed documents."""
    try:
        store = get_vector_store()
        docs = store.list_documents()
        return len(docs) > 0
    except Exception:
        return False


def should_continue(state: GraphState) -> str:
    return "build_final_response" if not state.get("is_safe", True) else "continue"


def build_workflow():
    if not LANGGRAPH_AVAILABLE:
        return SequentialWorkflow()

    g = StateGraph(GraphState)
    g.add_node("query_router",          query_router_agent)
    g.add_node("safety_check",          safety_agent)
    g.add_node("retrieval",             retrieval_agent)
    g.add_node("answer",                answer_agent)
    g.add_node("build_final_response",  build_final_response)

    g.set_entry_point("query_router")

    for src, nxt in [("query_router", "safety_check"),
                     ("safety_check",  "retrieval"),
                     ("retrieval",     "answer"),
                     ("answer",        "build_final_response")]:
        g.add_conditional_edges(src, should_continue,
                                {"continue": nxt,
                                 "build_final_response": "build_final_response"})
    g.add_edge("build_final_response", END)
    return g.compile()


class SequentialWorkflow:
    def invoke(self, state):
        state = query_router_agent(state)
        if state.get("is_safe", True): state = safety_agent(state)
        if state.get("is_safe", True): state = retrieval_agent(state)
        if state.get("is_safe", True): state = answer_agent(state)
        return build_final_response(state)


_workflow = None

def get_workflow():
    global _workflow
    if _workflow is None:
        _workflow = build_workflow()
        logger.info("Workflow compiled")
    return _workflow


def run_rag_pipeline(query: str,
                     user_id: Optional[str] = None,
                     doc_filter: Optional[str] = None) -> dict:
    has_docs = _has_documents()
    logger.info(f"Pipeline: query={query[:60]!r} has_docs={has_docs} doc_filter={doc_filter!r}")

    initial: GraphState = {
        "query":            query,
        "user_id":          user_id,
        "doc_filter":       doc_filter,
        "has_documents":    has_docs,
        "query_type":       None,
        "search_mode":      None,
        "retrieval_result": None,
        "web_results":      None,
        "context":          None,
        "is_safe":          True,
        "safety_reason":    None,
        "blocked_by":       None,
        "answer":           None,
        "citations":        None,
        "confidence":       None,
        "final_response":   None,
        "error":            None,
    }
    try:
        result = get_workflow().invoke(initial)
        return result.get("final_response",
                          {"answer": "Error: no response", "blocked": True})
    except Exception as e:
        logger.error(f"Workflow error: {e}")
        return {"answer": f"System error: {e}", "sources": [],
                "confidence": 0.0, "blocked": True, "blocked_by": "system_error",
                "safety_note": "System error. Please try again."}
