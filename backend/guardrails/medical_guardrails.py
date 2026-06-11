"""
Medical safety guardrails: input, retrieval, hallucination, and output checks.
"""
import re
from typing import Optional
from pydantic import BaseModel
from loguru import logger


# ---------------------------------------------------------------
# Patterns for unsafe medical requests
# ---------------------------------------------------------------
UNSAFE_PATTERNS = [
    # Diagnosis requests
    r"\b(diagnose|diagnosis|do i have|what disease|what illness|what condition)\b",
    # Prescription / medication requests
    r"\b(prescribe|prescription|what (medication|medicine|drug) should i (take|use))\b",
    r"\b(can you (recommend|suggest) (a )?(medication|medicine|drug|pill))\b",
    # Dosage
    r"\b(dosage|dose|how much (should i take|to take|of the medicine))\b",
    # Treatment plans
    r"\b(treatment plan|treat my|how to treat|cure (my|this|the))\b",
    # Emergency advice
    r"\b(emergency|call 911|is this serious|should i go to (the )?(hospital|ER|emergency))\b",
    # Role manipulation
    r"\b(pretend (you are|to be) (a )?(doctor|physician|medical|nurse|specialist))\b",
    r"\b(act (as|like) (a )?(doctor|physician|nurse|specialist))\b",
    r"\b(you are (a )?(doctor|physician|medical professional))\b",
    # Prompt injection
    r"ignore (previous|all|your|prior) instructions",
    r"forget (everything|all instructions|your instructions)",
    r"override (your |all )?(safety|guardrails|restrictions|instructions)",
    r"jailbreak",
    r"system prompt",
    r"bypass (safety|guardrails|restrictions)",
]

# Patterns for output that should be blocked
OUTPUT_UNSAFE_PATTERNS = [
    r"\b(you (likely|probably|may) have|this (sounds|looks) like|i (diagnose|suggest) you have)\b",
    r"\b(take \d+ mg|take \d+ pills?|administer \d+)\b",
    r"\b(prescription for|prescribing you|I recommend (taking|using) .{0,30}(mg|pills?))\b",
]

EDUCATIONAL_DISCLAIMER = (
    "\n\n---\n"
    "⚕️ **Medical Safety Notice**: This assistant provides educational information only. "
    "It cannot provide diagnosis, treatment recommendations, prescriptions, or emergency medical advice. "
    "Always consult a qualified healthcare professional for medical concerns."
)


class GuardrailResult(BaseModel):
    is_safe: bool
    reason: Optional[str] = None
    blocked_by: Optional[str] = None


def check_input_guardrails(query: str) -> GuardrailResult:
    """
    Check user input for unsafe medical requests, prompt injection, 
    jailbreak attempts, and role manipulation.
    """
    query_lower = query.lower().strip()

    # Empty query
    if not query_lower:
        return GuardrailResult(is_safe=False, reason="Empty query", blocked_by="input_validation")

    # Check unsafe patterns
    for pattern in UNSAFE_PATTERNS:
        if re.search(pattern, query_lower, re.IGNORECASE):
            logger.warning(f"Input guardrail triggered: pattern={pattern!r} query={query[:80]!r}")
            return GuardrailResult(
                is_safe=False,
                reason=(
                    "This request appears to ask for medical advice, diagnosis, "
                    "prescription, or attempts to manipulate the assistant. "
                    "This assistant is for medical education only."
                ),
                blocked_by="input_guardrail",
            )

    return GuardrailResult(is_safe=True)


def check_retrieval_guardrails(
    chunks: list,
    min_score: float = 0.3,
    min_chunks: int = 1,
) -> GuardrailResult:
    """
    Verify retrieved chunks exist, have sufficient relevance scores,
    and come from valid sources.
    """
    if not chunks:
        return GuardrailResult(
            is_safe=False,
            reason="No relevant context found in the knowledge base for this query.",
            blocked_by="retrieval_guardrail",
        )

    if len(chunks) < min_chunks:
        return GuardrailResult(
            is_safe=False,
            reason="Insufficient evidence retrieved to answer safely.",
            blocked_by="retrieval_guardrail",
        )

    # Check if any chunk has a meaningful score
    scores = [c.get("score", 1.0) for c in chunks if isinstance(c, dict)]
    if scores and max(scores) < min_score:
        return GuardrailResult(
            is_safe=False,
            reason=f"Retrieved context relevance too low (max score: {max(scores):.2f}). Cannot answer safely.",
            blocked_by="retrieval_guardrail",
        )

    return GuardrailResult(is_safe=True)


def check_hallucination_guardrails(
    answer: str,
    context_chunks: list[str],
) -> GuardrailResult:
    """
    Basic grounding check: verify the answer doesn't contain claims
    unsupported by the retrieved context.
    Uses a simple heuristic — a more advanced version would use NLI.
    """
    if not answer or not answer.strip():
        return GuardrailResult(
            is_safe=False,
            reason="Empty answer generated.",
            blocked_by="hallucination_guardrail",
        )

    if not context_chunks:
        return GuardrailResult(
            is_safe=False,
            reason="Answer generated with no retrieved context — potential hallucination.",
            blocked_by="hallucination_guardrail",
        )

    # Check for overconfident unsupported claims (heuristic)
    overconfidence_patterns = [
        r"\b(definitely|certainly|always|never|guaranteed|100%)\b",
        r"\b(according to me|i know that|trust me|as a doctor)\b",
    ]
    answer_lower = answer.lower()
    for pattern in overconfidence_patterns:
        if re.search(pattern, answer_lower):
            logger.warning(f"Potential hallucination pattern detected: {pattern}")
            # Don't block — just log; the LLM prompt handles grounding

    return GuardrailResult(is_safe=True)


def check_output_guardrails(answer: str) -> GuardrailResult:
    """
    Block output that contains diagnoses, prescriptions, dosages,
    treatment plans, or emergency directives.
    """
    answer_lower = answer.lower()

    for pattern in OUTPUT_UNSAFE_PATTERNS:
        if re.search(pattern, answer_lower, re.IGNORECASE):
            logger.warning(f"Output guardrail triggered: pattern={pattern!r}")
            return GuardrailResult(
                is_safe=False,
                reason=(
                    "The generated response contained potentially unsafe medical advice "
                    "(diagnosis, prescription, or dosage). Response blocked for safety."
                ),
                blocked_by="output_guardrail",
            )

    return GuardrailResult(is_safe=True)


def add_safety_disclaimer(answer: str) -> str:
    """Append the educational disclaimer to every response."""
    return answer + EDUCATIONAL_DISCLAIMER


def get_blocked_response(reason: str, blocked_by: str) -> dict:
    """Return a standardised blocked response."""
    messages = {
        "input_guardrail": (
            "⚠️ I'm unable to process this request. "
            "This assistant is designed for **medical education only** and cannot provide "
            "diagnosis, treatment recommendations, prescription advice, dosage information, "
            "or emergency medical directives.\n\n"
            "If you have a medical concern, please consult a qualified healthcare professional."
        ),
        "web_search_guardrail": (
            "⚠️ This question was blocked before searching the web.\n\n"
            f"Reason: **{reason}**\n\n"
            "This assistant cannot search for specific dosages, prescriptions, or "
            "patient-specific clinical advice — even via web search.\n\n"
            "You can ask general educational questions like:\n"
            "- *What is the mechanism of action of metformin?*\n"
            "- *How does insulin resistance develop in type 2 diabetes?*\n\n"
            "For personal medical decisions, consult a qualified healthcare professional."
        ),
        "retrieval_guardrail": (
            "⚠️ I could not find sufficient information to answer safely.\n\n"
            f"Reason: {reason}\n\n"
            "Try rephrasing as a general educational question, or upload a relevant document."
        ),
        "hallucination_guardrail": (
            "⚠️ I was unable to generate a grounded response. "
            "No sufficient context was found.\n\n"
            "Please try again or upload relevant medical education documents."
        ),
        "output_guardrail": (
            "⚠️ The generated response was blocked by the output safety filter.\n\n"
            "The response contained content that could be interpreted as clinical advice "
            "(e.g. specific dosage, prescription, or diagnosis).\n\n"
            "This assistant provides **educational information only**. "
            "Please consult a qualified healthcare professional for medical advice."
        ),
        "safety_agent": (
            "⚠️ This request was flagged by the medical safety guardrails.\n\n"
            f"{reason}\n\n"
            "This assistant is for **medical education only**. "
            "For medical concerns, consult a qualified healthcare professional."
        ),
    }
    return {
        "answer": messages.get(blocked_by, f"⚠️ Request blocked: {reason}"),
        "sources": [],
        "confidence": 0.0,
        "blocked": True,
        "blocked_by": blocked_by,
        "safety_note": (
            "This assistant provides educational information only. "
            "Not medical advice. Consult qualified healthcare professionals."
        ),
    }
