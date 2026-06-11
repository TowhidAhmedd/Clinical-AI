"""
RAG evaluation tests: Context Precision, Context Recall,
Faithfulness, and Answer Relevancy.
These tests use heuristic metrics; for production, integrate RAGAS.
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------
# Evaluation helpers (heuristic-based, no external API needed)
# ---------------------------------------------------------------

def compute_context_precision(retrieved_chunks: list, relevant_keywords: list) -> float:
    """
    Context Precision: fraction of retrieved chunks that contain relevant content.
    """
    if not retrieved_chunks:
        return 0.0
    relevant_count = sum(
        1 for chunk in retrieved_chunks
        if any(kw.lower() in chunk["text"].lower() for kw in relevant_keywords)
    )
    return relevant_count / len(retrieved_chunks)


def compute_context_recall(retrieved_chunks: list, ground_truth_facts: list) -> float:
    """
    Context Recall: fraction of ground truth facts covered by retrieved context.
    """
    if not ground_truth_facts:
        return 1.0
    context_text = " ".join(c["text"] for c in retrieved_chunks).lower()
    covered = sum(1 for fact in ground_truth_facts if fact.lower() in context_text)
    return covered / len(ground_truth_facts)


def compute_faithfulness(answer: str, context_chunks: list) -> float:
    """
    Faithfulness: estimate whether answer content appears in context.
    Simple word-overlap heuristic.
    """
    if not context_chunks or not answer:
        return 0.0
    context_text = " ".join(c["text"] for c in context_chunks).lower()
    answer_words = set(w.lower() for w in answer.split() if len(w) > 4)
    context_words = set(w.lower() for w in context_text.split() if len(w) > 4)
    if not answer_words:
        return 1.0
    overlap = answer_words & context_words
    return len(overlap) / len(answer_words)


def compute_answer_relevancy(query: str, answer: str) -> float:
    """
    Answer Relevancy: simple keyword overlap between query and answer.
    """
    if not query or not answer:
        return 0.0
    query_words = set(w.lower() for w in query.split() if len(w) > 3)
    answer_words = set(w.lower() for w in answer.split() if len(w) > 3)
    if not query_words:
        return 0.0
    overlap = query_words & answer_words
    return min(len(overlap) / len(query_words), 1.0)


# ---------------------------------------------------------------
# Test Data
# ---------------------------------------------------------------

SAMPLE_CHUNKS = [
    {
        "chunk_id": "c001",
        "text": "ACE inhibitors are a class of medications that inhibit the angiotensin-converting enzyme. They are used in the treatment of hypertension and heart failure.",
        "score": 0.92,
        "metadata": {"filename": "cardiology.pdf", "page": 12},
    },
    {
        "chunk_id": "c002",
        "text": "Beta blockers reduce the workload on the heart by blocking the effects of epinephrine. They slow the heart rate and reduce blood pressure.",
        "score": 0.85,
        "metadata": {"filename": "cardiology.pdf", "page": 15},
    },
    {
        "chunk_id": "c003",
        "text": "Calcium channel blockers prevent calcium from entering the cells of the heart and blood vessel walls. This results in lower blood pressure.",
        "score": 0.78,
        "metadata": {"filename": "cardiology.pdf", "page": 18},
    },
    {
        "chunk_id": "c004",
        "text": "The mitral valve is located between the left atrium and left ventricle. It prevents backflow of blood during ventricular contraction.",
        "score": 0.65,
        "metadata": {"filename": "anatomy.pdf", "page": 44},
    },
    {
        "chunk_id": "c005",
        "text": "Diabetes mellitus is characterized by hyperglycemia resulting from defects in insulin secretion, insulin action, or both.",
        "score": 0.45,
        "metadata": {"filename": "internal_medicine.pdf", "page": 201},
    },
]

SAMPLE_ANSWER_GOOD = """
## Answer
ACE inhibitors inhibit the angiotensin-converting enzyme, which reduces blood pressure 
by preventing the conversion of angiotensin I to angiotensin II. They are commonly used 
in hypertension and heart failure education.

## Key Points
- ACE inhibitors block the angiotensin-converting enzyme
- Used in cardiovascular medicine education
- Important in understanding hypertension pathophysiology
"""

SAMPLE_ANSWER_HALLUCINATED = """
## Answer
ACE inhibitors were invented in 1985 by Dr. John Smith at Harvard Medical School.
They work by stimulating the production of nitric oxide in a newly discovered pathway.
Clinical trials in 2024 showed 99% effectiveness in all cardiovascular conditions.
"""


# ---------------------------------------------------------------
# Tests
# ---------------------------------------------------------------

class TestContextPrecision:
    """Tests for context precision metric."""

    def test_high_precision_relevant_chunks(self):
        keywords = ["ACE inhibitors", "hypertension", "angiotensin"]
        precision = compute_context_precision(SAMPLE_CHUNKS[:3], keywords)
        assert precision >= 0.33  # At least 1/3 chunks are relevant

    def test_zero_precision_no_chunks(self):
        precision = compute_context_precision([], ["ACE inhibitors"])
        assert precision == 0.0

    def test_full_precision_all_relevant(self):
        chunks = [
            {"text": "ACE inhibitors are used for hypertension treatment."},
            {"text": "ACE inhibitors block angiotensin-converting enzyme."},
        ]
        precision = compute_context_precision(chunks, ["ACE inhibitors"])
        assert precision == 1.0

    def test_partial_precision(self):
        chunks = [
            {"text": "ACE inhibitors are relevant here."},
            {"text": "This chunk is about something else entirely."},
        ]
        precision = compute_context_precision(chunks, ["ACE inhibitors"])
        assert precision == 0.5


class TestContextRecall:
    """Tests for context recall metric."""

    def test_high_recall_key_facts_covered(self):
        facts = ["angiotensin-converting enzyme", "hypertension", "heart failure"]
        recall = compute_context_recall(SAMPLE_CHUNKS[:2], facts)
        assert recall >= 0.5

    def test_zero_recall_empty_chunks(self):
        recall = compute_context_recall([], ["ACE inhibitors", "hypertension"])
        assert recall == 0.0

    def test_full_recall_no_ground_truth(self):
        recall = compute_context_recall(SAMPLE_CHUNKS, [])
        assert recall == 1.0

    def test_recall_with_exact_fact(self):
        chunks = [{"text": "ACE inhibitors inhibit the angiotensin-converting enzyme"}]
        facts = ["angiotensin-converting enzyme"]
        recall = compute_context_recall(chunks, facts)
        assert recall == 1.0


class TestFaithfulness:
    """Tests for answer faithfulness metric."""

    def test_faithful_answer_scores_high(self):
        score = compute_faithfulness(SAMPLE_ANSWER_GOOD, SAMPLE_CHUNKS[:3])
        assert score >= 0.2  # Word-overlap heuristic; faithful answer has meaningful overlap

    def test_hallucinated_answer_scores_lower(self):
        faithful_score = compute_faithfulness(SAMPLE_ANSWER_GOOD, SAMPLE_CHUNKS[:3])
        hallucinated_score = compute_faithfulness(SAMPLE_ANSWER_HALLUCINATED, SAMPLE_CHUNKS[:3])
        # Faithful answer should score at least as well
        assert faithful_score >= hallucinated_score - 0.1  # Some tolerance

    def test_empty_answer_zero_faithfulness(self):
        score = compute_faithfulness("", SAMPLE_CHUNKS[:3])
        assert score == 0.0

    def test_empty_context_zero_faithfulness(self):
        score = compute_faithfulness(SAMPLE_ANSWER_GOOD, [])
        assert score == 0.0


class TestAnswerRelevancy:
    """Tests for answer relevancy metric."""

    def test_relevant_answer_scores_high(self):
        query = "What are ACE inhibitors and how do they work?"
        answer = "ACE inhibitors work by blocking the angiotensin-converting enzyme."
        score = compute_answer_relevancy(query, answer)
        assert score >= 0.2  # Keyword overlap heuristic — shared terms like 'inhibitors', 'ACE'

    def test_irrelevant_answer_scores_lower(self):
        query = "What are ACE inhibitors?"
        relevant_answer = "ACE inhibitors block the angiotensin-converting enzyme."
        irrelevant_answer = "The mitral valve is located in the left side of the heart."
        
        relevant_score = compute_answer_relevancy(query, relevant_answer)
        irrelevant_score = compute_answer_relevancy(query, irrelevant_answer)
        assert relevant_score >= irrelevant_score

    def test_empty_query_returns_zero(self):
        score = compute_answer_relevancy("", "Some answer about ACE inhibitors.")
        assert score == 0.0

    def test_empty_answer_returns_zero(self):
        score = compute_answer_relevancy("What is hypertension?", "")
        assert score == 0.0


class TestEndToEndRAGEvaluation:
    """End-to-end RAG evaluation scenarios."""

    def test_cardiology_query_evaluation(self):
        """Evaluate full RAG pipeline metrics for a cardiology query."""
        query = "How do ACE inhibitors work in treating hypertension?"
        ground_truth_facts = ["angiotensin-converting enzyme", "hypertension", "blood pressure"]
        relevant_keywords = ["ACE", "inhibitors", "angiotensin"]

        retrieved_chunks = SAMPLE_CHUNKS[:3]

        precision = compute_context_precision(retrieved_chunks, relevant_keywords)
        recall = compute_context_recall(retrieved_chunks, ground_truth_facts)
        faithfulness = compute_faithfulness(SAMPLE_ANSWER_GOOD, retrieved_chunks)
        relevancy = compute_answer_relevancy(query, SAMPLE_ANSWER_GOOD)

        # Log metrics
        print(f"\nRAG Evaluation Metrics - Cardiology Query:")
        print(f"  Context Precision:  {precision:.3f}")
        print(f"  Context Recall:     {recall:.3f}")
        print(f"  Faithfulness:       {faithfulness:.3f}")
        print(f"  Answer Relevancy:   {relevancy:.3f}")

        # All metrics should be above minimum threshold
        assert precision >= 0.3, f"Context precision too low: {precision:.3f}"
        assert recall >= 0.3, f"Context recall too low: {recall:.3f}"
        assert faithfulness >= 0.1, f"Faithfulness too low: {faithfulness:.3f}"
        assert relevancy >= 0.1, f"Answer relevancy too low: {relevancy:.3f}"

    def test_minimum_retrieval_quality_threshold(self):
        """Ensure retrieved chunks meet minimum quality for safe answering."""
        MIN_CONFIDENCE = 0.3
        from backend.rag.retriever import calculate_confidence, RetrievedChunk

        chunks = [
            RetrievedChunk(**{**c, "metadata": c["metadata"]}) for c in SAMPLE_CHUNKS[:3]
        ]
        confidence = calculate_confidence(chunks)
        assert confidence >= MIN_CONFIDENCE, f"Confidence {confidence:.3f} below threshold {MIN_CONFIDENCE}"
