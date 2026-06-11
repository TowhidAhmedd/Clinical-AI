# Architecture Guide

## System Overview

The Clinical RAG Assistant is a multi-tier application:

```
┌─────────────────────────────────────────────────────────┐
│                    USER BROWSER                         │
└───────────────────────┬─────────────────────────────────┘
                        │ HTTP
┌───────────────────────▼─────────────────────────────────┐
│            STREAMLIT FRONTEND (Port 8501)               │
│   • Login / JWT management                             │
│   • Document upload UI                                  │
│   • Chat interface with citation display               │
└───────────────────────┬─────────────────────────────────┘
                        │ REST API (JSON + multipart)
┌───────────────────────▼─────────────────────────────────┐
│            FASTAPI BACKEND (Port 8000)                  │
│                                                         │
│  /auth/login  → JWT token generation                    │
│  /chat/query  → Full RAG pipeline                       │
│  /documents/* → Upload / list / delete                  │
│  /health      → Health check                            │
│                                                         │
│  Middleware: CORS, Rate Limiting, JWT Validation        │
└───────┬───────────────────────┬─────────────────────────┘
        │                       │
        ▼                       ▼
┌───────────────┐   ┌──────────────────────────────────────┐
│   PINECONE    │   │        LANGGRAPH WORKFLOW            │
│ Vector Store  │   │                                      │
│               │   │  Query Router Agent                  │
│ - Chunk store │◄──│    ↓                                 │
│ - Cosine sim  │   │  Safety Agent                        │
│ - 384-dim     │   │    ↓                                 │
└───────────────┘   │  Retrieval Agent ──► Pinecone        │
                    │    ↓                                  │
┌───────────────┐   │  Answer Agent ──► Groq LLM           │
│  GROQ LLM API │◄──│    ↓                                 │
│               │   │  Build Final Response                │
│ - Llama 3     │   └──────────────────────────────────────┘
│ - Mixtral     │
│ - Gemma       │
└───────────────┘

┌───────────────────────────────────────────────────────────┐
│                  LOCAL PROCESSES                          │
│  • BAAI/bge-small-en-v1.5 embeddings (CPU)              │
│  • FlashRank reranking (CPU)                             │
│  • PDF/DOCX/TXT text extraction                         │
└───────────────────────────────────────────────────────────┘
```

## RAG Pipeline Detail

```
Document Upload
     │
     ▼ validate (type + size)
     │
     ▼ extract text (pypdf / python-docx / open())
     │
     ▼ clean text (whitespace, non-printable chars)
     │
     ▼ chunk (RecursiveCharacterTextSplitter, 512 chars, 64 overlap)
     │
     ▼ embed (BAAI/bge-small-en-v1.5, 384-dim, L2-normalized)
     │
     ▼ upsert → Pinecone (batch 100)
     │
   Indexed ✓

Query Time
     │
     ▼ embed query (same model)
     │
     ▼ retrieve top-15 from Pinecone (cosine similarity)
     │
     ▼ rerank top-5 (FlashRank ms-marco-MiniLM-L-12-v2)
     │
     ▼ compress context (token-budget truncation)
     │
     ▼ generate citations (doc name + page + chunk_id + score)
     │
     ▼ calculate confidence (weighted avg of retrieval scores)
     │
   RetrievalResult { chunks, citations, context, confidence }
```

## Safety Architecture

```
                    ┌─────────────────────────────────┐
User Input ────────►│   Layer 1: Input Guardrails      │
                    │   (regex patterns, instant)      │
                    │   • Blocks diagnosis requests    │
                    │   • Blocks prescription requests │
                    │   • Blocks prompt injection      │
                    │   • Blocks role manipulation     │
                    └──────────┬──────────────────────┘
                               │ SAFE
                    ┌──────────▼──────────────────────┐
                    │   Layer 2: Safety Agent (LLM)    │
                    │   (Groq inference, ~1-2s)        │
                    │   • Detects clinical advice      │
                    │   • Detects treatment planning   │
                    │   • Fails OPEN on LLM error      │
                    └──────────┬──────────────────────┘
                               │ SAFE
                    ┌──────────▼──────────────────────┐
                    │   Layer 3: Retrieval Guardrails  │
                    │   • Checks chunks exist          │
                    │   • Checks relevance scores      │
                    └──────────┬──────────────────────┘
                               │ SUFFICIENT CONTEXT
                    ┌──────────▼──────────────────────┐
                    │   Layer 4: Output Guardrails     │
                    │   (regex on generated answer)    │
                    │   • Blocks dosage in output      │
                    │   • Blocks diagnosis in output   │
                    └──────────┬──────────────────────┘
                               │ SAFE
                          Final Answer +
                          Educational Disclaimer
```

## LangGraph State Machine

```python
GraphState = {
    query, user_id, doc_filter,    # Input
    query_type,                     # Router output
    retrieval_result, context,      # Retrieval output
    is_safe, safety_reason,         # Safety flags
    answer, citations, confidence,  # Answer output
    final_response, error           # Final output
}

Edges:
  query_router → should_continue?
    → SAFE    → safety_check
    → UNSAFE  → build_final_response

  safety_check → should_continue?
    → SAFE    → retrieval
    → UNSAFE  → build_final_response

  retrieval → should_continue?
    → SAFE    → answer
    → UNSAFE  → build_final_response

  answer → build_final_response → END
```
